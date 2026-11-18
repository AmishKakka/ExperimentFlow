import inspect
import json
import os
import tempfile
import threading
import time
from pathlib import Path
import mlflow
import numpy as np

# These estimators print tree, iteration, or optimizer progress when verbose is set.
_VERBOSE_PROGRESS_MODELS = {
    "RandomForestRegressor",
    "RandomForestClassifier",
    "GradientBoostingRegressor",
    "GradientBoostingClassifier",
    "SVC",
    "SVR",
    "KMeans",
}


def configure_mlflow(tracking_dir="mlruns", experiment_name="mlops_pipeline"):
    """Point MLflow at a local SQLite store and open the project experiment.
    MLflow's run metadata lives in mlflow.db and model artifacts live under artifacts/.
    """
    root = Path(tracking_dir).resolve()
    artifact_root = root / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    db_path = root / "mlflow.db"
    tracking_uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.MlflowClient()
    if client.get_experiment_by_name(experiment_name) is None:
        client.create_experiment(
            experiment_name,
            artifact_location=artifact_root.as_uri(),
        )
    experiment = mlflow.set_experiment(experiment_name)
    name = getattr(experiment, "name", experiment_name)
    print("\n" + "-" * 65)
    print("MLflow")
    print("-" * 65)
    print(f"Experiment : {name}")
    print(f"Tracking   : {tracking_uri}")
    print(f"Artifacts  : {artifact_root}")
    print("-" * 65)
    return tracking_uri


def apply_live_verbosity(model_name, model_cls, hyperparameters):
    """Turn on estimator progress logs without overriding a planned verbose value."""
    params = dict(hyperparameters or {})
    if model_name not in _VERBOSE_PROGRESS_MODELS or "verbose" in params:
        return params
    try:
        signature = inspect.signature(model_cls.__init__)
    except (TypeError, ValueError):
        return params
    verbose_param = signature.parameters.get("verbose")
    if verbose_param is None:
        return params
    params["verbose"] = True if isinstance(verbose_param.default, bool) else 1
    return params


def run_with_live_progress(label, action):
    """Run action and print a flushed elapsed-time line until it returns."""
    stop = threading.Event()
    start = time.perf_counter()
    print(f"\n  ▶ Training {label}", flush=True)
    print(f"    [progress] {label} | elapsed   0.0s | status=fitting", flush=True)

    def heartbeat():
        while not stop.wait(5.0):
            elapsed = time.perf_counter() - start
            print(
                f"    [progress] {label} | elapsed {elapsed:6.1f}s | status=fitting",
                flush=True,
            )
    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    try:
        result = action()
    except Exception:
        stop.set()
        worker.join()
        elapsed = time.perf_counter() - start
        print(
            f"    [progress] {label} | elapsed {elapsed:6.1f}s | status=failed",
            flush=True,
        )
        raise
    stop.set()
    worker.join()
    elapsed = time.perf_counter() - start
    print(
        f"    [progress] {label} | elapsed {elapsed:6.1f}s | status=complete",
        flush=True,
    )
    return result, elapsed

def _param_value(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if value is None:
        return "null"
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text

def _flatten_params(values, prefix=""):
    flat = {}
    for key, value in dict(values or {}).items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten_params(value, prefix=f"{name}."))
            continue
        flat[name.replace("/", "_")[:250]] = _param_value(value)
    return flat

def _json_ready(value):
    return json.loads(json.dumps(value, default=str))

def _input_example(frame):
    if frame is None:
        return None
    try:
        head = getattr(frame, "head", None)
        to_pandas = getattr(frame, "to_pandas", None)
        if callable(head) and callable(to_pandas):
            return head(2).to_pandas() # pyright: ignore[reportAttributeAccessIssue]
        if callable(head):
            sample = head(2)
            to_numpy = getattr(sample, "to_numpy", None)
            if callable(to_numpy):
                return np.asarray(to_numpy())
            return sample
        return np.asarray(frame)[:2]
    except Exception:
        return None

def _silence_training_verbose(model):
    """Drop the progress flag before the fitted model is saved."""
    current = getattr(model, "verbose", None)
    if current in (None, 0, False):
        return
    restored = False if isinstance(current, bool) else 0
    try:
        model.set_params(verbose=restored)
    except Exception:
        try:
            model.verbose = restored
        except Exception:
            return

def _skops_trusted_types(model):
    """
    Current MLflow saves estimators with skops and refuses internal types such
    as decision-tree nodes until the caller names them. The object has not
    been loaded from an outside file.
    """
    try:
        import skops.io
    except ImportError:
        return None
    fd, path = tempfile.mkstemp(suffix=".skops")
    os.close(fd)
    try:
        skops.io.dump(model, path)
        trusted = list(skops.io.get_untrusted_types(file=path))
        return trusted or None
    except Exception:
        return None
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

def _log_sklearn_model(model, input_example):
    signature = inspect.signature(mlflow.sklearn.log_model) # pyright: ignore[reportPrivateImportUsage]
    if "name" in signature.parameters:
        location_kw = "name"
    elif "artifact_path" in signature.parameters:
        location_kw = "artifact_path"
    else:
        location_kw = None

    base = {}
    if "skops_trusted_types" in signature.parameters:
        trusted = _skops_trusted_types(model)
        if trusted:
            base["skops_trusted_types"] = trusted

    attempts = []
    if input_example is not None and "input_example" in signature.parameters:
        attempts.append({**base, "input_example": input_example})
    attempts.append(dict(base))
    if "serialization_format" in signature.parameters:
        fallback = {"serialization_format": "cloudpickle"}
        if input_example is not None and "input_example" in signature.parameters:
            fallback["input_example"] = input_example
        attempts.append(fallback)

    last_error = None
    for extra in attempts:
        kwargs = dict(extra)
        if location_kw:
            kwargs[location_kw] = "model"
        try:
            if location_kw:
                mlflow.sklearn.log_model(model, **kwargs) # pyright: ignore[reportPrivateImportUsage]
            else:
                mlflow.sklearn.log_model(model, "model", **kwargs) # pyright: ignore[reportPrivateImportUsage]
            return
        except TypeError as exc:
            last_error = exc
            continue
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"Could not log the fitted model to MLflow: {last_error}")


def execute_model(
    *,
    model_name,
    model_cls,
    hyperparameters,
    shared_config,
    fit_and_score,
    format_success,
    input_frame=None,
):
    """
    Fit one candidate, print live progress, and on success log the run,
    its configuration, metrics, and the fitted model.
    """
    planned = dict(hyperparameters or {})
    runtime_params = apply_live_verbosity(model_name, model_cls, planned)
    task_type = shared_config.get("task_type") or "model"
    target_column = shared_config.get("target_column")
    run_config = {
        "model_name": model_name,
        "hyperparameters": planned,
        "training_verbose": runtime_params.get("verbose"),
        "user_initial_query": shared_config.get("user_initial_query"),
        "user_final_query": shared_config.get("user_final_query"),
        "task_type": task_type,
        "target_column": target_column,
        "test_size": shared_config.get("test_size"),
        "random_state": shared_config.get("random_state"),
        "transformations": shared_config.get("transformations") or [],
        "validator_corrections": shared_config.get("validator_corrections") or [],
        "validator_warnings": shared_config.get("validator_warnings") or [],
    }
    logged_params = {
        "model_name": model_name,
        "task_type": str(task_type),
        "target_column": "none" if target_column in (None, "") else str(target_column),
        "test_size": _param_value(shared_config.get("test_size")),
        "random_state": _param_value(shared_config.get("random_state")),
        "user_initial_query": _param_value(shared_config.get("user_initial_query")),
    }
    logged_params.update(_flatten_params(planned, prefix="hp."))

    print(f"    hyperparameters: {planned}", flush=True)
    try:
        with mlflow.start_run(run_name=f"{task_type}-{model_name}") as run:
            mlflow.log_params(logged_params)
            mlflow.set_tags(
                {
                    "task_type": str(task_type),
                    "model_name": model_name,
                    "target_column": logged_params["target_column"],
                }
            )
            mlflow.log_dict(_json_ready(run_config), "config/run_config.json")

            model = model_cls(**runtime_params)
            metrics, elapsed = run_with_live_progress(
                model_name,
                lambda: fit_and_score(model),
            )
            if "verbose" not in planned:
                _silence_training_verbose(model)
            metrics = dict(metrics)
            metrics["fit_seconds"] = round(float(elapsed), 4)
            numeric_metrics = {
                key: float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            mlflow.log_metrics(numeric_metrics)
            _log_sklearn_model(model, _input_example(input_frame))
            print(format_success(metrics), flush=True)
            print(f"     MLflow run {run.info.run_id} | model and config saved", flush=True)
            return run.info.run_id
    except Exception as exc:
        print(f"  ❌ Error training {model_name}: {exc}", flush=True)
        return None