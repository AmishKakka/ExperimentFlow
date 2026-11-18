import json
from pprint import pprint
import polars as pl
import numpy as np
from data_ingestion import load_yaml_file, get_data_info, resolve_user_initial_query
from data_analysis import get_analysis, get_concrete_plan, Analysis, InitialAnalysis
from data_pipeline import transformation_pipeline
from experiment_tracking import configure_mlflow, execute_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    accuracy_score,
    f1_score,
    silhouette_score,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

REGRESSOR_MAP = {
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "SVR": SVR,
}

CLASSIFIER_MAP = {
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "SVC": SVC,
    "KNeighborsClassifier": KNeighborsClassifier,
}

CLUSTERER_MAP = {
    "KMeans": KMeans,
    "DBSCAN": DBSCAN,
    "AgglomerativeClustering": AgglomerativeClustering,
}


def main():
    # Data ingestion and schema configuration
    df, schema, data_config = load_yaml_file(filepath="data.yaml")
    df_schema = get_data_info(df)

    print("\n" + "=" * 65)
    print("LLM-POWERED MLOPS PIPELINE: ")
    print("=" * 65)
    query = data_config.get("user_initial_query") if isinstance(data_config, dict) else None
    user_initial_query = resolve_user_initial_query(data_config)
    if not isinstance(query, str) or not query.strip():
        print(
            "\n[Step 1] 'user_initial_query' is missing or empty in data.yaml. "
            "Using the default query."
        )
    else:
        print("\n[Step 1] Using user_initial_query from data.yaml.")
    print(user_initial_query)

    # Initial assessment
    initial_analysis: InitialAnalysis = get_analysis(
        df_head=df.head(5),
        schema=df_schema,
        user_query=user_initial_query
    )

    print("\n" + "-" * 65)
    print("Gemini's Dataset understanding:")
    print("-" * 65)
    print(f"Overview:\n{initial_analysis.overview}\n")
    print(f"Recommended Task Type : {initial_analysis.task_type.upper()}")
    if initial_analysis.target_column:
        print(f"Recommended Target    : {initial_analysis.target_column}")
    if initial_analysis.candidate_target_columns:
        print(f"Candidate Targets     : {', '.join(initial_analysis.candidate_target_columns)}")
    print(f"\nReasoning & Opportunities:\n{initial_analysis.reasoning}\n")
    if initial_analysis.suggested_models:
        print(f"Candidate Models      : {', '.join(initial_analysis.suggested_models)}")
    print("-" * 65)

    # User's query for final concrete plan
    user_final_query = input(
        "\n[Step 2] Enter your final concrete plan / preferences\n"
        "(e.g., confirm or change target, task type, or specific transformers/models;\n"
        " press Enter to proceed with the recommendation): "
    ).strip()

    if not user_final_query:
        user_final_query = (
            f"Proceed with recommended {initial_analysis.task_type} "
            f"targeting {initial_analysis.target_column or 'unsupervised features'}."
        )

    # Generate Concrete Plan
    print("\nGenerating concrete execution plan...")
    concrete_plan: Analysis = get_concrete_plan(
        df_head=df.head(5),
        schema=df_schema,
        initial_query=user_initial_query,
        final_query=user_final_query,
        initial_analysis=initial_analysis
    )

    print("\n" + "-" * 65)
    print(f"Concrete Plan Generated: Task={concrete_plan.task_type.upper()}, Target={concrete_plan.target_column}")
    print(f"Transformations Proposed: {len(concrete_plan.transformations)}")
    for t in concrete_plan.transformations:
        print(f"  • {t.transformer} on {t.columns} (params: {t.parameters})")
    print("-" * 65)

    # Transformation Pipeline with Validator & pure Scikit-Learn execution
    target_col = concrete_plan.target_column
    if not target_col and schema.get("target_column") and concrete_plan.task_type != "clustering":
        target_cfg = schema.get("target_column")
        target_col = target_cfg[0] if isinstance(target_cfg, list) else target_cfg

    drop_cols = schema.get("drop_columns", [])

    df_transformed, preprocessor, validation_report = transformation_pipeline(
        analysis=concrete_plan,
        df=df,
        target_column=target_col,
        drop_columns=drop_cols
    )

    # Model Training & Evaluation based on Task Type
    task_type = concrete_plan.task_type
    test_size = schema.get("test_split", 0.2)
    random_state = 42
    configure_mlflow()
    shared_config = {
        "user_initial_query": user_initial_query,
        "user_final_query": user_final_query,
        "task_type": task_type,
        "target_column": target_col,
        "test_size": test_size,
        "random_state": random_state,
        "transformations": [
            step.model_dump() if hasattr(step, "model_dump") else step
            for step in (concrete_plan.transformations or [])
        ],
        "validator_corrections": list(validation_report.corrections),
        "validator_warnings": list(validation_report.warnings),
    }

    if task_type == "regression":
        if not target_col or target_col not in df_transformed.columns:
            raise ValueError(f"Target column '{target_col}' not found in transformed data.")

        X = df_transformed.drop(target_col)
        y = df_transformed[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        print("\nTraining Suggested Regression Models...")
        suggested = concrete_plan.suggested_regression_models or []
        if not suggested:
            print("No regression models specified in plan. Falling back to LinearRegression and RandomForestRegressor.")
            from data_analysis import Regressor
            suggested = [
                Regressor(model_name="LinearRegression"),
                Regressor(model_name="RandomForestRegressor", hyperparameters={"n_estimators": 50, "max_depth": 10})
            ]

        def fit_regression(model):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            return {
                "r2": float(r2_score(y_test, preds)),
                "mae": float(mean_absolute_error(y_test, preds)),
            }

        for model_info in suggested:
            m_name = model_info.model_name
            m_cls = REGRESSOR_MAP.get(m_name)
            if not m_cls:
                print(f"Model '{m_name}' not recognized in REGRESSOR_MAP. Skipping.")
                continue
            execute_model(
                model_name=m_name,
                model_cls=m_cls,
                hyperparameters=model_info.hyperparameters,
                shared_config=shared_config,
                fit_and_score=fit_regression,
                format_success=lambda metrics, name=m_name: (
                    f"  ✅ {name} | R² Score: {metrics['r2']:7.4f} | MAE: {metrics['mae']:7.4f}"
                ),
                input_frame=X_train,
            )

    elif task_type == "classification":
        if not target_col or target_col not in df_transformed.columns:
            raise ValueError(f"Target column '{target_col}' not found in transformed data.")

        X = df_transformed.drop(target_col)
        y = df_transformed[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        print("\nTraining Suggested Classification Models...")
        suggested = concrete_plan.suggested_classification_models or []
        if not suggested:
            from data_analysis import Classifier
            suggested = [
                Classifier(model_name="LogisticRegression"),
                Classifier(model_name="RandomForestClassifier", hyperparameters={"n_estimators": 50})
            ]

        def fit_classification(model):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            return {
                "accuracy": float(accuracy_score(y_test, preds)),
                "f1_weighted": float(f1_score(y_test, preds, average="weighted")),
            }

        for model_info in suggested:
            m_name = model_info.model_name
            m_cls = CLASSIFIER_MAP.get(m_name)
            if not m_cls:
                print(f"Model '{m_name}' not recognized in CLASSIFIER_MAP. Skipping.")
                continue
            execute_model(
                model_name=m_name,
                model_cls=m_cls,
                hyperparameters=model_info.hyperparameters,
                shared_config=shared_config,
                fit_and_score=fit_classification,
                format_success=lambda metrics, name=m_name: (
                    f"  ✅ {name} | Accuracy: {metrics['accuracy']:7.4f} | "
                    f"F1 (weighted): {metrics['f1_weighted']:7.4f}"
                ),
                input_frame=X_train,
            )

    elif task_type == "clustering":
        X = df_transformed
        if target_col and target_col in X.columns:
            X = X.drop(target_col)
        X_mat = X.to_numpy()

        print("\nTraining Suggested Clustering Models...")
        suggested = concrete_plan.suggested_clustering_models or []
        if not suggested:
            from data_analysis import Clusterer
            suggested = [Clusterer(model_name="KMeans", hyperparameters={"n_clusters": 3, "random_state": 42})]

        def fit_clustering(model):
            labels = model.fit_predict(X_mat)
            unique_labels = set(labels) - {-1}
            metrics = {"n_clusters": int(len(unique_labels))}
            if len(unique_labels) > 1:
                metrics["silhouette"] = float(silhouette_score(X_mat, labels)) # pyright: ignore[reportArgumentType]
            return metrics

        for model_info in suggested:
            m_name = model_info.model_name
            m_cls = CLUSTERER_MAP.get(m_name)
            if not m_cls:
                print(f"Model '{m_name}' not recognized in CLUSTERER_MAP. Skipping.")
                continue
            execute_model(
                model_name=m_name,
                model_cls=m_cls,
                hyperparameters=model_info.hyperparameters,
                shared_config=shared_config,
                fit_and_score=fit_clustering,
                format_success=lambda metrics, name=m_name: (
                    f"  ✅ {name} | Silhouette Score: {metrics['silhouette']:7.4f} | "
                    f"Clusters: {int(metrics['n_clusters'])}"
                    if "silhouette" in metrics
                    else f"  ✅ {name} | Fit completed (Clusters formed: {int(metrics['n_clusters'])})"
                ),
                input_frame=X_mat,
            )
    print("\nPipeline execution successfully finished!")
    print("Each model run, its configuration, and the fitted model is in ./mlruns (mlflow.db and artifacts/).")


if __name__ == "__main__":
    main()