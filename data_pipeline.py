from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple, Literal, cast
import polars as pl
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
    OrdinalEncoder,
)
from validator import validate_and_fix_plan, ValidationReport

SKLEARN_TRANSFORMER_REGISTRY = {
    "SimpleImputer": SimpleImputer,
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
    "RobustScaler": RobustScaler,
    "OneHotEncoder": OneHotEncoder,
    "OrdinalEncoder": OrdinalEncoder,
}


def build_sklearn_column_transformer(
    transformations,
    remainder: Literal["drop", "passthrough"] = "passthrough"
) -> ColumnTransformer:
    """
    Constructs an sklearn ColumnTransformer from a list of Transformation objects.
    
    Groups columns that share the exact same sequence of transformation steps into
    composite Pipeline pipelines (e.g. SimpleImputer -> StandardScaler).
    Enables native polars output via .set_output(transform='polars').
    """
    col_steps = defaultdict(list)

    for transform in transformations:
        t_name = transform.transformer
        params = dict(transform.parameters) if transform.parameters else {}
        transformer_cls = SKLEARN_TRANSFORMER_REGISTRY.get(t_name)
        if not transformer_cls:
            raise ValueError(f"Unsupported transformer '{t_name}'.")

        transformer_instance = transformer_cls(**params)
        for col in transform.columns:
            col_steps[col].append((t_name, transformer_instance))

    # Group columns that share identical pipeline step signatures
    pipeline_groups = defaultdict(list)
    for col, steps in col_steps.items():
        sig = tuple((name, repr(t.get_params())) for name, t in steps)
        pipeline_groups[sig].append((col, steps))

    transformers = []
    for i, (sig, col_step_list) in enumerate(pipeline_groups.items()):
        cols = [col for col, _ in col_step_list]
        steps = col_step_list[0][1]
        pipe_steps = [
            (f"{name.lower()}_{j}", transformer)
            for j, (name, transformer) in enumerate(steps)
        ]
        composite_pipe = Pipeline(pipe_steps)
        transformers.append((f"pipeline_group_{i}", composite_pipe, cols))

    column_transformer = ColumnTransformer(
        transformers=transformers,
        remainder=remainder
    )

    # Enable native Polars DataFrame output if supported by installed sklearn version
    try:
        set_output_func = getattr(column_transformer, "set_output", None)
        if callable(set_output_func):
            set_output_func(transform="polars")
    except Exception:
        pass

    return column_transformer


def transformation_pipeline(
    analysis,
    df: pl.DataFrame,
    target_column: Optional[str] = None,
    drop_columns: Optional[List[str]] = None,
) -> Tuple[pl.DataFrame, ColumnTransformer, ValidationReport]:
    """
    Validates the plan, constructs a pure scikit-learn ColumnTransformer,
    fits and transforms the features, and returns the transformed data along
    with the fitted preprocessor artifact and validation report.
    """
    print("\n🔍 Running Plan Validator...")
    validated_analysis, report = validate_and_fix_plan(
        analysis=analysis,
        df=df,
        target_column=target_column,
        drop_columns=drop_columns,
        auto_impute_nulls=True
    )
    report.display()

    # Separate features from target
    target_series = None
    feature_df = df
    if drop_columns:
        valid_drops = [c for c in drop_columns if c in feature_df.columns]
        if valid_drops:
            feature_df = feature_df.drop(valid_drops)

    if target_column:
        target_col_name = target_column[0] if isinstance(target_column, list) else target_column
        if target_col_name in feature_df.columns:
            target_series = feature_df[target_col_name]
            feature_df = feature_df.drop(target_col_name)

    print("Building Scikit-Learn ColumnTransformer...")
    preprocessor = build_sklearn_column_transformer(
        validated_analysis.transformations,
        remainder="passthrough"
    )

    print("Fitting and transforming features using Scikit-Learn...")
    transformed_output = preprocessor.fit_transform(cast(Any, feature_df))

    if isinstance(transformed_output, pl.DataFrame):
        features_transformed = transformed_output
    else:
        # Convert sparse matrix or raw array to a dense 2D NumPy array
        if hasattr(transformed_output, "toarray"):
            np_data = transformed_output.toarray() # pyright: ignore[reportAttributeAccessIssue]
        else:
            np_data = np.asarray(transformed_output)

        try:
            col_names = [str(name) for name in preprocessor.get_feature_names_out()]
        except Exception:
            col_names = [f"feature_{i}" for i in range(np_data.shape[1])]

        features_transformed = pl.from_numpy(np_data, schema=col_names)
    if target_series is not None:
        features_transformed = features_transformed.with_columns(target_series)

    print(f"✅ Data Transformation completed. Output shape: {features_transformed.shape}")
    return features_transformed, preprocessor, report