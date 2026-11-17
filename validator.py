import inspect
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder

# Supported scikit-learn transformers
SUPPORTED_TRANSFORMERS = {
    "SimpleImputer": SimpleImputer,
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
    "RobustScaler": RobustScaler,
    "OneHotEncoder": OneHotEncoder,
    "OrdinalEncoder": OrdinalEncoder,
}

NUMERIC_TRANSFORMERS = {"StandardScaler", "MinMaxScaler", "RobustScaler"}
CATEGORICAL_TRANSFORMERS = {"OneHotEncoder", "OrdinalEncoder"}

NUMERIC_POLARS_TYPES = {
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
}

CATEGORICAL_POLARS_TYPES = {
    pl.String, pl.Utf8, pl.Categorical, pl.Enum, pl.Boolean,
}


@dataclass
class ValidationReport:
    """Detailed log of validations and automated corrections applied to the plan."""
    is_valid: bool = True
    corrections: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)

    def log_correction(self, msg: str):
        self.corrections.append(f"[Validator Correction] {msg}")

    def log_warning(self, msg: str):
        self.warnings.append(f"[Validator Warning] {msg}")

    def log_info(self, msg: str):
        self.info.append(f"[Validator Info] {msg}")

    def display(self):
        print("\n" + "=" * 60)
        print("PLAN VALIDATOR")
        print("=" * 60)
        if self.info:
            for item in self.info:
                print(f"ℹ️  {item}")
        if self.corrections:
            for item in self.corrections:
                print(f"🔧 {item}")
        if self.warnings:
            for item in self.warnings:
                print(f"⚠️  {item}")
        if not self.corrections and not self.warnings:
            print("All plan specifications are within valid Scikit-Learn bounds.")
        print("=" * 60 + "\n")


def _filter_valid_params(transformer_cls, params: Dict[str, Any]) -> Dict[str, Any]:
    """Filter parameters dictionary so only valid constructor args are preserved."""
    valid_args = set(inspect.signature(transformer_cls.__init__).parameters.keys()) - {'self'}
    return {k: v for k, v in params.items() if k in valid_args}


def validate_and_fix_plan(
    analysis,
    df: pl.DataFrame,
    target_column: Optional[str] = None,
    drop_columns: Optional[List[str]] = None,
    auto_impute_nulls: bool = True
) -> Tuple[Any, ValidationReport]:
    """
    Validates and automatically repairs an Analysis plan to ensure strictly valid
    execution within Scikit-Learn and Polars data boundaries.

    Validations & Fixes:
    1. Removes target and drop_columns from feature transformations.
    2. Drops non-existent columns (hallucinated column names).
    3. Remaps 'LabelEncoder' to 'OrdinalEncoder' (since LabelEncoder breaks ColumnTransformer).
    4. Enforces data type compatibility:
       - Scalers only applied to numeric columns.
       - Encoders only applied to categorical/discrete columns.
       - SimpleImputer strategies adjusted for non-numeric columns ('mean'/'median' -> 'most_frequent').
    5. Sets safe scikit-learn defaults:
       - OneHotEncoder: sparse_output=False, handle_unknown='ignore'.
       - OrdinalEncoder: handle_unknown='use_encoded_value', unknown_value=-1.
    6. Sanitizes parameter dictionaries to strip invalid kwargs.
    7. Optionally adds SimpleImputer for columns containing null values without an assigned imputer.
    """
    report = ValidationReport()
    report.log_info(f"Validating plan against DataFrame ({df.width} columns, {df.height} rows)")

    available_cols = set(df.columns)
    excluded_cols: Set[str] = set()

    # Determine columns to exclude
    if target_column:
        if isinstance(target_column, list):
            excluded_cols.update(target_column)
        else:
            excluded_cols.add(target_column)
        report.log_info(f"Target column(s) excluded from feature transforms: {excluded_cols}")

    if drop_columns:
        excluded_cols.update(drop_columns)
        report.log_info(f"Configured drop columns excluded from feature transforms: {drop_columns}")

    # Inspect schema types
    numeric_cols = {col for col, dtype in df.schema.items() if dtype in NUMERIC_POLARS_TYPES}
    categorical_cols = {col for col, dtype in df.schema.items() if dtype in CATEGORICAL_POLARS_TYPES}
    null_counts = {col: df[col].null_count() for col in df.columns if col not in excluded_cols}

    raw_transformations = getattr(analysis, "transformations", [])
    valid_transformations = []
    columns_with_imputer: Set[str] = set()

    for transform in raw_transformations:
        t_name = transform.transformer
        params = dict(transform.parameters) if transform.parameters else {}
        cols = list(transform.columns)

        # 1. Remap LabelEncoder to OrdinalEncoder
        if t_name == "LabelEncoder":
            report.log_correction(
                f"Replaced 'LabelEncoder' with 'OrdinalEncoder' on {cols}. "
                "LabelEncoder is only for 1D targets and causes errors in ColumnTransformer."
            )
            t_name = "OrdinalEncoder"
            transform.transformer = "OrdinalEncoder"

        if t_name not in SUPPORTED_TRANSFORMERS:
            report.log_warning(f"Unsupported transformer '{t_name}' requested. Skipping.")
            continue

        transformer_cls = SUPPORTED_TRANSFORMERS[t_name]

        # 2. Exclude non-existent, target, and dropped columns
        valid_cols = []
        for c in cols:
            if c in excluded_cols:
                report.log_correction(f"Removed excluded column '{c}' from {t_name}.")
                continue
            if c not in available_cols:
                report.log_correction(f"Removed non-existent column '{c}' from {t_name}.")
                continue
            valid_cols.append(c)

        if not valid_cols:
            report.log_warning(f"No valid columns remaining for {t_name}. Step dropped.")
            continue

        # 3. Type compatibility checks
        if t_name in NUMERIC_TRANSFORMERS:
            type_checked_cols = []
            for c in valid_cols:
                if c not in numeric_cols:
                    report.log_correction(
                        f"Removed non-numeric column '{c}' ({df.schema.get(c)}) from numeric scaler {t_name}."
                    )
                else:
                    type_checked_cols.append(c)
            valid_cols = type_checked_cols

        elif t_name in CATEGORICAL_TRANSFORMERS:
            type_checked_cols = []
            for c in valid_cols:
                # If column is high precision float, warn and disallow encoding
                if df.schema.get(c) in {pl.Float32, pl.Float64}:
                    report.log_correction(
                        f"Removed continuous float column '{c}' from categorical encoder {t_name}."
                    )
                else:
                    type_checked_cols.append(c)
            valid_cols = type_checked_cols

        elif t_name == "SimpleImputer":
            # Check strategy compatibility
            strategy = params.get("strategy", "mean")
            if strategy == "mode":
                report.log_correction("Converted SimpleImputer strategy 'mode' to sklearn's 'most_frequent'.")
                params["strategy"] = "most_frequent"
                strategy = "most_frequent"

            # Check if any columns are non-numeric with mean/median
            non_num_in_imputer = [c for c in valid_cols if c not in numeric_cols]
            if non_num_in_imputer and strategy in {"mean", "median"}:
                report.log_correction(
                    f"SimpleImputer strategy '{strategy}' is invalid for non-numeric columns {non_num_in_imputer}. "
                    "Splitting or fixing strategy to 'most_frequent'."
                )
                params["strategy"] = "most_frequent"

            if params.get("strategy") == "constant" and "fill_value" not in params:
                default_fill = "missing" if any(c in categorical_cols for c in valid_cols) else 0
                params["fill_value"] = default_fill
                report.log_correction(f"Added default fill_value='{default_fill}' for constant SimpleImputer.")

            columns_with_imputer.update(valid_cols)

        if not valid_cols:
            continue

        # 4. Enforce robust Scikit-Learn defaults
        if t_name == "OneHotEncoder":
            if not params.get("sparse_output", False):
                params["sparse_output"] = False
                params.pop("sparse", None)
            if params.get("handle_unknown") != "ignore":
                params["handle_unknown"] = "ignore"
                report.log_info(f"Configured OneHotEncoder with handle_unknown='ignore' and sparse_output=False.")

        elif t_name == "OrdinalEncoder":
            if params.get("handle_unknown") != "use_encoded_value":
                params["handle_unknown"] = "use_encoded_value"
                params["unknown_value"] = -1
                report.log_info("Configured OrdinalEncoder with handle_unknown='use_encoded_value' and unknown_value=-1.")

        # 5. Sanitize parameters
        clean_params = _filter_valid_params(transformer_cls, params)
        removed_params = set(params.keys()) - set(clean_params.keys())
        if removed_params:
            report.log_correction(f"Removed invalid parameter kwargs {removed_params} for {t_name}.")

        transform.columns = valid_cols
        transform.parameters = clean_params
        valid_transformations.append(transform)

    # 6. Auto-impute missing values if columns have nulls and no imputer assigned
    if auto_impute_nulls:
        unhandled_nulls_num = [
            c for c, count in null_counts.items()
            if count > 0 and c in numeric_cols and c not in columns_with_imputer
        ]
        unhandled_nulls_cat = [
            c for c, count in null_counts.items()
            if count > 0 and c not in numeric_cols and c not in columns_with_imputer
        ]

        # Use the Transformation class from analysis module or create dict-compatible object
        TransformationCls = type(raw_transformations[0]) if raw_transformations else None

        if unhandled_nulls_num and TransformationCls:
            report.log_correction(
                f"Auto-added SimpleImputer(strategy='median') for numeric columns with nulls: {unhandled_nulls_num}"
            )
            valid_transformations.insert(
                0,
                TransformationCls(
                    columns=unhandled_nulls_num,
                    transformer="SimpleImputer",
                    parameters={"strategy": "median"}
                )
            )

        if unhandled_nulls_cat and TransformationCls:
            report.log_correction(
                f"Auto-added SimpleImputer(strategy='most_frequent') for categorical columns with nulls: {unhandled_nulls_cat}"
            )
            valid_transformations.insert(
                0,
                TransformationCls(
                    columns=unhandled_nulls_cat,
                    transformer="SimpleImputer",
                    parameters={"strategy": "most_frequent"}
                )
            )

    analysis.transformations = valid_transformations
    report.is_valid = len(valid_transformations) > 0
    return analysis, report
