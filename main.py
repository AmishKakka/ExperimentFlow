import json
from pprint import pprint
import polars as pl
import numpy as np
from data_ingestion import load_yaml_file, get_data_info
from data_analysis import get_analysis, get_concrete_plan, Analysis, InitialAnalysis
from data_pipeline import transformation_pipeline, build_sklearn_column_transformer
from validator import validate_and_fix_plan
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
    df, schema = load_yaml_file(filepath="data.yaml")
    df_schema = get_data_info(df)

    # User's high-level intent query
    print("\n" + "=" * 65)
    print("LLM-POWERED MLOPS PIPELINE: ")
    print("=" * 65)
    user_initial_query = input(
        "\n[Step 1] What kind of training or output would you like to achieve with this dataset?\n"
        "(e.g., 'predict fare amount', 'classify high vs low tip', or 'explore options'): "
    ).strip()

    if not user_initial_query:
        user_initial_query = "Analyze the dataset, find suitable targets, and recommend the best ML task."

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

    if task_type == "regression":
        if not target_col or target_col not in df_transformed.columns:
            raise ValueError(f"Target column '{target_col}' not found in transformed data.")

        X = df_transformed.drop(target_col)
        y = df_transformed[target_col]
        test_size = schema.get("test_split", 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        print("\nTraining Suggested Regression Models...")
        suggested = concrete_plan.suggested_regression_models or []
        if not suggested:
            print("No regression models specified in plan. Falling back to LinearRegression and RandomForestRegressor.")
            from data_analysis import Regressor
            suggested = [
                Regressor(model_name="LinearRegression"),
                Regressor(model_name="RandomForestRegressor", hyperparameters={"n_estimators": 50, "max_depth": 10})
            ]

        for model_info in suggested:
            m_name = model_info.model_name
            m_cls = REGRESSOR_MAP.get(m_name)
            if not m_cls:
                print(f"Model '{m_name}' not recognized in REGRESSOR_MAP. Skipping.")
                continue

            try:
                model = m_cls(**model_info.hyperparameters)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                r2 = r2_score(y_test, preds)
                mae = mean_absolute_error(y_test, preds)
                print(f"  ✅ {m_name} | R² Score: {r2:7.4f} | MAE: {mae:7.4f}")
            except Exception as e:
                print(f"  ❌ Error training {m_name}: {e}")

    elif task_type == "classification":
        if not target_col or target_col not in df_transformed.columns:
            raise ValueError(f"Target column '{target_col}' not found in transformed data.")

        X = df_transformed.drop(target_col)
        y = df_transformed[target_col]
        test_size = schema.get("test_split", 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        print("\nTraining Suggested Classification Models...")
        suggested = concrete_plan.suggested_classification_models or []
        if not suggested:
            from data_analysis import Classifier
            suggested = [
                Classifier(model_name="LogisticRegression"),
                Classifier(model_name="RandomForestClassifier", hyperparameters={"n_estimators": 50})
            ]

        for model_info in suggested:
            m_name = model_info.model_name
            m_cls = CLASSIFIER_MAP.get(m_name)
            if not m_cls:
                print(f"Model '{m_name}' not recognized in CLASSIFIER_MAP. Skipping.")
                continue

            try:
                model = m_cls(**model_info.hyperparameters)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                f1 = f1_score(y_test, preds, average="weighted")
                print(f"  ✅ {m_name} | Accuracy: {acc:7.4f} | F1 (weighted): {f1:7.4f}")
            except Exception as e:
                print(f"  ❌ Error training {m_name}: {e}")

    elif task_type == "clustering":
        X = df_transformed
        if target_col and target_col in X.columns:
            X = X.drop(target_col)

        print("\nTraining Suggested Clustering Models...")
        suggested = concrete_plan.suggested_clustering_models or []
        if not suggested:
            from data_analysis import Clusterer
            suggested = [Clusterer(model_name="KMeans", hyperparameters={"n_clusters": 3, "random_state": 42})]

        for model_info in suggested:
            m_name = model_info.model_name
            m_cls = CLUSTERER_MAP.get(m_name)
            if not m_cls:
                print(f"Model '{m_name}' not recognized in CLUSTERER_MAP. Skipping.")
                continue

            try:
                model = m_cls(**model_info.hyperparameters)
                X_mat = X.to_numpy()
                labels = model.fit_predict(X_mat)
                unique_labels = set(labels) - {-1}
                if len(unique_labels) > 1:
                    score = silhouette_score(X_mat, labels)
                    print(f"  ✅ {m_name} | Silhouette Score: {score:7.4f} | Clusters: {len(unique_labels)}")
                else:
                    print(f"  ✅ {m_name} | Fit completed (Clusters formed: {len(unique_labels)})")
            except Exception as e:
                print(f"  ❌ Error training {m_name}: {e}")

    print("\nPipeline execution successfully finished!")


if __name__ == "__main__":
    main()