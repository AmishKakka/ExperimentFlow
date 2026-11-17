import os
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field
from google import genai
from dotenv import load_dotenv

load_dotenv()


def _get_genai_client() -> genai.Client:
    api_key = os.getenv("API_Key") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "API key not found. Please set 'API_Key' or 'GEMINI_API_KEY' in your environment or in a .env file."
        )
    return genai.Client(api_key=api_key, http_options={"api_version": "v1beta"})


class Transformation(BaseModel):
    """A structured scikit-learn data transformation step."""
    columns: List[str] = Field(description="List of column names this transformation applies to.")
    transformer: Literal[
        "SimpleImputer",
        "StandardScaler",
        "MinMaxScaler",
        "RobustScaler",
        "OneHotEncoder",
        "OrdinalEncoder",
        "LabelEncoder"
    ] = Field(description="The scikit-learn class name of the transformer to use.")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="A dictionary of valid constructor parameters for the transformer, e.g. {'strategy': 'median'}."
    )


class Classifier(BaseModel):
    """A structured classification model."""
    model_name: Literal[
        "LogisticRegression",
        "RandomForestClassifier",
        "GradientBoostingClassifier",
        "SVC",
        "KNeighborsClassifier"
    ] = Field(description="The sklearn classifier class name.")
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="A dictionary of hyperparameters for the model."
    )


class Regressor(BaseModel):
    """A structured regression model."""
    model_name: Literal[
        "LinearRegression",
        "Ridge",
        "Lasso",
        "RandomForestRegressor",
        "GradientBoostingRegressor",
        "SVR"
    ] = Field(description="The sklearn regressor class name.")
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="A dictionary of hyperparameters for the model."
    )


class Clusterer(BaseModel):
    """A structured clustering model."""
    model_name: Literal[
        "KMeans",
        "DBSCAN",
        "AgglomerativeClustering"
    ] = Field(description="The sklearn clustering class name.")
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="A dictionary of hyperparameters for the model."
    )


class InitialAnalysis(BaseModel):
    """Initial exploration and evaluation of what can be done on the dataset based on user query."""
    overview: str = Field(
        description="A summary of the dataset: number of features, data types, missing values, and high-level domain context."
    )
    task_type: Literal["regression", "classification", "clustering"] = Field(
        description="The primary recommended ML task type matching the dataset and user intent."
    )
    target_column: Optional[str] = Field(
        default=None,
        description="Recommended target column for supervised learning (null if clustering)."
    )
    candidate_target_columns: List[str] = Field(
        default_factory=list,
        description="Other viable target columns for prediction or analysis."
    )
    reasoning: str = Field(
        description="Detailed explanation of what the LLM thinks we can do with this dataset, opportunities, and trade-offs."
    )
    suggested_models: List[str] = Field(
        default_factory=list,
        description="Candidate model names recommended for this task."
    )
    preliminary_transformations: List[Transformation] = Field(
        default_factory=list,
        description="Preliminary list of sklearn transformations suggested for this dataset."
    )


class Analysis(BaseModel):
    """A complete and concrete structured analysis and execution plan."""
    overview: str = Field(
        description="A concise summary of the agreed-upon plan and dataset."
    )
    task_type: Literal["regression", "classification", "clustering"] = Field(
        description="The finalized ML task type."
    )
    target_column: Optional[str] = Field(
        default=None,
        description="Final target column to predict (null if clustering)."
    )
    transformations: List[Transformation] = Field(
        description="A structured list of scikit-learn transformations to perform on feature columns."
    )
    suggested_classification_models: Optional[List[Classifier]] = Field(
        default=None,
        description="A list of suggested classification models if task_type is classification."
    )
    suggested_regression_models: Optional[List[Regressor]] = Field(
        default=None,
        description="A list of suggested regression models if task_type is regression."
    )
    suggested_clustering_models: Optional[List[Clusterer]] = Field(
        default=None,
        description="A list of suggested clustering models if task_type is clustering."
    )


def get_analysis(df_head, schema, user_query: str) -> InitialAnalysis:
    """
    Step 1: Analyzes the dataset schema, sample rows, and initial user query.
    Returns what the LLM thinks we can achieve on the dataset.
    """
    client = _get_genai_client()

    sample_rows = df_head.to_dicts() if hasattr(df_head, "to_dicts") else str(df_head)
    schema_str = str(schema)

    prompt = (
        "You are a lead machine learning engineer and data scientist. "
        "Analyze the following dataset sample and schema against the user's initial training/output goal.\n\n"
        f"USER GOAL / QUERY: \"{user_query}\"\n\n"
        f"DATASET SCHEMA: {schema_str}\n"
        f"DATASET SAMPLE (first rows): {sample_rows}\n\n"
        "Explain what we can do with this dataset. Evaluate potential target variables, "
        "determine the most viable ML task (regression, classification, clustering), "
        "propose candidate target columns, and suggest preliminary preprocessing steps "
        "using standard scikit-learn transformers ('SimpleImputer', 'StandardScaler', "
        "'MinMaxScaler', 'RobustScaler', 'OneHotEncoder', 'OrdinalEncoder')."
    )

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": InitialAnalysis.model_json_schema(),
        }
    )
    if not response.text:
        raise ValueError("Received empty response from Gemini.")

    cleaned_text = response.text.strip()
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[7:]
    if cleaned_text.startswith("```"):
        cleaned_text = cleaned_text[3:]
    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3]
    initial_analysis = InitialAnalysis.model_validate_json(cleaned_text.strip())
    return initial_analysis


def get_concrete_plan(
    df_head,
    schema,
    initial_query: str,
    final_query: str,
    initial_analysis: Optional[InitialAnalysis] = None
) -> Analysis:
    """
    Step 2: Synthesize the initial understanding and the user's final query
    to generate the final concrete execution plan with validated Scikit-Learn transformations.
    """
    client = _get_genai_client()

    sample_rows = df_head.to_dicts() if hasattr(df_head, "to_dicts") else str(df_head)
    schema_str = str(schema)
    initial_context = initial_analysis.model_dump_json() if initial_analysis else "None"

    prompt = f"""You are a lead machine learning engineer. Produce a finalized, concrete, production-ready ML plan.
    INITIAL USER QUERY: "{initial_query}"
    USER'S FINAL REFINEMENT / CONCRETE PREFERENCES: "{final_query}"
    INITIAL ASSESSMENT CONTEXT: {initial_context}
    DATASET SCHEMA: {schema_str}
    DATASET SAMPLE (first rows): {sample_rows}

    CRITICAL RULES FOR TRANSFORMATIONS:
    1. Use ONLY standard scikit-learn transformers from this list:
    ['SimpleImputer', 'StandardScaler', 'MinMaxScaler', 'RobustScaler', 'OneHotEncoder', 'OrdinalEncoder'].
    2. NEVER include the target column in transformations. Transformations apply ONLY to feature columns.
    3. For categorical features, suggest 'OneHotEncoder' (with 'sparse_output': False) or 'OrdinalEncoder'. Do NOT use 'LabelEncoder' for features.
    4. For SimpleImputer on numeric columns, use 'mean' or 'median'. On string/categorical columns, use 'most_frequent' or 'constant'.
    5. Only suggest models relevant to the final task_type.
    - regression: ['LinearRegression', 'Ridge', 'Lasso', 'RandomForestRegressor', 'GradientBoostingRegressor', 'SVR']
    - classification: ['LogisticRegression', 'RandomForestClassifier', 'GradientBoostingClassifier', 'SVC', 'KNeighborsClassifier']
    - clustering: ['KMeans', 'DBSCAN', 'AgglomerativeClustering']
    """
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": Analysis.model_json_schema(),
        }
    )

    if not response.text:
        raise ValueError("Received empty or null response from Gemini API.")

    cleaned_text = response.text.strip()
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[7:]
    if cleaned_text.startswith("```"):
        cleaned_text = cleaned_text[3:]
    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3]
    concrete_plan = Analysis.model_validate_json(cleaned_text.strip())
    return concrete_plan