<h1 align="center">ExperimentFlow</h1>

An LLM-driven machine learning pipeline that uses **Gemini** to analyze dataset schemas, suggest end-to-end workflows, and compile them into executable **scikit-learn** pipelines via a **Planner → Validator → Executor** model.


## How It Works

1. **Planner (LLM + Structured Output):**
   - **Step 1:** Reads `user_initial_query` from `data.yaml`. When that field is missing or blank, the query is `Analyze the dataset, find suitable targets, and recommend the best ML task.` Gemini returns an `InitialAnalysis` (overview, feasible tasks, candidate targets).
   - **Step 2:** Takes the user's follow-up preferences and generates a concrete `Analysis` plan specifying preprocessing transformers and candidate models.

2. **Validator (Guardrails):**
   - Strips target and drop columns from feature transformations to prevent leakage.
   - Enforces type compatibility (e.g., scalers only on numeric columns).
   - Replaces incompatible transformers (e.g., `LabelEncoder` → `OrdinalEncoder`).
   - Sanitizes parameters (`sparse_output=False`, `handle_unknown="ignore"`).
   - Auto-adds imputers for unhandled missing values.

3. **Executor:**
   - Fits and transforms data, preserving clean Polars DataFrame outputs.
   - Trains recommended models (Regression, Classification, or Clustering) and evaluates performance.
   - Prints live progress while each model fits.
   - Logs every model run to a local MLflow store at `./mlruns`: parameters, a `config/run_config.json` artifact, metrics.

## Why MLflow is used?

One prompt can train several models, and the terminal only keeps the last score on screen. I use MLflow so each model is its own run: the hyperparameters, the validated plan, the metrics, and the fitted estimator. Those runs stay in `./mlruns` and can be compared in the UI after the process exits. All the trained models will be in this directory.

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000).

---

### 1. Installation

```bash
git clone https://github.com/AmishKakka/ExperimentFlow.git
cd ExperimentFlow

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API Key

Create a `.env` file in the project root:
```env
GEMINI_API_KEY="your_api_key_here"
```

### 3. Run

Set `data.user_initial_query` in `data.yaml`, along with the data source. Leave the query blank to use the default analysis prompt. Then run:
```bash
python main.py
```

## Pointers
1. You can only use a single dataset ata time.
2. Will add nice UI to better manage the flow.
