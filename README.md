# mlops_pipeline

An LLM-driven machine learning pipeline that uses **Gemini** to analyze dataset schemas, suggest end-to-end workflows, and compile them into executable **scikit-learn** pipelines via a **Planner → Validator → Executor** model.

---

## How It Works

1. **Planner (LLM + Structured Output):**
   - **Step 1:** Sends dataset sample rows and schema to Gemini with the user's intent query. Gemini returns an `InitialAnalysis` (overview, feasible tasks, candidate targets).
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

---

### 1. Installation

```bash
git clone https://github.com/AmishKakka/mlops_pipeline.git
cd mlops_pipeline

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

Configure your data source in `data.yaml`, then run:
```bash
python main.py
```