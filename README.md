# MedSignal
## Try the app



[Open MedSignal] https://medsignal03.streamlit.app/

MedSignal is an educational prototype for exploring a public heart-disease dataset
and a machine-learning classification workflow. It is not a medical diagnostic
tool and must not be used to make healthcare decisions.

> **Current status:** Phase 5. UCI dataset loading, inspection, preprocessing,
> baseline training, held-out evaluation, fitted-pipeline persistence, and
> saved-model prediction, dashboard, and basic coefficient insights are
> implemented.

## Project layout

```text
medsignal/
├── app.py
├── requirements.txt
├── data/
│   └── raw/                 # Downloaded public-data cache (not committed)
├── models/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── insights.py
│   ├── model_artifact.py
│   └── preprocessing.py
├── train_model.py
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_evaluate.py
│   ├── test_insights.py
│   ├── test_model_artifact.py
│   └── test_preprocessing.py
```

The Streamlit UI, model training, and data handling are kept in separate modules.

## Phase 2: Load and inspect the dataset

The project retrieves the public [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
through `ucimlrepo` (dataset ID 45). The initial download is cached as
`data/raw/uci_heart_disease.csv`; that generated copy is excluded from Git.
Question-mark values are normalized to missing values (`NaN`), and columns are
checked/coerced as numeric. Missing values are reported, not filled in during
this inspection phase.

The original target column is `num`: `0` means no angiographic heart disease;
values `1` through `4` indicate its presence in the source data's encoding. The
training target is binary: `0` remains no disease and `1`-`4` become disease
present. These labels are dataset annotations, not a clinical diagnosis.

The fetched dataset contains 303 rows and 13 input features. In the observed
download, `ca` has 4 missing values and `thal` has 2; all other columns and the
target have no missing values. The original target distribution is 164 records
with label `0`, 55 with label `1`, 36 with label `2`, 35 with label `3`, and 13
with label `4`. After the documented binary mapping this is 164 records labeled
no disease and 139 labeled disease present. These are dataset counts, not model
performance results.

| Feature | Meaning and source encoding |
| --- | --- |
| `age` | Age in years |
| `sex` | `0` female, `1` male |
| `cp` | Chest-pain type: `1` typical angina, `2` atypical angina, `3` non-anginal pain, `4` asymptomatic |
| `trestbps` | Resting blood pressure, mm Hg |
| `chol` | Serum cholesterol, mg/dl |
| `fbs` | Fasting blood sugar > 120 mg/dl: `0` false, `1` true |
| `restecg` | Resting ECG: `0` normal, `1` ST-T abnormality, `2` probable/definite left-ventricular hypertrophy |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina: `0` no, `1` yes |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Peak-exercise ST slope: `1` upsloping, `2` flat, `3` downsloping |
| `ca` | Number of major vessels colored by fluoroscopy (`0`-`3`); 4 records are missing |
| `thal` | Thalassemia category: `3` normal, `6` fixed defect, `7` reversible defect; 2 records are missing |

## Phase 3: Preprocess, train, and evaluate

The preprocessing and Logistic Regression estimator are joined in a Scikit-learn
pipeline. Numeric measurements are median-imputed and standardized; encoded
categorical features are most-frequent-imputed and one-hot encoded. The data is
split into stratified 80% training and 20% test sets with random seed `42` before
the pipeline is fitted, so imputation, scaling, and category encoding learn only
from training data. This avoids fitting preprocessing on the held-out test set.

Train the baseline and print actual held-out metrics with:

```powershell
.\.venv\Scripts\python.exe train_model.py
```

The output contains accuracy, precision, recall, F1, ROC-AUC, and a confusion
matrix in `[actual 0, actual 1]` row order and `[predicted 0, predicted 1]`
column order. Metrics describe one small test split, are not fabricated or
clinically validated, and may vary if the source data changes.

Observed run on the cached 303-row dataset with seed `42` (61 held-out records):

| Metric | Value |
| --- | ---: |
| Accuracy | 0.8852 |
| Precision | 0.8387 |
| Recall | 0.9286 |
| F1 | 0.8814 |
| ROC-AUC | 0.9665 |

Confusion matrix: `[[28, 5], [2, 26]]` (28 true negatives, 5 false positives,
2 false negatives, 26 true positives). This is one small held-out split, not
evidence of clinical readiness or expected performance on other populations.

Run the focused tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_data_loader.py tests\test_preprocessing.py tests\test_evaluate.py -q
```

## Phase 4: Save and verify the model

Training writes `models/model_pipeline.joblib`, containing the complete fitted
preprocessing-and-classifier pipeline, held-out metrics, and exact feature
schema. This generated artifact is excluded from Git. The app can load this
artifact later instead of retraining on every launch.

Create or refresh the saved pipeline:

```powershell
.\.venv\Scripts\python.exe train_model.py
```

The script prints actual evaluation metrics and confirms where the artifact was
saved. `load_model_artifact()` reports a missing model with instructions to train
it, and `predict_with_artifact()` rejects inputs whose feature schema does not
match. The round-trip test saves and reloads a fitted pipeline, then checks its
prediction and class probabilities.

## Phase 5: Streamlit dashboard

The application loads the saved model artifact; it does not retrain on app
startup. Its three sidebar sections provide an overview with held-out metrics,
dataset distribution and a confusion matrix; a 13-feature input form with
validation, reset, prediction, and model-estimated class probability; and model
insights with Logistic Regression coefficient magnitude, local score
contributions, evaluation metrics, and limitations.

First train the artifact, then launch the app:

```powershell
.\.venv\Scripts\python.exe train_model.py
.\.venv\Scripts\streamlit.exe run app.py
```

The app displays a clear message with the training command if its artifact is
missing. If your `.venv` uses Python 3.15, preserve it and create a separate
Python 3.13 environment for Streamlit:

```powershell
py -3.13 -m venv .venv313
.\.venv313\Scripts\python.exe -m pip install --upgrade pip
.\.venv313\Scripts\python.exe -m pip install -r requirements.txt
.\.venv313\Scripts\python.exe train_model.py
.\.venv313\Scripts\streamlit.exe run app.py
```

If you already created `.venv313`, use its Python and Streamlit executables for
the commands above. Keep the terminal open while the app is running.

After installing the dependencies, inspect the fetched/cached data with:

```powershell
.\.venv\Scripts\python.exe -m src.data_loader
.\.venv\Scripts\python.exe -m pytest tests\test_data_loader.py
```

The first command prints the row and feature counts, missing values, and target
class distribution derived from the dataset. Use `refresh=True` in
`load_heart_disease_data()` to fetch a fresh copy instead of using the local cache.

## Requirements

- Windows, macOS, or Linux
- Python 3.13 (recommended for broad package-wheel compatibility)
- Internet access for installing packages and retrieving the public dataset in a
  later phase

## Set up on Windows (PowerShell)

Run these commands from the project folder:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\streamlit.exe run app.py
```

The first command creates a local virtual environment. The following commands
install the project's Python dependencies and launch the Streamlit setup page.
Using the environment's executable directly avoids needing to change PowerShell's
script-execution policy to activate it.

If you run a Python command without the `.venv\Scripts\python.exe` prefix, make
sure the virtual environment is active first:

```powershell
.\.venv\Scripts\Activate.ps1
python -m src.data_loader
```

Otherwise, use `.\.venv\Scripts\python.exe -m src.data_loader` directly. A plain
`python` command may select a different Python installation that does not have
the project's dependencies installed.

To stop Streamlit, press **Ctrl+C** in the terminal.

## Planned workflow

1. Retrieve and inspect the UCI Heart Disease dataset. **Implemented.**
2. Define the target-label interpretation and validate input features. **Target mapping documented; prediction form constrains values and maps the complete feature schema.**
3. Split the data before fitting preprocessing; train and evaluate a baseline
   model using a reproducible random seed. **Implemented.**
4. Save and test the fitted preprocessing-and-model pipeline. **Implemented.**
5. Load that saved pipeline from the Streamlit app to make predictions and show
   real evaluation results and basic model insights. **Interface implemented;
   launching and browser verification require Python 3.13 because Streamlit
   dependencies could not be installed in this machine's Python 3.15 environment.**

The dataset is small and may not represent other populations; model performance
must not be interpreted as clinical validation.
