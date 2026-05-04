from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

DEMO_DATA_PATH = DATA_DIR / "demo_toxicity.csv"
JIGSAW_TRAIN_PATH = DATA_DIR / "jigsaw" / "train.csv"

LABELS = ["safe", "toxic", "abusive", "policy_violation"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

BASELINE_MODEL_PATH = MODEL_DIR / "baseline_tfidf_logreg.joblib"
BASELINE_METRICS_PATH = REPORT_DIR / "baseline_metrics.json"
TRANSFORMER_METRICS_PATH = REPORT_DIR / "transformer_metrics.json"
COMPARISON_PATH = REPORT_DIR / "model_comparison.json"
CONFUSION_MATRIX_CSV = REPORT_DIR / "confusion_matrix.csv"
CONFUSION_MATRIX_PNG = REPORT_DIR / "confusion_matrix.png"

DEFAULT_TRANSFORMER_MODEL = "distilbert-base-uncased"
