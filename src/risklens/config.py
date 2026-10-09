from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_PATH = MODEL_DIR / "risklens.joblib"

SEED = 42
TARGET = "default"  # 1 = borrower had serious delinquency within 2 years

# Business cost model (illustrative; change to your portfolio economics).
# Approving a borrower who defaults loses roughly principal * LGD; rejecting a good borrower loses margin.
COST_FALSE_NEGATIVE = 10_000   # approved a defaulter
COST_FALSE_POSITIVE = 1_000    # rejected a good customer

# Score scaling: 600 points = 50:1 good/bad odds, +20 points doubles the odds ("PDO")
BASE_SCORE, BASE_ODDS, PDO = 600, 50, 20
