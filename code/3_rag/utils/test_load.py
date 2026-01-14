import json
import os
from pathlib import Path

import pandas as pd

BASE_DIR = Path(os.getenv("FSE_BASE_DIR", Path(__file__).resolve().parents[3]))
EXAMS_DIR = Path(
    os.getenv(
        "FSE_INPUT_DIR",
        BASE_DIR / "results/1_data_preparation/6_json_final",
    )
)

if not EXAMS_DIR.exists():
    raise FileNotFoundError(f"EXAMS_DIR not found: {EXAMS_DIR}")

files = [path.name for path in EXAMS_DIR.glob("*.json")]
print(f"Found {len(files)} JSONs")

for filename in files[:3]:
    path = EXAMS_DIR / filename
    with open(path, "r", encoding="utf-8") as infile:
        data = json.load(infile)
        preguntas = [q for q in data.get("preguntas", []) if q.get("tipo") == "texto"]
        print(f"{filename}: {len(preguntas)} preguntas tipo texto")

out = Path(os.getenv("FSE_OUTPUT_DIR", BASE_DIR / "results/test_metrics.xlsx"))
pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_excel(out, index=False)
print("OK:", out.exists())
