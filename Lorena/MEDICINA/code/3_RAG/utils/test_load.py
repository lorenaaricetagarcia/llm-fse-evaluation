import os, json

EXAMS_DIR = "/home/xs1/Desktop/Lorena/MEDICINA/results/1_data_preparation/6_json_final/prueba"

files = [f for f in os.listdir(EXAMS_DIR) if f.endswith(".json")]
print(f"Found {len(files)} JSONs")

for f in files[:3]:
    path = os.path.join(EXAMS_DIR, f)
    with open(path, "r", encoding="utf-8") as infile:
        data = json.load(infile)
        preguntas = [q for q in data.get("preguntas", []) if q.get("tipo") == "texto"]
        print(f"{f}: {len(preguntas)} preguntas tipo texto")


import pandas as pd
import os

out = "/home/xs1/Desktop/Lorena/MEDICINA/results/test_metrics.xlsx"
pd.DataFrame({"a":[1,2],"b":[3,4]}).to_excel(out, index=False)
print("OK:", os.path.exists(out))

