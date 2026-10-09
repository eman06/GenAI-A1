"""Point the MLflow DB downloaded from Colab at a local artifact folder so `mlflow ui` can show images/files.

    python scripts/localize_mlflow.py tracking/mlflow.db tracking/mlruns
    mlflow ui --backend-store-uri sqlite:///tracking/mlflow.db --port 5050
"""
import os
import sqlite3
import sys

db, mlruns = sys.argv[1], sys.argv[2]
root = os.path.abspath(mlruns).replace(os.sep, "/")
c = sqlite3.connect(db)
for prefix in ("file:///content/outputs/mlruns",):
    c.execute("update experiments set artifact_location = replace(artifact_location, ?, ?)", (prefix, "file:///" + root))
    c.execute("update runs set artifact_uri = replace(artifact_uri, ?, ?)", (prefix, "file:///" + root))
c.commit()
print(c.execute("select artifact_uri from runs limit 1").fetchone())
print("runs:", c.execute("select count(*) from runs").fetchone()[0], "metric rows:", c.execute("select count(*) from metrics").fetchone()[0])
