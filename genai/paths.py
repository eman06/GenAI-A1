"""Default locations. Override with environment variables on Colab / Docker."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("GENAI_DATA", os.path.join(ROOT, "data"))
OUT_ROOT = os.environ.get("GENAI_OUT", os.path.join(ROOT, "outputs"))
MANIFEST_DIR = os.environ.get("GENAI_MANIFESTS", os.path.join(ROOT, "manifests"))  # deterministic -> committed
PET_CACHE = os.path.join(DATA_ROOT, "pets_128.npz")
CKPT_DIR = os.path.join(OUT_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(OUT_ROOT, "results")
OPTUNA_DIR = os.path.join(OUT_ROOT, "optuna")
ONNX_DIR = os.path.join(ROOT, "models_onnx")
MLRUNS_DIR = os.path.join(OUT_ROOT, "mlruns")


def ensure_dirs():
    for d in (DATA_ROOT, OUT_ROOT, CKPT_DIR, RESULTS_DIR, OPTUNA_DIR, ONNX_DIR, MLRUNS_DIR):
        os.makedirs(d, exist_ok=True)


def optuna_storage(name):
    ensure_dirs()
    return f"sqlite:///{os.path.join(OPTUNA_DIR, name + '.db')}"


def setup_mlflow(experiment):
    import mlflow
    ensure_dirs()
    uri = os.environ.get("MLFLOW_TRACKING_URI", "file:" + MLRUNS_DIR.replace("\\", "/"))
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    return mlflow
