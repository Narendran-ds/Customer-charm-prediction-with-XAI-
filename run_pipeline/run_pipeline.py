import os
import subprocess
import sys
import time

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resolve_python():
    """Prefer this project's own venv if it exists; fall back to whichever
    interpreter is running this script so the pipeline still works on a
    machine where .venv hasn't been (re)created yet."""
    venv_python = os.path.join(BASE_DIR, ".venv", "Scripts", "python.exe")
    return venv_python if os.path.exists(venv_python) else sys.executable


def run_step(description, command_list, cwd=BASE_DIR):
    print(f"\n{description}")
    print(f"Running command: {command_list}")
    try:
        result = subprocess.run(command_list, cwd=cwd)
    except OSError as e:
        print(f"Failed at: {description} ({e})")
        sys.exit(1)
    if result.returncode != 0:
        print(f"Failed at: {description}")
        sys.exit(1)
    print(f"Completed: {description}")


def main():
    print("Starting Customer Churn Prediction Full Pipeline")
    python_exe = resolve_python()

    # Invoked as modules (python -m src.xxx) with cwd=BASE_DIR, rather than by
    # file path, so each script's `from src...` imports resolve regardless of
    # where run_pipeline.py itself was launched from.
    run_step("Running Data Preprocessing", [python_exe, "-m", "src.data_preprocessing"])
    run_step("Running EDA", [python_exe, "-m", "src.eda"])
    run_step("Training Model", [python_exe, "-m", "src.train_model"])
    run_step("Running Explainability", [python_exe, "-m", "src.explainability"])

    print("\nPipeline completed successfully.")
    time.sleep(2)

    print("\nLaunching Streamlit App")
    subprocess.Popen(
        [python_exe, "-m", "streamlit", "run", os.path.join(BASE_DIR, "app", "app.py")],
        cwd=BASE_DIR,
    )
    print("Streamlit app started on http://localhost:8501")


if __name__ == "__main__":
    main()
