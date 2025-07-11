import os
import sys
import subprocess
import time

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def run_step(description, command_list):
    print(f"\n🚀 {description}")
    print(f"👉 Running command: {command_list}")
    result = subprocess.run(command_list)
    if result.returncode != 0:
        print(f"❌ Failed at: {description}")
        sys.exit(1)
    else:
        print(f"✅ Completed: {description}")

print("📊 Starting Customer Churn Prediction Full Pipeline")

venv_python = os.path.join(BASE_DIR, "venv", "Scripts", "python.exe")
streamlit_exe = os.path.join(BASE_DIR, "venv", "Scripts", "streamlit.exe")

run_step("Running Data Preprocessing", [venv_python, os.path.join(BASE_DIR, "src", "data_preprocessing.py")])
run_step("Running EDA", [venv_python, os.path.join(BASE_DIR, "src", "eda.py")])
run_step("Training Model", [venv_python, os.path.join(BASE_DIR, "src", "train_model.py")])
run_step("Running Explainability", [venv_python, os.path.join(BASE_DIR, "src", "explainability.py")])

print("\n🎉 Pipeline completed successfully.")
time.sleep(2)

print("\n🚀 Launching Streamlit App")
subprocess.Popen([streamlit_exe, "run", os.path.join(BASE_DIR, "app", "app.py")])

print("✅ Streamlit app started on http://localhost:8501")
