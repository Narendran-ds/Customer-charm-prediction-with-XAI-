__version__ = "1.0.0"
__author__ = "Narendran"
__project__ = "Customer Churn Prediction with Explainable AI (XAI)"

import logging
from .utils import (
    load_model_and_scaler,
    evaluate_classification_metrics,
    print_metrics,
    preprocess_single_input,
    save_object,
    load_object,
    summary_statistics
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logging.info(f"🔍 Loaded project package: {__project__} v{__version__} by {__author__}")
