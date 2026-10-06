from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# FRONTEND
# ============================================================

FRONTEND_DIR = PROJECT_DIR / "frontend"


# ============================================================
# DATASET
# ============================================================

DATASET_DIR = PROJECT_DIR / "dataset"

NORMAL_DIR = DATASET_DIR / "normal"

DEFECT_DIR = DATASET_DIR / "defect"


# ============================================================
# MODELS
# ============================================================

MODEL_DIR = PROJECT_DIR / "models"

CLASSIFIER_MODEL = MODEL_DIR / "mobilenetv2_classifier.keras"

UNET_MODEL = MODEL_DIR / "unet_segmentation.keras"

SVM_MODEL = MODEL_DIR / "lbp_svm.pkl"


# ============================================================
# UPLOADS
# ============================================================

UPLOAD_DIR = PROJECT_DIR / "uploads"


# ============================================================
# RESULTS
# ============================================================

RESULTS_DIR = PROJECT_DIR / "results"

GRADCAM_DIR = RESULTS_DIR / "gradcam"

MASK_DIR = RESULTS_DIR / "masks"

PREDICTION_DIR = RESULTS_DIR / "predictions"


# ============================================================
# IMAGE SETTINGS
# ============================================================

CLASSIFICATION_IMAGE_SIZE = (224, 224)

SEGMENTATION_IMAGE_SIZE = (256, 256)


# ============================================================
# ALLOWED FILE TYPES
# ============================================================

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}


# ============================================================
# SERVER SETTINGS
# ============================================================

MAX_CONTENT_LENGTH = 16 * 1024 * 1024

