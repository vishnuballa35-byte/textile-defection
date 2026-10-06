from flask import Flask, send_from_directory
from flask_cors import CORS

import os
from backend.config import (
    FRONTEND_DIR,
    UPLOAD_DIR,
    RESULTS_DIR,
    GRADCAM_DIR,
    MASK_DIR,
    PREDICTION_DIR,
)

from backend.routes.prediction import prediction_bp


# ============================================================
# CREATE FLASK APPLICATION
# ============================================================

app = Flask(
    __name__,
    static_folder=str(FRONTEND_DIR),
    static_url_path=""
)


# ============================================================
# FLASK SETTINGS
# ============================================================

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


# ============================================================
# ENABLE CORS
# ============================================================

CORS(app)


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

GRADCAM_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MASK_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# REGISTER API ROUTES
# ============================================================

app.register_blueprint(prediction_bp)


# ============================================================
# FRONTEND ROUTE
# ============================================================

@app.route("/")
def index():
    """
    Serve the frontend homepage.
    """

    return send_from_directory(
        FRONTEND_DIR,
        "index.html"
    )
@app.route("/results/<path:filename>")
def serve_result(filename):
    return send_from_directory(
        RESULTS_DIR,
        filename
    )

# ============================================================
# RUN SERVER
# ============================================================


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
