from pathlib import Path

import cv2
import joblib
import numpy as np
from backend.localization.gradcam import generate_gradcam

from PIL import Image
from flask import Blueprint, jsonify, request
from skimage.feature import local_binary_pattern

from backend.config import (
    UPLOAD_DIR,
    ALLOWED_EXTENSIONS,
    CLASSIFIER_MODEL,
    UNET_MODEL,
    SVM_MODEL,
    CLASSIFICATION_IMAGE_SIZE,
    SEGMENTATION_IMAGE_SIZE,
    GRADCAM_DIR,
    MASK_DIR,
    DEFECT_DIR,
)

from backend.localization.gradcam import generate_gradcam

from backend.utils.file_utils import (
    ensure_directory,
    is_allowed_file,
    secure_file_name,
)


# ============================================================
# BLUEPRINT
# ============================================================

prediction_bp = Blueprint(
    "prediction",
    __name__,
    url_prefix="/api"
)


# ============================================================
# LOAD CNN MODEL
# ============================================================

classifier_model = tf.keras.models.load_model(
    CLASSIFIER_MODEL
)

print("===================================")
print("MobileNetV2 model loaded:")
print(CLASSIFIER_MODEL)
print("===================================")


# ============================================================
# LOAD U-NET MODEL
# ============================================================

unet_model = tf.keras.models.load_model(
    UNET_MODEL,
    compile=False
)

print("===================================")
print("U-Net model loaded:")
print(UNET_MODEL)

print("U-Net expected input shape:")
print(unet_model.input_shape)
print("===================================")


# ============================================================
# GET U-NET INPUT SIZE DIRECTLY FROM MODEL
# ============================================================

def get_unet_input_size():

    input_shape = unet_model.input_shape

    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if len(input_shape) != 4:
        raise ValueError(
            f"Unexpected U-Net input shape: {input_shape}"
        )

    height = input_shape[1]
    width = input_shape[2]
    channels = input_shape[3]

    if height is None or width is None:
        raise ValueError(
            f"U-Net input dimensions are undefined: {input_shape}"
        )

    if channels != 3:
        raise ValueError(
            f"U-Net expects {channels} channels instead of 3."
        )

    # PIL uses (width, height)
    return (
        int(width),
        int(height)
    )


UNET_INPUT_SIZE = get_unet_input_size()

print("U-Net preprocessing size:")
print(UNET_INPUT_SIZE)

print("Config segmentation size:")
print(SEGMENTATION_IMAGE_SIZE)

if UNET_INPUT_SIZE != tuple(SEGMENTATION_IMAGE_SIZE):

    print(
        "WARNING: Config segmentation size does not match "
        "the U-Net model."
    )

    print(
        "Using actual U-Net model input size:",
        UNET_INPUT_SIZE
    )

print("===================================")


# ============================================================
# LOAD LBP + SVM MODEL
# ============================================================

svm_model = joblib.load(
    SVM_MODEL
)

print("LBP + SVM model loaded:")
print(SVM_MODEL)
print("===================================")


# ============================================================
# CONSTANTS
# ============================================================

# Lowered from 0.40 to 0.30 so borderline defects
# are less likely to be classified as NORMAL.
CNN_THRESHOLD = 0.30


# Keep this at 0.05 temporarily.
# After the new 256x256 U-Net training/evaluation,
# replace this using the validation threshold sweep.
SEGMENTATION_THRESHOLD = 0.05


# ============================================================
# CNN CLASSIFICATION
# ============================================================

def predict_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        CLASSIFICATION_IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = (
        tf.keras.applications
        .mobilenet_v2
        .preprocess_input(
            image_array
        )
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    prediction = float(
        classifier_model.predict(
            image_array,
            verbose=0
        )[0][0]
    )

    if prediction >= CNN_THRESHOLD:

        cnn_class = "DEFECT"

        confidence = prediction * 100

    else:

        cnn_class = "NORMAL"

        confidence = (1.0 - prediction) * 100

    # --------------------------------------------------------
    # CNN DIAGNOSTIC
    # --------------------------------------------------------

    print("")
    print("===================================")
    print("CNN DIAGNOSTIC")
    print("===================================")

    print(
        "CNN Image:",
        image_path
    )

    print(
        "Raw CNN probability:",
        prediction
    )

    print(
        "CNN threshold:",
        CNN_THRESHOLD
    )

    print(
        "CNN classification:",
        cnn_class
    )

    print(
        "CNN confidence:",
        round(
            confidence,
            2
        ),
        "%"
    )

    print("===================================")

    return {

        "cnn_class":
            cnn_class,

        "confidence":
            round(
                confidence,
                2
            ),

        "raw_prediction":
            round(
                prediction,
                6
            )
    }


# ============================================================
# LBP FEATURE EXTRACTION
# ============================================================

def extract_lbp_features(
    image_path,
    image_size=(256, 256),
    num_points=24,
    radius=3
):

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    image = cv2.resize(
        image,
        image_size
    )

    lbp = local_binary_pattern(
        image,
        num_points,
        radius,
        method="uniform"
    )

    num_bins = num_points + 2

    histogram, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(
            0,
            num_bins + 1
        ),
        range=(
            0,
            num_bins
        )
    )

    histogram = histogram.astype(
        np.float32
    )

    histogram /= (
        histogram.sum() + 1e-7
    )

    return histogram


# ============================================================
# SVM CLASSIFICATION
# ============================================================

def predict_svm(image_path):

    features = extract_lbp_features(
        image_path
    )

    features = features.reshape(
        1,
        -1
    )

    svm_prediction = int(
        svm_model.predict(
            features
        )[0]
    )

    probabilities = svm_model.predict_proba(
        features
    )[0]

    class_to_probability = {

        int(cls): float(probability)

        for cls, probability in zip(
            svm_model.classes_,
            probabilities
        )
    }

    if svm_prediction == 1:

        svm_class = "DEFECT"

        svm_confidence = (
            class_to_probability.get(
                1,
                0.0
            ) * 100
        )

    else:

        svm_class = "NORMAL"

        svm_confidence = (
            class_to_probability.get(
                0,
                0.0
            ) * 100
        )

    # --------------------------------------------------------
    # SVM DIAGNOSTIC
    # --------------------------------------------------------

    print("")
    print("===================================")
    print("SVM DIAGNOSTIC")
    print("===================================")

    print(
        "SVM Image:",
        image_path
    )

    print(
        "SVM raw prediction:",
        svm_prediction
    )

    print(
        "SVM classification:",
        svm_class
    )

    print(
        "SVM confidence:",
        round(
            svm_confidence,
            2
        ),
        "%"
    )

    print("===================================")

    return {

        "svm_class":
            svm_class,

        "svm_confidence":
            round(
                svm_confidence,
                2
            ),

        "svm_raw_prediction":
            svm_prediction
    }


# ============================================================
# FINAL INSPECTION DECISION
# ============================================================

def determine_final_status(
    cnn_class,
    svm_class
):

    # --------------------------------------------------------
    # DEFECT-SENSITIVE ENSEMBLE
    # --------------------------------------------------------
    #
    # If EITHER classifier detects a defect,
    # reject the textile sample.
    #
    # This is intentionally conservative because
    # missing a real defect is worse than sending
    # a questionable sample for rejection/review.
    #

    if (
        cnn_class == "DEFECT"
        or
        svm_class == "DEFECT"
    ):

        return "REJECT"

    # --------------------------------------------------------
    # BOTH MODELS SAY NORMAL
    # --------------------------------------------------------

    if (
        cnn_class == "NORMAL"
        and
        svm_class == "NORMAL"
    ):

        return "PASS"

    return "REVIEW"


# ============================================================
# FIND GROUND-TRUTH MASK
# ============================================================

def find_ground_truth_mask(filename):

    image_stem = Path(
        filename
    ).stem

    ground_truth_path = (
        DEFECT_DIR /
        f"{image_stem}.png"
    )

    print("")
    print("===================================")
    print("GROUND-TRUTH SEARCH")
    print("===================================")

    print(
        "Uploaded filename:",
        filename
    )

    print(
        "Image stem:",
        image_stem
    )

    print(
        "Expected mask:",
        ground_truth_path
    )

    print(
        "Mask exists:",
        ground_truth_path.exists()
    )

    print("===================================")

    if ground_truth_path.exists():

        return ground_truth_path

    return None


# ============================================================
# IoU CALCULATION
# ============================================================

def calculate_iou(
    ground_truth,
    predicted_mask
):

    ground_truth = ground_truth.astype(
        bool
    )

    predicted_mask = predicted_mask.astype(
        bool
    )

    intersection = np.logical_and(
        ground_truth,
        predicted_mask
    ).sum()

    union = np.logical_or(
        ground_truth,
        predicted_mask
    ).sum()

    if union == 0:

        if intersection == 0:
            return 1.0

        return 0.0

    return (
        intersection /
        union
    )


# ============================================================
# DICE CALCULATION
# ============================================================

def calculate_dice(
    ground_truth,
    predicted_mask
):

    ground_truth = ground_truth.astype(
        bool
    )

    predicted_mask = predicted_mask.astype(
        bool
    )

    intersection = np.logical_and(
        ground_truth,
        predicted_mask
    ).sum()

    ground_truth_area = (
        ground_truth.sum()
    )

    predicted_area = (
        predicted_mask.sum()
    )

    denominator = (
        ground_truth_area +
        predicted_area
    )

    if denominator == 0:

        if intersection == 0:
            return 1.0

        return 0.0

    return (
        2.0 * intersection
    ) / denominator


# ============================================================
# U-NET SEGMENTATION
# ============================================================

def segment_image(
    image_path,
    output_path,
    ground_truth_path=None
):

    # --------------------------------------------------------
    # LOAD ORIGINAL IMAGE
    # --------------------------------------------------------

    original_image = Image.open(
        image_path
    ).convert("RGB")

    original_size = (
        original_image.size
    )


    # --------------------------------------------------------
    # RESIZE FOR U-NET
    # --------------------------------------------------------

    image = original_image.resize(
        UNET_INPUT_SIZE,
        Image.Resampling.BILINEAR
    )


    # --------------------------------------------------------
    # NUMPY
    # --------------------------------------------------------

    image_array = np.array(
        image,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    image_array = (
        image_array / 255.0
    )


    # --------------------------------------------------------
    # ADD BATCH DIMENSION
    # --------------------------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # --------------------------------------------------------
    # INPUT DIAGNOSTIC
    # --------------------------------------------------------

    print("")
    print("===================================")
    print("U-NET INPUT CHECK")
    print("===================================")

    print(
        "Model expected:",
        unet_model.input_shape
    )

    print(
        "Actual image array:",
        image_array.shape
    )

    print("===================================")


    # --------------------------------------------------------
    # U-NET PREDICTION
    # --------------------------------------------------------

    prediction = (
        unet_model.predict(
            image_array,
            verbose=0
        )[0, :, :, 0]
    )


    # --------------------------------------------------------
    # THRESHOLD
    # --------------------------------------------------------

    threshold = SEGMENTATION_THRESHOLD


    # --------------------------------------------------------
    # U-NET STATISTICS
    # --------------------------------------------------------

    print("")
    print("===================================")
    print("U-NET PREDICTION STATISTICS")
    print("===================================")

    print(
        "Prediction shape:",
        prediction.shape
    )

    print(
        "Minimum:",
        float(
            prediction.min()
        )
    )

    print(
        "Maximum:",
        float(
            prediction.max()
        )
    )

    print(
        "Mean:",
        float(
            prediction.mean()
        )
    )

    print(
        "Pixels > 0.10:",
        int(
            (prediction > 0.10).sum()
        )
    )

    print(
        "Pixels > 0.20:",
        int(
            (prediction > 0.20).sum()
        )
    )

    print(
        "Pixels > 0.30:",
        int(
            (prediction > 0.30).sum()
        )
    )

    print(
        "Pixels > 0.50:",
        int(
            (prediction > 0.50).sum()
        )
    )

    print(
        "Threshold:",
        threshold
    )

    print("===================================")


    # --------------------------------------------------------
    # BINARY MASK
    # --------------------------------------------------------

    predicted_binary = (
        prediction > threshold
    ).astype(
        np.uint8
    )


    # --------------------------------------------------------
    # RESIZE PROBABILITY MAP
    # --------------------------------------------------------

    probability_display = cv2.resize(
        prediction,
        original_size,
        interpolation=cv2.INTER_LINEAR
    )

    display_rgb = np.array(
        original_image,
        dtype=np.uint8
    )


    # --------------------------------------------------------
    # REMOVE BORDER ARTIFACT
    # --------------------------------------------------------

    border = max(
        2,
        min(original_size) // 100
    )

    probability_display[
        :border,
        :
    ] = 0

    probability_display[
        -border:,
        :
    ] = 0

    probability_display[
        :,
        :border
    ] = 0

    probability_display[
        :,
        -border:
    ] = 0


    # --------------------------------------------------------
    # RED PROBABILITY OVERLAY
    # --------------------------------------------------------

    excess = np.maximum(
        probability_display - threshold,
        0.0
    )

    max_excess = float(
        excess.max()
    )

    if max_excess > 0:

        alpha = np.clip(
            excess / max_excess,
            0.0,
            1.0
        )

    else:

        alpha = np.zeros_like(
            probability_display,
            dtype=np.float32
        )


    alpha = np.where(
        alpha > 0,
        0.25 + (0.55 * alpha),
        0.0
    ).astype(
        np.float32
    )


    red_overlay = np.zeros_like(
        display_rgb
    )

    red_overlay[:, :, 0] = 255

    alpha_3 = alpha[:, :, np.newaxis]

    blended = (
        display_rgb.astype(np.float32)
        *
        (1.0 - alpha_3)
        +
        red_overlay.astype(np.float32)
        *
        alpha_3
    )

    display_mask = np.clip(
        blended,
        0,
        255
    ).astype(
        np.uint8
    )


    # --------------------------------------------------------
    # DEFECT AREA
    # --------------------------------------------------------

    defect_area_percent = (
        np.mean(
            predicted_binary > 0
        )
        *
        100
    )


    # --------------------------------------------------------
    # SAVE VISUALIZATION
    # --------------------------------------------------------

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    mask_image = Image.fromarray(
        display_mask
    ).convert("RGB")

    mask_image.save(
        output_path
    )


    # --------------------------------------------------------
    # DEFAULT METRICS
    # --------------------------------------------------------

    iou = None
    dice = None


    # --------------------------------------------------------
    # GROUND-TRUTH COMPARISON
    # --------------------------------------------------------

    if ground_truth_path is not None:

        print("")
        print("===================================")
        print("GROUND-TRUTH COMPARISON")
        print("===================================")

        print(
            "Ground-truth mask:",
            ground_truth_path
        )


        # ----------------------------------------------------
        # LOAD MASK
        # ----------------------------------------------------

        ground_truth_image = Image.open(
            ground_truth_path
        ).convert("L")


        # ----------------------------------------------------
        # RESIZE MASK
        # ----------------------------------------------------

        ground_truth_image = (
            ground_truth_image.resize(
                UNET_INPUT_SIZE,
                Image.Resampling.NEAREST
            )
        )


        # ----------------------------------------------------
        # NUMPY
        # ----------------------------------------------------

        ground_truth_array = np.array(
            ground_truth_image
        )


        # ----------------------------------------------------
        # BINARIZE
        # ----------------------------------------------------

        ground_truth_binary = (
            ground_truth_array > 127
        ).astype(
            np.uint8
        )


        # ----------------------------------------------------
        # IoU
        # ----------------------------------------------------

        iou = calculate_iou(
            ground_truth_binary,
            predicted_binary
        )


        # ----------------------------------------------------
        # DICE
        # ----------------------------------------------------

        dice = calculate_dice(
            ground_truth_binary,
            predicted_binary
        )


        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        print(
            "IoU:",
            round(
                iou * 100,
                2
            ),
            "%"
        )

        print(
            "Dice:",
            round(
                dice * 100,
                2
            ),
            "%"
        )

        print("===================================")


    # --------------------------------------------------------
    # FINAL U-NET INFORMATION
    # --------------------------------------------------------

    print("")
    print("===================================")
    print("U-NET SEGMENTATION COMPLETE")
    print("===================================")

    print(
        "U-Net input size:",
        UNET_INPUT_SIZE
    )

    print(
        "Saved visualization:",
        output_path
    )

    print(
        "Defect area:",
        round(
            defect_area_percent,
            2
        ),
        "%"
    )

    if iou is not None:

        print(
            "Final IoU:",
            round(
                iou * 100,
                2
            ),
            "%"
        )

    else:

        print(
            "Final IoU: N/A"
        )

    if dice is not None:

        print(
            "Final Dice:",
            round(
                dice * 100,
                2
            ),
            "%"
        )

    else:

        print(
            "Final Dice: N/A"
        )

    print("===================================")


    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return {

        "output_path":
            str(
                output_path
            ),

        "defect_area_percent":
            round(
                float(
                    defect_area_percent
                ),
                2
            ),

        "iou":
            None
            if iou is None
            else round(
                float(
                    iou * 100
                ),
                2
            ),

        "dice":
            None
            if dice is None
            else round(
                float(
                    dice * 100
                ),
                2
            )
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@prediction_bp.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status":
            "success",

        "message":
            "TextileVision backend is running"

    })


# ============================================================
# PREDICTION API
# ============================================================

@prediction_bp.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # ========================================================
    # CHECK IMAGE
    # ========================================================

    if "image" not in request.files:

        return jsonify({

            "status":
                "error",

            "message":
                "No image uploaded"

        }), 400


    image = request.files["image"]


    if image.filename == "":

        return jsonify({

            "status":
                "error",

            "message":
                "No file selected"

        }), 400


    if not is_allowed_file(
        image.filename,
        ALLOWED_EXTENSIONS
    ):

        return jsonify({

            "status":
                "error",

            "message":
                "Unsupported image format"

        }), 400


    # ========================================================
    # SAVE IMAGE
    # ========================================================

    ensure_directory(
        UPLOAD_DIR
    )

    filename = secure_file_name(
        image.filename
    )

    save_path = (
        Path(UPLOAD_DIR) /
        filename
    )

    image.save(
        save_path
    )


    try:

        # ====================================================
        # 1. CNN
        # ====================================================

        result = predict_image(
            save_path
        )


        # ====================================================
        # 2. SVM
        # ====================================================

        svm_result = predict_svm(
            save_path
        )

        result.update(
            svm_result
        )


        # ====================================================
        # 3. FINAL DECISION
        # ====================================================

        final_status = determine_final_status(

            result["cnn_class"],

            result["svm_class"]

        )

        result["final_status"] = (
            final_status
        )

        result["inspection_status"] = (
            final_status
        )


        # ====================================================
        # PRINT FINAL DECISION
        # ====================================================

        print("")
        print("===================================")
        print("FINAL INSPECTION DECISION")
        print("===================================")

        print(
            "CNN CLASS:",
            result["cnn_class"]
        )

        print(
            "CNN CONFIDENCE:",
            result["confidence"],
            "%"
        )

        print(
            "CNN RAW SCORE:",
            result["raw_prediction"]
        )

        print("-----------------------------------")

        print(
            "SVM CLASS:",
            result["svm_class"]
        )

        print(
            "SVM CONFIDENCE:",
            result["svm_confidence"],
            "%"
        )

        print("-----------------------------------")

        print(
            "FINAL STATUS:",
            final_status
        )

        print("===================================")
        print("")


        # ====================================================
        # 4. GRAD-CAM
        # ====================================================

        gradcam_filename = (
            Path(filename).stem +
            "_gradcam.jpg"
        )

        gradcam_path = (
            GRADCAM_DIR /
            gradcam_filename
        )

        generate_gradcam(
            classifier_model,
            save_path,
            gradcam_path
        )

        result["gradcam_url"] = (
            "/results/gradcam/" +
            gradcam_filename
        )


        # ====================================================
        # 5. FIND GROUND TRUTH
        # ====================================================

        ground_truth_path = (
            find_ground_truth_mask(
                filename
            )
        )


        # ====================================================
        # 6. ALWAYS RUN U-NET
        # ====================================================

        mask_filename = (
            Path(filename).stem +
            "_mask.png"
        )

        mask_path = (
            MASK_DIR /
            mask_filename
        )


        # ====================================================
        # 7. U-NET SEGMENTATION
        # ====================================================

        segmentation_result = segment_image(

            save_path,

            mask_path,

            ground_truth_path

        )


        result["segmentation_url"] = (
            "/results/masks/" +
            mask_filename
        )

        result["defect_area"] = (
            segmentation_result[
                "defect_area_percent"
            ]
        )

        result["iou"] = (
            segmentation_result[
                "iou"
            ]
        )

        result["dice"] = (
            segmentation_result[
                "dice"
            ]
        )


        # ====================================================
        # 8. SEGMENTATION RESULTS
        # ====================================================

        print("")
        print("===================================")
        print("SEGMENTATION EVALUATION")
        print("===================================")

        print(
            "U-Net input size:",
            UNET_INPUT_SIZE
        )

        print(
            "IoU:",
            result["iou"]
        )

        print(
            "Dice:",
            result["dice"]
        )

        print(
            "Defect Area:",
            result["defect_area"],
            "%"
        )

        print("===================================")
        print("")


        # ====================================================
        # 9. FINAL JSON
        # ====================================================

        return jsonify({

            "status":
                "success",

            "message":
                "Image analyzed successfully",

            "filename":
                filename,

            "prediction":
                result

        })


    except Exception as error:

        print(
            "Prediction error:",
            error
        )

        return jsonify({

            "status":
                "error",

            "message":
                f"Prediction failed: {str(error)}"

        }), 500
