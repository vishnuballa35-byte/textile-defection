from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
import cv2


def find_mobilenet_base_model(model):
    """
    Find the nested MobileNetV2 model inside the classifier.
    """

    for layer in model.layers:

        if isinstance(layer, tf.keras.Model):

            print("Found base model:", layer.name)

            return layer

    raise ValueError(
        "Could not find the MobileNetV2 base model."
    )


def find_last_conv_layer(base_model):
    """
    Find the last convolutional layer inside MobileNetV2.
    """

    for layer in reversed(base_model.layers):

        if isinstance(
            layer,
            tf.keras.layers.Conv2D
        ):

            return layer

    raise ValueError(
        "Could not find a Conv2D layer."
    )


def generate_gradcam(
    model,
    image_path,
    output_path,
    image_size=(224, 224)
):

    # -------------------------------------------------
    # Load original image
    # -------------------------------------------------

    original_image = Image.open(
        image_path
    ).convert("RGB")

    original_array = np.array(
        original_image
    )

    # -------------------------------------------------
    # Prepare image
    # -------------------------------------------------

    image = original_image.resize(
        image_size
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = (
        tf.keras.applications.mobilenet_v2
        .preprocess_input(image_array)
    )

    image_array = tf.expand_dims(
        image_array,
        axis=0
    )

    # -------------------------------------------------
    # Find MobileNetV2
    # -------------------------------------------------

    base_model = find_mobilenet_base_model(
        model
    )

    last_conv_layer = find_last_conv_layer(
        base_model
    )

    print(
        "Grad-CAM layer:",
        last_conv_layer.name
    )

    # -------------------------------------------------
    # Grad-CAM model
    # -------------------------------------------------

    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[
            last_conv_layer.output,
            base_model.output
        ]
    )

    # -------------------------------------------------
    # Calculate gradients
    # -------------------------------------------------

    with tf.GradientTape() as tape:

        conv_outputs, base_output = grad_model(
            image_array,
            training=False
        )

        x = base_output

        # Pass through classifier head
        for layer in model.layers:

            # Skip the MobileNetV2 base model
            if layer is base_model:
                continue

            x = layer(
                x,
                training=False
            )

        prediction = x[:, 0]

    # -------------------------------------------------
    # Gradients
    # -------------------------------------------------

    gradients = tape.gradient(
        prediction,
        conv_outputs
    )

    if gradients is None:

        raise ValueError(
            "Gradients could not be calculated."
        )

    # -------------------------------------------------
    # Pool gradients
    # -------------------------------------------------

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = (
        conv_outputs *
        pooled_gradients
    )

    heatmap = tf.reduce_sum(
        heatmap,
        axis=-1
    )

    # -------------------------------------------------
    # Normalize
    # -------------------------------------------------

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = tf.where(
        max_value > 0,
        heatmap / max_value,
        heatmap
    )

    heatmap = heatmap.numpy()

    # -------------------------------------------------
    # Resize heatmap
    # -------------------------------------------------

    heatmap = cv2.resize(
        heatmap,
        (
            original_array.shape[1],
            original_array.shape[0]
        )
    )

    heatmap_uint8 = np.uint8(
        heatmap * 255
    )

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    # -------------------------------------------------
    # Original image RGB → BGR
    # -------------------------------------------------

    original_bgr = cv2.cvtColor(
        original_array,
        cv2.COLOR_RGB2BGR
    )

    # -------------------------------------------------
    # Overlay
    # -------------------------------------------------

    overlay = cv2.addWeighted(
        original_bgr,
        0.60,
        heatmap_color,
        0.40,
        0
    )

    # -------------------------------------------------
    # Save
    # -------------------------------------------------

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        str(output_path),
        overlay
    )

    print(
        "Grad-CAM saved:",
        output_path
    )

    return {
        "prediction": float(
            prediction.numpy()[0]
        ),
        "output_path": str(
            output_path
        ),
        "last_conv_layer": last_conv_layer.name
    }