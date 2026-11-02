```python
import os

import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModelForImageClassification


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MODEL_ID = "hafeezaismail/fruit-freshness-convnext"

try:
    SECRET_MODEL_ID = st.secrets.get("HF_MODEL_ID", DEFAULT_MODEL_ID)
except Exception:
    SECRET_MODEL_ID = DEFAULT_MODEL_ID

MODEL_ID = os.getenv("HF_MODEL_ID", SECRET_MODEL_ID)

MAX_SIDE = 1024
LOW_CONFIDENCE = 0.50
MEDIUM_CONFIDENCE = 0.70
TOP_K = 5

st.set_page_config(
    page_title="Fruit Freshness Detector",
    page_icon="🍎",
    layout="centered",
)

torch.set_num_threads(2)


# ============================================================
# APPROXIMATE NUTRITION PER 100 GRAMS
# ============================================================

NUTRITION = {
    "apple": {
        "kcal": 52,
        "fiber_g": 2.4,
        "vitamin_c_mg": 4.6,
    },
    "banana": {
        "kcal": 89,
        "fiber_g": 2.6,
        "vitamin_c_mg": 8.7,
    },
    "grape": {
        "kcal": 69,
        "fiber_g": 0.9,
        "vitamin_c_mg": 3.2,
    },
    "guava": {
        "kcal": 68,
        "fiber_g": 5.4,
        "vitamin_c_mg": 228,
    },
    "jujube": {
        "kcal": 79,
        "fiber_g": 0.6,
        "vitamin_c_mg": 69,
    },
    "orange": {
        "kcal": 47,
        "fiber_g": 2.4,
        "vitamin_c_mg": 53,
    },
    "pomegranate": {
        "kcal": 83,
        "fiber_g": 4.0,
        "vitamin_c_mg": 10,
    },
    "strawberry": {
        "kcal": 32,
        "fiber_g": 2.0,
        "vitamin_c_mg": 59,
    },
}


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading AI model from Hugging Face...")
def load_model(model_id):
    processor = AutoImageProcessor.from_pretrained(model_id)

    model = AutoModelForImageClassification.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
    )

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return processor, model


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def prepare_image(uploaded_file):
    """
    Open the uploaded image safely, correct its orientation,
    convert it to RGB and limit its size.

    The Hugging Face image processor subsequently performs
    the model-specific resizing and normalization.
    """

    with Image.open(uploaded_file) as original:
        image = ImageOps.exif_transpose(original)
        image = image.convert("RGB")
        image.thumbnail((MAX_SIDE, MAX_SIDE))

        # Copy the image so it remains available after the
        # original file has been closed.
        return image.copy()


# ============================================================
# PREDICTION
# ============================================================

def predict(image, processor, model, top_k=TOP_K):
    """
    Return the top predictions using the model's class labels.
    """

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    with torch.inference_mode():
        outputs = model(**inputs)
        logits = outputs.logits

    probabilities = torch.softmax(logits, dim=-1)[0]

    k = min(top_k, probabilities.numel())

    top_results = torch.topk(probabilities, k=k)

    results = []

    for probability, index in zip(
        top_results.values,
        top_results.indices,
    ):
        label = model.config.id2label.get(
            index.item(),
            f"unknown_class_{index.item()}",
        )

        results.append(
            {
                "label": str(label),
                "confidence": float(probability.item()),
            }
        )

    return results


# ============================================================
# LABEL PROCESSING
# ============================================================

def split_label(label):
    """
    Expected labels:
        fresh_apple
        rotten_apple
        fresh_strawberry
        rotten_strawberry
    """

    status, separator, fruit = label.strip().lower().partition("_")

    if not separator:
        return "unknown", label.strip().lower()

    if status not in ("fresh", "rotten"):
        return "unknown", fruit

    return status, fruit


def readable_label(label):
    return label.replace("_", " ").title()


# ============================================================
# USER INTERFACE
# ============================================================

st.title("🍎 Fruit Freshness Detector")

st.write(
    "Upload a fruit photo or take a picture using your camera. "
    "The AI model predicts the fruit type and whether it looks "
    "fresh or rotten."
)

st.info(
    "This is an educational AI demo. Predictions may be incorrect, "
    "especially for unusual images, poor lighting or unfamiliar "
    "spoilage patterns. Do not use it as a food-safety test."
)


# Sidebar

with st.sidebar:
    st.header("About This Project")

    st.write(
        "**Model:** ConvNeXt-Tiny\n\n"
        "**Approach:** Transfer Learning\n\n"
        "**Dataset:** Fresh and Rotten Fruits\n\n"
        "**Supported fruits:** 8\n\n"
        "**Output classes:** 16"
    )

    st.markdown(
        f"[View model on Hugging Face]"
        f"(https://huggingface.co/{MODEL_ID})"
    )

    st.divider()

    input_mode = st.radio(
        "Choose image input",
        ["Upload Image", "Use Camera"],
    )

    st.caption(
        "The model is loaded once and cached to reduce "
        "repeated loading time."
    )


# ============================================================
# MODEL INITIALIZATION
# ============================================================

try:
    processor, model = load_model(MODEL_ID)

except Exception as exc:
    st.error(
        "Unable to load the AI model. Please check the Hugging "
        "Face repository name, internet connection and dependencies."
    )

    with st.expander("Technical error details"):
        st.code(str(exc))

    st.stop()


# ============================================================
# IMAGE INPUT
# ============================================================

if input_mode == "Upload Image":
    uploaded_file = st.file_uploader(
        "Choose a fruit photo",
        type=["jpg", "jpeg", "png", "webp"],
    )
else:
    uploaded_file = st.camera_input("Take a fruit photo")


# ============================================================
# RUN PREDICTION
# ============================================================

if uploaded_file is None:
    st.info("Upload or capture a fruit image to start prediction.")

else:
    try:
        image = prepare_image(uploaded_file)

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ):
        st.error(
            "The uploaded file could not be read as an image. "
            "Please upload a valid JPG, PNG or WEBP image."
        )
        st.stop()

    st.image(
        image,
        caption="Uploaded fruit image",
        use_container_width=True,
    )

    with st.spinner("Analysing fruit image..."):
        try:
            results = predict(image, processor, model)

        except Exception as exc:
            st.error("Prediction failed. Please try another image.")

            with st.expander("Technical error details"):
                st.code(str(exc))

            st.stop()

    if not results:
        st.error("The model did not return any predictions.")
        st.stop()

    # Best prediction

    best_result = results[0]

    best_label = best_result["label"]
    best_confidence = best_result["confidence"]

    status, fruit = split_label(best_label)

    confidence_percent = best_confidence * 100

    st.divider()

    st.header("🔍 Prediction Result")

    st.metric(
        label="Model Confidence",
        value=f"{confidence_percent:.1f}%",
    )

    # --------------------------------------------------------
    # CONFIDENCE-BASED MESSAGES
    # --------------------------------------------------------

    if status == "unknown":
        st.warning(
            f"The model returned an unrecognized label: "
            f"{readable_label(best_label)}. "
            "Please check the model's class labels."
        )

    elif best_confidence < LOW_CONFIDENCE:
        st.warning(
            f"⚠️ Low confidence. Best guess: "
            f"{readable_label(best_label)} "
            f"({confidence_percent:.1f}%). "
            "The prediction may be incorrect. "
            "Try a clearer image with one fruit centred."
        )

    elif best_confidence < MEDIUM_CONFIDENCE:
        st.info(
            f"🔎 Prediction: {readable_label(best_label)} "
            f"({confidence_percent:.1f}%). "
            "Confidence is moderate. Please verify the result "
            "visually; the model may be wrong."
        )

    elif status == "fresh":
        st.success(
            f"✅ Predicted fruit: {fruit.title()}\n\n"
            f"Freshness classification: Fresh\n\n"
            f"Confidence: {confidence_percent:.1f}%"
        )

    else:
        st.error(
            f"⚠️ Predicted fruit: {fruit.title()}\n\n"
            f"Freshness classification: Rotten\n\n"
            f"Confidence: {confidence_percent:.1f}%"
        )

    # --------------------------------------------------------
    # TOP PREDICTIONS
    # --------------------------------------------------------

    st.subheader("📊 Top 5 Predictions")

    st.caption(
        "These percentages represent the model's softmax scores. "
        "They are not guaranteed to be calibrated probabilities "
        "of correctness."
    )

    chart_df = pd.DataFrame(
        {
            "Class": [
                readable_label(result["label"])
                for result in results
            ],
            "Confidence (%)": [
                result["confidence"] * 100
                for result in results
            ],
        }
    )

    st.bar_chart(
        chart_df.set_index("Class"),
        y="Confidence (%)",
    )

    # Exact values are useful for debugging and demonstrations.

    with st.expander("View exact prediction scores"):
        display_df = chart_df.copy()

        display_df["Confidence (%)"] = display_df[
            "Confidence (%)"
        ].map(lambda value: f"{value:.2f}%")

        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # NUTRITION INFORMATION
    # --------------------------------------------------------

    if status == "fresh" and fruit in NUTRITION:
        st.divider()

        st.subheader(
            f"🥗 Approximate Nutrition: {fruit.title()}"
        )

        st.caption(
            "Approximate reference values per 100 g of fresh fruit. "
            "These values are not measured from the uploaded image."
        )

        nutrition = NUTRITION[fruit]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Calories",
            f"{nutrition['kcal']} kcal",
        )

        col2.metric(
            "Fiber",
            f"{nutrition['fiber_g']} g",
        )

        col3.metric(
            "Vitamin C",
            f"{nutrition['vitamin_c_mg']} mg",
        )

    elif status == "rotten":
        st.info(
            "Nutrition information is hidden because the model "
            "classified this fruit as rotten. Please verify the "
            "prediction visually."
        )

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    st.divider()

    st.caption(
        "Fruit Freshness Detector | ConvNeXt-Tiny | "
        "Hugging Face + Streamlit"
    )
```
