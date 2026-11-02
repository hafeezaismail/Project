import os

import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ---------------------------------------------------------------------------
# CONFIG: replace with your own Hugging Face repo (username/model-name).
# You can also set it as an environment variable or Streamlit secret HF_MODEL_ID.
# ---------------------------------------------------------------------------
DEFAULT_MODEL_ID = "hafeezaismail/fruit-freshness-convnext"
MODEL_ID = os.getenv("HF_MODEL_ID")
if not MODEL_ID:
    try:
        MODEL_ID = st.secrets.get("HF_MODEL_ID", DEFAULT_MODEL_ID)
    except Exception:  # no secrets file present
        MODEL_ID = DEFAULT_MODEL_ID

# Approximate nutrition per 100 g of fresh fruit (general reference values).
NUTRITION = {
    "apple": {"kcal": 52, "fiber_g": 2.4, "vitamin_c_mg": 4.6},
    "banana": {"kcal": 89, "fiber_g": 2.6, "vitamin_c_mg": 8.7},
    "grape": {"kcal": 69, "fiber_g": 0.9, "vitamin_c_mg": 3.2},
    "guava": {"kcal": 68, "fiber_g": 5.4, "vitamin_c_mg": 228},
    "jujube": {"kcal": 79, "fiber_g": 0.6, "vitamin_c_mg": 69},
    "orange": {"kcal": 47, "fiber_g": 2.4, "vitamin_c_mg": 53},
    "pomegranate": {"kcal": 83, "fiber_g": 4.0, "vitamin_c_mg": 10},
    "strawberry": {"kcal": 32, "fiber_g": 2.0, "vitamin_c_mg": 59},
}

st.set_page_config(page_title="Fruit Freshness Detector", page_icon="🍎", layout="centered")

# Keep CPU/RAM use small: Streamlit Community Cloud free tier is limited (documented as ~1 GB RAM).
torch.set_num_threads(2)
MAX_SIDE = 1024  # large phone photos are shrunk before use to avoid memory spikes


@st.cache_resource(show_spinner="Loading model from Hugging Face...")
def load_model(model_id: str):
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id, low_cpu_mem_usage=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return processor, model


def predict(image: Image.Image, processor, model, top_k: int = 5):
    inputs = processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    top = torch.topk(probs, k=min(top_k, probs.numel()))
    return [
        (model.config.id2label[idx.item()], prob.item())
        for prob, idx in zip(top.values, top.indices)
    ]


def split_label(label: str):
    status, _, fruit = label.partition("_")
    return status, fruit


# ------------------------------- UI ----------------------------------------
st.title("🍎 Fruit Freshness Detector")
st.write(
    "Upload or capture a photo of a fruit and the model will tell you **which fruit it is** "
    "and whether it looks **fresh or rotten**."
)

with st.sidebar:
    st.header("About")
    st.write(
        "Fine-tuned **ConvNeXt-Tiny** (transfer learning) on the "
        "*Fresh and Rotten Fruits* dataset (8 fruits, 3,200 images)."
    )
    st.markdown(f"Model: [{MODEL_ID}](https://huggingface.co/{MODEL_ID})")
    st.caption("Demo project. Do not use it as a food-safety check.")
    input_mode = st.radio("Input", ["Upload image", "Use camera"])

try:
    processor, model = load_model(MODEL_ID)
except Exception as exc:  # noqa: BLE001
    st.error(
        f"Could not load the model `{MODEL_ID}`. Check that the repo name is correct and public.\n\n{exc}"
    )
    st.stop()

if input_mode == "Upload image":
    file = st.file_uploader("Choose a fruit photo", type=["jpg", "jpeg", "png", "webp"])
else:
    file = st.camera_input("Take a photo of a fruit")

if file is not None:
    img = Image.open(file)
    img.draft("RGB", (MAX_SIDE * 2, MAX_SIDE * 2))  # JPEG: decode at reduced size
    image = ImageOps.exif_transpose(img).convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    st.image(image, caption="Your image", use_container_width=True)

    with st.spinner("Analysing..."):
        results = predict(image, processor, model)

    best_label, best_prob = results[0]
    status, fruit = split_label(best_label)

    if best_prob < 0.5:
        st.warning(
            f"The model is not sure (best guess: {status} {fruit}, {best_prob:.0%}). "
            "Try a clearer photo with the fruit in the centre."
        )
    elif status == "fresh":
        st.success(f"✅ Fresh {fruit} ({best_prob:.1%} confidence)")
    else:
        st.error(f"⚠️ Rotten {fruit} ({best_prob:.1%} confidence)")

    st.subheader("Top predictions")
    chart_df = pd.DataFrame(
        {"class": [r[0].replace("_", " ") for r in results], "confidence": [r[1] for r in results]}
    ).set_index("class")
    st.bar_chart(chart_df)

    info = NUTRITION.get(fruit.lower())
    if info and status == "fresh":
        st.subheader(f"Nutrition of {fruit} (per 100 g, approx.)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Calories", f"{info['kcal']} kcal")
        c2.metric("Fiber", f"{info['fiber_g']} g")
        c3.metric("Vitamin C", f"{info['vitamin_c_mg']} mg")
else:
    st.info("Waiting for an image...")
