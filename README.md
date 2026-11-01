# 🍎 Fruit Freshness Detector

An end-to-end deep learning project: a ConvNeXt-Tiny image classifier, fine-tuned with transfer learning to
recognise **8 fruits** and decide whether each one is **fresh or rotten** (16 classes). The model is hosted on
Hugging Face and served through a Streamlit web app.

| | Link |
|---|---|
| Live app | `https://YOUR-APP-NAME.streamlit.app` |
| Fine-tuned model | `https://huggingface.co/hafeezaismail/fruit-freshness-convnext` |
| Base model | [facebook/convnext-tiny-224](https://huggingface.co/facebook/convnext-tiny-224) |
| Dataset | [Project-AgML/fresh_rotten_fruit_classification](https://huggingface.co/datasets/Project-AgML/fresh_rotten_fruit_classification) |

## Problem statement
Spoiled fruit is a major source of food waste for households, shops and supply chains. Manual inspection is slow
and subjective. This project builds an image classifier that checks fruit freshness from a single photo.

## Dataset
*Fresh and Rotten Fruits* (Sultana et al., 2022, Data in Brief), loaded from Hugging Face. The `raw` configuration
has 3,200 images (1,600 fresh and 1,600 rotten) across 8 fruit types. Fruit type and freshness are merged into one
label (for example `fresh_apple`), then split 70% train / 15% validation / 15% test, stratified by class.

## Method
1. Explore the data (class balance, image sizes, sample images).
2. Resize and normalise images; augment training data (random crop, flip, colour jitter).
3. Replace the 1000-class ImageNet head of `facebook/convnext-tiny-224` with a 16-class head.
4. Fine-tune for 6 epochs with AdamW (lr 5e-5, batch size 16, mixed precision) using Hugging Face `Trainer`.
5. Keep the best checkpoint by validation accuracy and evaluate on the untouched test set.
6. Push model and processor to the Hugging Face Hub and load them in the app.

## Results
| Metric (test set) | Value |
|---|---|
| Accuracy | 0.9917 (476 of 480 test images correct) |
| Macro F1 | 0.9916 |

Confusion matrix: see `confusion_matrix.png` on the model page.

## Project structure
```
.
├── app.py                      # Streamlit application
├── requirements.txt            # dependencies for Streamlit Community Cloud
├── notebooks/
│   └── train_fruit_freshness.ipynb   # data prep, fine-tuning, evaluation, Hub upload
├── docs/
│   ├── PROJECT_REPORT.md       # project report
│   └── DEPLOY_GUIDE.md         # step-by-step deployment guide
└── .streamlit/config.toml      # theme
```

## Run locally
```bash
git clone https://github.com/hafeezaismail/Project.git
cd Project
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Set your own model with `HF_MODEL_ID=hafeezaismail/fruit-freshness-convnext streamlit run app.py`,
or edit `DEFAULT_MODEL_ID` in `app.py`.

## Retrain the model
Open `notebooks/train_fruit_freshness.ipynb` in Google Colab (T4 GPU), set `HF_USERNAME`, and run all cells.

## Limitations
- Trained on a small dataset with clean backgrounds, so cluttered scenes can lower accuracy.
- Only the 8 fruits in the dataset are supported.
- A demo for learning, not a food-safety tool.

## Acknowledgements
Dataset: Sultana, Jahan and Uddin (2022), CC BY 4.0. Base model: Meta AI ConvNeXt, via Hugging Face.
