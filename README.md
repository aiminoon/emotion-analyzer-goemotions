# Emotion Analyzer for Text

A fine-tuned DistilRoBERTa model for classifying text into 7 Ekman emotion categories, with integrated sarcasm detection and explainability. Built on the GoEmotions dataset and deployed as an interactive Streamlit dashboard.

## Overview

Traditional sentiment analysis is limited to positive/negative/neutral outputs, which fails to capture the nuance of real-world text. This project classifies text into seven discrete emotions — **anger, disgust, fear, joy, sadness, surprise, and neutral** — and pairs predictions with token-level explanations and sarcasm confidence scores.

## System Architecture

1. **User Input** — raw text
2. **Preprocessing** — tokenization, truncation/padding (max length 128), special tokens, class weighting
3. **Emotion Classification** — fine-tuned DistilRoBERTa (primary), with Hartmann DistilRoBERTa and TF-IDF + Logistic Regression as baselines
4. **Sarcasm Detection** — RoBERTa-based sarcasm classifier
5. **Output & Dashboard** — predicted emotion, class probabilities, token attribution heatmaps, sarcasm prediction — visualized in Streamlit with Plotly

## Dataset

- **Source:** Google [GoEmotions](https://doi.org/10.18653/v1/2020.acl-main.372) — 58,000+ Reddit comments
- **Label remapping:** original 27 emotion labels → 7 Ekman categories
- **Split:** 43,410 train / 5,426 validation / 5,427 test
- Class weighting applied to reduce the impact of label imbalance

## Model Development

- **Primary model:** DistilRoBERTa (82M parameters), fine-tuned on GoEmotions
- **Baselines:** Hartmann (zero-shot) DistilRoBERTa, TF-IDF + Logistic Regression
- **Training config:** learning rate 2e-5, batch size 16, up to 5 epochs, weighted cross-entropy loss, early stopping
- **Explainability:** Captum Integrated Gradients for token attribution heatmaps
- **Deployment:** Streamlit + Plotly

## Results

Best checkpoint selected at Epoch 1 based on validation loss (training continued to epoch 5, but validation loss began climbing after epoch 1, indicating overfitting).

| Epoch | Training Loss | Validation Loss | Accuracy |
|-------|---------------|------------------|----------|
| 1     | 1.088         | 1.032            | 62.6%    |
| 2     | 0.998         | 1.048            | 63.9%    |
| 3     | 0.845         | 1.084            | 64.3%    |
| 4     | 0.676         | 1.152            | 65.3%    |
| 5     | 0.599         | 1.232            | 65.9%    |

The fine-tuned DistilRoBERTa model outperformed both baselines across all emotion categories, reaching an **0.81 F1-score for Joy** (best-performing class).

**Key error patterns (from confusion matrix analysis):**
- *Disgust ↔ Anger* — frequent confusion, likely due to overlapping aggressive/toxic language patterns
- *Neutral → Surprise/Anger* — model sensitivity to punctuation and emphatic phrasing

## Strengths & Limitations

| Strengths | Limitations |
|---|---|
| 7-category emotion classification | English-only support |
| Lightweight DistilRoBERTa architecture | Class imbalance (e.g. Fear: 618, Disgust: 579 samples) |
| Explainable predictions (Integrated Gradients) | Lower performance on minority emotions |
| Additional sarcasm detection layer | Potential error propagation between sarcasm and emotion modules |

## Future Improvements

- Data augmentation for minority emotion classes
- Multilingual support
- Larger architectures (RoBERTa-large, DeBERTa)
- Joint sarcasm-emotion learning to reduce cascading errors
- Real-time social media API integration

## Tech Stack

`Python` · `PyTorch` · `Hugging Face Transformers` · `DistilRoBERTa` · `Captum` · `Streamlit` · `Plotly` · `scikit-learn`

## Team

This project was developed as a group assignment for **WID3002 Natural Language Processing**, Faculty of Computer Science & IT, Universiti Malaya, under Dr. Mohamed N.M. Lubani.

- Nur Qistina Allysha Binti Mohd Joharizal
- Aisya Saffiyah Binti Kamarul Nizam
- Muhammad Aiman Bin Sharuddin
- Dennis Aimin Oon Bin Jeffrey Oon (Me)
- Muhammad Imran Bin Ilias
- Fairuz Anika Mysha

## References

- Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020). *GoEmotions: A dataset of fine-grained emotions.* ACL. https://doi.org/10.18653/v1/2020.acl-main.372
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT.* arXiv. https://doi.org/10.48550/arXiv.1910.01108
- Sundararajan, M., Taly, A., & Yan, Q. (2017). *Axiomatic attribution for deep networks.* ICML. https://proceedings.mlr.press/v70/sundararajan17a.html
