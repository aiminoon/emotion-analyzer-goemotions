# emotion_dashboard.py
# Run: streamlit run emotion_dashboard.py

import pickle
import numpy as np
import torch
import streamlit as st
import plotly.graph_objects as go
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)
from captum.attr import LayerIntegratedGradients

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion & Sarcasm Analyzer",
    page_icon="🎭",
    layout="wide",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMOTION_COLORS = {
    "anger":    "#e74c3c",
    "disgust":  "#8e44ad",
    "fear":     "#2c3e50",
    "joy":      "#f1c40f",
    "sadness":  "#2980b9",
    "surprise": "#e67e22",
    "neutral":  "#95a5a6",
}

EMOTION_MODEL_PATH = "./models/ekman-emotion"
SARCASM_MODEL_CKPT = "jkhan447/sarcasm-detection-RoBerta-base-CR"
HARTMANN_CKPT      = "j-hartmann/emotion-english-distilroberta-base"
CLASSICAL_PKL      = "./models/classical_models.pkl"
MAX_LEN            = 128


# ─── Embedding layer auto-detect ──────────────────────────────────────────────
def get_embed_layer(model):
    for attr in ["roberta", "distilbert", "bert", "electra", "deberta"]:
        if hasattr(model, attr):
            return getattr(model, attr).embeddings
    raise AttributeError(
        f"Can't find embedding layer. Model type: {type(model).__name__}. "
        f"Available attrs: {[a for a in dir(model) if not a.startswith('_')]}"
    )


# ─── Load models ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_emotion_model():
    tok   = AutoTokenizer.from_pretrained(EMOTION_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL_PATH).to(DEVICE)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    return tok, model, id2label


@st.cache_resource
def load_sarcasm_model():
    tok   = AutoTokenizer.from_pretrained(SARCASM_MODEL_CKPT)
    model = AutoModelForSequenceClassification.from_pretrained(SARCASM_MODEL_CKPT).to(DEVICE)
    model.eval()
    return tok, model


@st.cache_resource
def load_hartmann():
    return pipeline(
        "text-classification",
        model=HARTMANN_CKPT,
        top_k=None,
        device=0 if torch.cuda.is_available() else -1,
        truncation=True,
        max_length=MAX_LEN,
    )


@st.cache_resource
def load_classical():
    with open(CLASSICAL_PKL, "rb") as f:
        return pickle.load(f)


emotion_tok, emotion_mdl, ID2LABEL = load_emotion_model()
sarcasm_tok, sarcasm_mdl           = load_sarcasm_model()
hartmann_pipe                      = load_hartmann()
classical_bundle                   = load_classical()
tfidf                              = classical_bundle["tfidf"]
classical_models                   = classical_bundle["models"]


# ─── Inference ────────────────────────────────────────────────────────────────
def predict_emotion(text):
    enc = emotion_tok(
        text, return_tensors="pt",
        max_length=MAX_LEN, truncation=True, padding="max_length",
    )
    input_ids      = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits = emotion_mdl(input_ids=input_ids, attention_mask=attention_mask).logits
    probs      = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    pred_class = int(np.argmax(probs))

    embed_layer = get_embed_layer(emotion_mdl)

    def fwd(ids):
        # ids may be [n_steps, seq_len] during IG interpolation — expand mask to match
        batch_size = ids.shape[0]
        mask = attention_mask.expand(batch_size, -1)
        return emotion_mdl(input_ids=ids, attention_mask=mask).logits[:, pred_class]

    lig      = LayerIntegratedGradients(fwd, embed_layer)
    baseline = torch.full_like(input_ids, emotion_tok.pad_token_id)
    baseline[input_ids == emotion_tok.cls_token_id] = emotion_tok.cls_token_id
    baseline[input_ids == emotion_tok.sep_token_id] = emotion_tok.sep_token_id

    attr, _ = lig.attribute(
        input_ids, baselines=baseline,
        n_steps=30, return_convergence_delta=True,
    )
    attr = attr.sum(dim=-1).squeeze(0) * attention_mask.squeeze(0).float()
    attr = torch.abs(attr).cpu().numpy()
    if attr.max() > attr.min():
        attr = (attr - attr.min()) / (attr.max() - attr.min())

    ids_list   = input_ids.squeeze(0).cpu().tolist()
    tokens_raw = emotion_tok.convert_ids_to_tokens(ids_list)
    mask_list  = attention_mask.squeeze(0).cpu().tolist()
    special    = {
        emotion_tok.cls_token, emotion_tok.sep_token,
        emotion_tok.pad_token, "<s>", "</s>", "<pad>",
    }
    token_attrs = [
        (t.replace("Ġ", "").replace("▁", ""), float(a))
        for t, a, m in zip(tokens_raw, attr, mask_list)
        if m == 1 and t not in special and t.strip()
    ]

    emotions = [(ID2LABEL[i], float(probs[i])) for i in range(len(probs))]
    emotions.sort(key=lambda x: x[1], reverse=True)
    return emotions, token_attrs


def detect_sarcasm(text):
    inputs = sarcasm_tok(
        text, return_tensors="pt",
        truncation=True, max_length=MAX_LEN, padding="max_length",
    ).to(DEVICE)
    with torch.no_grad():
        logits = sarcasm_mdl(**inputs).logits
        probs  = torch.nn.functional.softmax(logits, dim=-1)
    sarc_score = probs[0, 1].item()
    label      = "🙃 Sarcasm" if sarc_score > 0.5 else "✅ Not Sarcasm"
    return label, round(sarc_score, 4)


def predict_hartmann(text):
    raw      = hartmann_pipe(text)[0]
    emotions = [(r["label"].lower(), round(r["score"], 4)) for r in raw]
    emotions.sort(key=lambda x: x[1], reverse=True)
    return emotions


def classical_predict(text):
    vec        = tfidf.transform([text])
    n_labels   = len(ID2LABEL)
    feat_names = np.array(tfidf.get_feature_names_out())
    tfidf_vals = vec.toarray()[0]
    results    = {}

    for name, clf in classical_models.items():
        pred      = clf.predict(vec)[0]
        raw_probs = clf.predict_proba(vec)[0]

        probs_arr = np.zeros(n_labels)
        for col_idx, class_id in enumerate(clf.classes_):
            probs_arr[int(class_id)] = raw_probs[col_idx]

        emotions = [(ID2LABEL[i], float(probs_arr[i])) for i in range(n_labels)]
        emotions.sort(key=lambda x: x[1], reverse=True)

        base_clf = clf.estimator if hasattr(clf, "estimator") else clf
        if hasattr(base_clf, "coef_"):
            coefs  = base_clf.coef_[pred] if base_clf.coef_.ndim > 1 else base_clf.coef_[0]
            scores = tfidf_vals * np.abs(coefs)
        elif hasattr(base_clf, "feature_importances_"):
            scores = tfidf_vals * base_clf.feature_importances_
        else:
            scores = tfidf_vals

        non_zero = np.nonzero(scores)[0]
        if len(non_zero) > 0:
            top_idx     = non_zero[np.argsort(scores[non_zero])[::-1][:12]]
            top_s       = scores[top_idx]
            if top_s.max() > 0:
                top_s = top_s / top_s.max()
            token_attrs = list(zip(feat_names[top_idx].tolist(), top_s.tolist()))
        else:
            token_attrs = []

        results[name] = {"emotions": emotions, "token_attrs": token_attrs}
    return results


# ─── Plotly helpers ───────────────────────────────────────────────────────────
def confidence_bar(emotions, title=""):
    labels = [e[0] for e in emotions]
    scores = [e[1] for e in emotions]
    colors = [EMOTION_COLORS.get(l, "#888") for l in labels]
    fig = go.Figure(go.Bar(
        x=scores, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{s:.1%}" for s in scores],
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        xaxis_range=[0, 1.15],
        xaxis_title="Confidence",
        height=300,
        margin=dict(l=10, r=20, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def attribution_chart(token_attrs, title="Token Attribution"):
    if not token_attrs:
        return None
    tokens, scores = zip(*token_attrs)
    fig = go.Figure(go.Bar(
        x=list(tokens), y=list(scores),
        marker=dict(
            color=list(scores),
            colorscale="YlOrRd",
            cmin=0, cmax=1,
            colorbar=dict(title="Attribution", thickness=12),
        ),
    ))
    fig.update_layout(
        title=title,
        yaxis_title="Attribution Score",
        yaxis_range=[0, 1.1],
        height=280,
        margin=dict(l=10, r=10, t=40, b=50),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─── UI ───────────────────────────────────────────────────────────────────────
st.title("🎭 Emotion & Sarcasm Analyzer")
st.markdown(
    "Enter any English text to see **Ekman emotion predictions** from three models, "
    "**token-level attribution**, and **sarcasm detection**."
)

with st.form(key="analysis_form"):
    user_text = st.text_area(
        "Enter text",
        placeholder="e.g. Oh sure, another meeting that could have been an email!",
        height=100,
    )
    submitted = st.form_submit_button("Analyze", type="primary")

if submitted and user_text.strip():
    with st.spinner("Running models..."):
        emotions, token_attrs = predict_emotion(user_text)
        sarcasm_label, sarc_conf = detect_sarcasm(user_text)
        hartmann_emotions        = predict_hartmann(user_text)
        classical_results        = classical_predict(user_text)

    top_em   = emotions[0][0]
    top_conf = emotions[0][1]
    em_color = EMOTION_COLORS.get(top_em, "#888")

    st.markdown(f"""
    <div style="background:{em_color}22;border-left:5px solid {em_color};
                padding:12px 18px;border-radius:6px;margin-bottom:12px;">
        <h3 style="margin:0;color:{em_color}">
            {top_em.upper()} &nbsp;
            <span style="font-size:0.8em;color:#555">{top_conf:.1%} confidence</span>
        </h3>
        <p style="margin:4px 0 0;color:#555">
            Sarcasm: <b>{sarcasm_label}</b> &nbsp;({sarc_conf:.1%})
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── DistilRoBERTa (your fine-tuned model) ─────────────────────────────────
    st.subheader("🤖 DistilRoBERTa (fine-tuned on GoEmotions → Ekman)")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            confidence_bar(emotions, "Emotion Confidence"),
            use_container_width=True,
        )
    with c2:
        fig_attr = attribution_chart(token_attrs, "Token Attribution (Integrated Gradients)")
        if fig_attr:
            st.plotly_chart(fig_attr, use_container_width=True)

    # ── Hartmann pretrained ───────────────────────────────────────────────────
    st.subheader("🔬 Hartmann (pretrained DistilRoBERTa — zero-shot baseline)")
    st.plotly_chart(
        confidence_bar(hartmann_emotions, "Emotion Confidence"),
        use_container_width=True,
    )

    # ── Classical baselines ───────────────────────────────────────────────────
    st.subheader("📊 Classical ML Baselines")
    cols = st.columns(len(classical_results))
    for col, (name, res) in zip(cols, classical_results.items()):
        with col:
            st.markdown(f"**{name}**")
            top = res["emotions"][0]
            st.metric(label=top[0], value=f"{top[1]:.1%}")
            st.plotly_chart(
                confidence_bar(res["emotions"][:6], ""),
                use_container_width=True,
            )
            fig_ca = attribution_chart(res["token_attrs"][:8], "Top contributing n-grams")
            if fig_ca:
                st.plotly_chart(fig_ca, use_container_width=True)

    # ── Raw data ──────────────────────────────────────────────────────────────
    with st.expander("Raw prediction data"):
        st.json({
            "distilroberta_finetuned": {
                "emotions":    {e: round(c, 4) for e, c in emotions},
                "token_attrs": {t: round(s, 4) for t, s in token_attrs},
            },
            "hartmann": {e: round(c, 4) for e, c in hartmann_emotions},
            "sarcasm":  {"label": sarcasm_label, "confidence": sarc_conf},
            "classical": {
                name: {
                    "top":        res["emotions"][0][0],
                    "confidence": round(res["emotions"][0][1], 4),
                }
                for name, res in classical_results.items()
            },
        })

elif submitted:
    st.warning("Please enter some text first.")