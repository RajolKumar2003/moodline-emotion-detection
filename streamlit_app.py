from pathlib import Path
import html
import pickle
import re

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Artifacts" / "BiGRU_Model.keras"
TOKENIZER_PATH = BASE_DIR / "Artifacts" / "tokenizer.pkl"

MAX_SEQUENCE_LENGTH = 50
MAX_TEXT_LENGTH = 2000

EMOTION_LABELS = [
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise",
]

EMOTION_EMOJIS = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}

ACCENTS = {
    "sadness": ("#5b7fde", "rgba(91,127,222,0.16)"),
    "joy": ("#f2b705", "rgba(242,183,5,0.16)"),
    "love": ("#e85d75", "rgba(232,93,117,0.16)"),
    "anger": ("#e4572e", "rgba(228,87,46,0.16)"),
    "fear": ("#8b5fbf", "rgba(139,95,191,0.16)"),
    "surprise": ("#17bebb", "rgba(23,190,187,0.16)"),
}


st.set_page_config(
    page_title="Moodline — read the emotion in your words",
    page_icon="✎",
    layout="centered",
    initial_sidebar_state="collapsed",
)


def inject_styles():
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;1,9..144,500;1,9..144,600&family=JetBrains+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg: #0e0f14;
    --surface: #16171f;
    --surface-2: #1c1e29;
    --border: rgba(255,255,255,0.08);
    --text: #f2f0ea;
    --text-dim: #9497a6;
    --text-faint: #5c5f6d;
    --accent: #6c7a89;
    --accent-soft: rgba(108,122,137,0.18);
    --font-display: "Fraunces", serif;
    --font-body: "Space Grotesk", sans-serif;
    --font-mono: "JetBrains Mono", monospace;
}

html,
body,
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(
            60% 50% at 50% -10%,
            var(--accent-soft),
            transparent 70%
        ),
        var(--bg);
    color: var(--text);
    font-family: var(--font-body);
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
#MainMenu,
footer {
    visibility: hidden;
}

[data-testid="stAppViewBlockContainer"] {
    max-width: 960px;
    padding-top: 48px;
    padding-bottom: 40px;
}

.topbar {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 16px;
    flex-wrap: wrap;
    margin-bottom: 56px;
}

.brand {
    display: flex;
    align-items: center;
    gap: 10px;
}

.brand-mark {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: #52d17c;
    box-shadow: 0 0 0 3px rgba(82,209,124,0.18);
}

.brand-name {
    font-family: var(--font-display);
    font-weight: 600;
    font-size: 20px;
}

.brand-tag {
    margin: 0;
    color: var(--text-dim);
    font-size: 13px;
}

.eyebrow {
    display: block;
    font-family: var(--font-mono);
    font-size: 11px;
    letter-spacing: 0.14em;
    color: var(--text-faint);
    margin-bottom: 14px;
}

.st-key-console_panel {
    padding: 40px;
    border: 1px solid var(--border);
    border-radius: 28px;
    background: linear-gradient(
        180deg,
        var(--surface),
        var(--surface-2)
    );
}

.orb-stage {
    position: relative;
    height: 220px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.orb-glow {
    position: absolute;
    inset: 10%;
    border-radius: 50%;
    background: radial-gradient(
        circle,
        var(--accent) 0%,
        transparent 72%
    );
    filter: blur(26px);
    opacity: 0.55;
    animation: breathe 6s ease-in-out infinite;
}

.orb {
    position: relative;
    width: 148px;
    height: 148px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(
        135deg,
        var(--accent),
        color-mix(in srgb, var(--accent) 55%, #0e0f14)
    );
    border-radius: 42% 58% 65% 35% / 48% 42% 58% 52%;
    box-shadow:
        inset 0 0 30px rgba(255,255,255,0.14),
        0 20px 40px -18px var(--accent);
    animation:
        morph 9s ease-in-out infinite,
        breathe 6s ease-in-out infinite;
}

.orb-emoji {
    font-size: 38px;
}

.st-key-console_panel [data-testid="stTextArea"] textarea {
    min-height: 118px !important;
    background: rgba(0,0,0,0.18) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 16px 18px !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
    font-size: 16px !important;
    line-height: 1.55 !important;
    box-shadow: none !important;
}

.st-key-console_panel [data-testid="stTextArea"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 4px var(--accent-soft) !important;
}

.st-key-console_panel [data-testid="stWidgetLabel"] {
    display: none;
}

.st-key-char_count {
    display: flex;
    align-items: center;
    min-height: 44px;
    color: var(--text-faint);
    font-family: var(--font-mono);
    font-size: 12px;
}

.st-key-analyze_button button {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-height: 44px;
    padding: 12px 20px;
    border-radius: 999px;
    border: none;
    background: var(--text);
    color: #0e0f14;
    font-family: var(--font-body);
    font-weight: 600;
    font-size: 14.5px;
}

.st-key-analyze_button button:hover {
    transform: translateY(-2px);
    background: var(--accent);
    color: #ffffff;
}

.result-grid {
    margin-top: 32px;
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 24px;
}

.result-card {
    min-width: 0;
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 32px;
    background: linear-gradient(
        180deg,
        var(--surface),
        var(--surface-2)
    );
}

.reveal-main {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 10px;
}

.reveal-emoji {
    font-size: 40px;
    line-height: 1;
    flex-shrink: 0;
}

.emotion-word {
    margin: 0;
    font-family: var(--font-display);
    font-style: italic;
    font-weight: 600;
    font-size: clamp(36px, 5vw, 52px);
    color: var(--accent);
    white-space: nowrap;
}

.confidence-mono {
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 13px;
    color: var(--text-dim);
    margin-bottom: 20px;
}

.echoed-text {
    margin: 0;
    padding-top: 18px;
    border-top: 1px solid var(--border);
    font-size: 14.5px;
    line-height: 1.6;
    color: var(--text-dim);
    font-style: italic;
}

.bars {
    display: flex;
    flex-direction: column;
    gap: 14px;
}

.bar-row {
    display: grid;
    grid-template-columns: 90px minmax(0, 1fr) 52px;
    align-items: center;
    gap: 12px;
}

.bar-label {
    font-size: 13px;
    color: var(--text-dim);
    white-space: nowrap;
    text-transform: capitalize;
}

.bar-track {
    height: 8px;
    border-radius: 999px;
    background: rgba(255,255,255,0.06);
    overflow: hidden;
}

.bar-fill {
    display: block;
    height: 100%;
    border-radius: 999px;
}

.bar-sadness {
    background: #5b7fde;
}

.bar-joy {
    background: #f2b705;
}

.bar-love {
    background: #e85d75;
}

.bar-anger {
    background: #e4572e;
}

.bar-fear {
    background: #8b5fbf;
}

.bar-surprise {
    background: #17bebb;
}

.bar-pct {
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--text-faint);
    text-align: right;
}

.footbar {
    margin-top: 40px;
    display: flex;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
    font-size: 12px;
    color: var(--text-faint);
    font-family: var(--font-mono);
}

.streamlit-error {
    margin: 12px 2px 0;
    font-size: 13px;
    color: #ff8a72;
    font-family: var(--font-mono);
}

@keyframes breathe {
    0%, 100% {
        transform: scale(1);
    }

    50% {
        transform: scale(1.05);
    }
}

@keyframes morph {
    0%, 100% {
        border-radius:
            42% 58% 65% 35% /
            48% 42% 58% 52%;
    }

    33% {
        border-radius:
            60% 40% 44% 56% /
            40% 62% 38% 60%;
    }

    66% {
        border-radius:
            48% 52% 38% 62% /
            62% 40% 60% 38%;
    }
}

@media (max-width: 720px) {
    [data-testid="stAppViewBlockContainer"] {
        padding-left: 20px;
        padding-right: 20px;
    }

    .st-key-console_panel {
        padding: 28px;
    }

    .result-grid {
        grid-template-columns: 1fr;
    }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def set_accent(emotion=None):
    accent, accent_soft = ACCENTS.get(
        emotion,
        ("#6c7a89", "rgba(108,122,137,0.18)"),
    )

    st.markdown(
        f"""
<style>
:root {{
    --accent: {accent};
    --accent-soft: {accent_soft};
}}
</style>
        """,
        unsafe_allow_html=True,
    )


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text


@st.cache_resource(show_spinner=False)
def load_assets():
    model = load_model(
        MODEL_PATH,
        compile=False,
    )

    with TOKENIZER_PATH.open("rb") as file:
        tokenizer = pickle.load(file)

    return model, tokenizer


def predict_emotion(text, model, tokenizer):
    cleaned_text = preprocess_text(text)

    tokenized_text = tokenizer.texts_to_sequences(
        [cleaned_text]
    )

    padded_sequence = pad_sequences(
        tokenized_text,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
    )

    probabilities = model.predict(
        padded_sequence,
        verbose=0,
    )[0]

    top_emotion_index = int(
        np.argmax(probabilities)
    )

    all_probabilities = {
        label: float(prob)
        for prob, label in zip(
            probabilities,
            EMOTION_LABELS,
        )
    }

    return {
        "text": text,
        "predicted_emotion":
            EMOTION_LABELS[top_emotion_index],
        "confidence":
            float(probabilities[top_emotion_index]),
        "all_probabilities":
            all_probabilities,
    }


def render_header():
    html_block = (
        '<div class="topbar">'
        '<div class="brand">'
        '<span class="brand-mark"></span>'
        '<span class="brand-name">Moodline</span>'
        '</div>'
        '<p class="brand-tag">'
        'an instrument that reads the mood inside a sentence'
        '</p>'
        '</div>'
    )

    st.markdown(
        html_block,
        unsafe_allow_html=True,
    )


def render_orb(emotion=None):
    emoji = EMOTION_EMOJIS.get(
        emotion,
        "✎",
    )

    html_block = (
        '<div class="orb-stage">'
        '<div class="orb-glow"></div>'
        '<div class="orb">'
        f'<span class="orb-emoji">{emoji}</span>'
        '</div>'
        '</div>'
    )

    st.markdown(
        html_block,
        unsafe_allow_html=True,
    )


def render_result(result):
    emotion = result["predicted_emotion"]

    emoji = EMOTION_EMOJIS.get(
        emotion,
        "🙂",
    )

    confidence = (
        result["confidence"] * 100
    )

    original_text = html.escape(
        result["text"]
    )

    entries = sorted(
        result["all_probabilities"].items(),
        key=lambda item: item[1],
        reverse=True,
    )

    bars_html = ""

    for label, value in entries:
        pct = value * 100

        bar_emoji = EMOTION_EMOJIS.get(
            label,
            "",
        )

        bars_html += (
            '<div class="bar-row">'
            f'<span class="bar-label">'
            f'{bar_emoji} {html.escape(label)}'
            '</span>'
            '<span class="bar-track">'
            f'<span class="bar-fill bar-{label}" '
            f'style="width:{pct:.1f}%"></span>'
            '</span>'
            f'<span class="bar-pct">'
            f'{pct:.1f}%'
            '</span>'
            '</div>'
        )

    result_html = (
        '<div class="result-grid">'

        '<div class="result-card">'
        '<span class="eyebrow">DETECTED</span>'

        '<div class="reveal-main">'
        f'<span class="reveal-emoji">{emoji}</span>'
        f'<h1 class="emotion-word">'
        f'{html.escape(emotion.capitalize())}'
        '</h1>'
        '</div>'

        f'<span class="confidence-mono">'
        f'{confidence:.1f}% confidence'
        '</span>'

        f'<blockquote class="echoed-text">'
        f'“{original_text}”'
        '</blockquote>'

        '</div>'

        '<div class="result-card">'
        '<span class="eyebrow">BREAKDOWN</span>'
        '<div class="bars">'
        f'{bars_html}'
        '</div>'
        '</div>'

        '</div>'
    )

    st.markdown(
        result_html,
        unsafe_allow_html=True,
    )


def render_footer(model_ready):
    if model_ready:
        status = "model ready — say something"
    else:
        status = "model unavailable"

    html_block = (
        '<div class="footbar">'
        f'<span>{status}</span>'
        '<span>BiGRU emotion classifier</span>'
        '</div>'
    )

    st.markdown(
        html_block,
        unsafe_allow_html=True,
    )


inject_styles()

if "result" not in st.session_state:
    st.session_state.result = None


current_emotion = None

if st.session_state.result:
    current_emotion = (
        st.session_state.result[
            "predicted_emotion"
        ]
    )


set_accent(current_emotion)


model_ready = False
load_error = None

try:
    model, tokenizer = load_assets()
    model_ready = True

except Exception as exc:
    model = None
    tokenizer = None
    load_error = str(exc)


render_header()


with st.container(
    key="console_panel"
):

    orb_col, input_col = st.columns(
        [0.31, 0.69],
        gap="large",
    )

    with orb_col:
        render_orb(
            current_emotion
        )

    with input_col:

        st.markdown(
            '<span class="eyebrow">'
            'INPUT'
            '</span>',
            unsafe_allow_html=True,
        )

        text = st.text_area(
            "Input",
            value="",
            max_chars=MAX_TEXT_LENGTH,
            placeholder=(
                "Write a sentence and let it "
                "speak for itself\n"
                "— e.g. “I can't believe we "
                "actually pulled this off.”"
            ),
            label_visibility="collapsed",
            key="mood_text",
        )

        count_col, button_col = (
            st.columns([1, 1])
        )

        with count_col:

            st.markdown(
                (
                    '<div class="st-key-char_count">'
                    f'{len(text)} / '
                    f'{MAX_TEXT_LENGTH}'
                    '</div>'
                ),
                unsafe_allow_html=True,
            )

        with button_col:

            analyze = st.button(
                "Read the mood →",
                disabled=not model_ready,
                key="analyze_button",
            )

        if load_error:

            st.markdown(
                (
                    '<p class="streamlit-error">'
                    'The model could not be loaded. '
                    'Check that '
                    '<code>'
                    'Artifacts/BiGRU_Model.keras'
                    '</code> and '
                    '<code>'
                    'Artifacts/tokenizer.pkl'
                    '</code> exist.'
                    '</p>'
                ),
                unsafe_allow_html=True,
            )


if analyze:

    if not text.strip():

        st.markdown(
            (
                '<p class="streamlit-error">'
                'Please enter a sentence.'
                '</p>'
            ),
            unsafe_allow_html=True,
        )

    elif len(text) > MAX_TEXT_LENGTH:

        st.markdown(
            (
                '<p class="streamlit-error">'
                f'Text must be at most '
                f'{MAX_TEXT_LENGTH} characters.'
                '</p>'
            ),
            unsafe_allow_html=True,
        )

    else:

        with st.spinner(
            "Reading…"
        ):

            st.session_state.result = (
                predict_emotion(
                    text,
                    model,
                    tokenizer,
                )
            )

        st.rerun()


if st.session_state.result:

    render_result(
        st.session_state.result
    )


render_footer(
    model_ready
)