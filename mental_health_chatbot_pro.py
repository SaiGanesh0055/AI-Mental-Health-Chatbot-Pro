"""
AI Mental Health Support Chatbot - Pro Version
Features:
 - Transformer-based emotion detection (multi-class)
 - AI replies using DialoGPT (context-aware)
 - Text-to-speech (pyttsx3 primary, gTTS fallback)
 - Emotion trend visualization (matplotlib)
 - Chat history & emotion log (Streamlit session_state)
Author: Vinjamuri Sai Ganesh
"""

import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline
import random
import matplotlib.pyplot as plt
import pandas as pd
import io
import os
import re
from gtts import gTTS
from textblob import TextBlob
import base64
import tempfile
import time

# Optional: pyttsx3 used if available and working on the platform
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except Exception:
    PYTTSX3_AVAILABLE = False

# -----------------------------
# Helper: Load models (with caching)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_emotion_model():
    # Multi-class emotion model (distilroberta / other reliable model)
    # This may download weights on first run.
    try:
        emotion_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=False)
    except Exception:
        # fallback to simpler sentiment pipeline if the above isn't available
        emotion_pipe = pipeline("sentiment-analysis")
    return emotion_pipe

@st.cache_resource(show_spinner=False)
def load_dialogpt_model(model_name="microsoft/DialoGPT-medium"):
    """
    Load DialoGPT model and tokenizer for conversational reply generation.
    NOTE: DialoGPT-medium is reasonably sized; use -small if you need less memory.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        gen_pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=-1)  # device=-1 -> CPU
        return gen_pipeline
    except Exception as e:
        st.warning("Could not load DialoGPT (may be large). Falling back to rule-based replies.")
        st.write("")  # visual spacing
        return None

# -----------------------------
# Helper: TTS (pyttsx3 primary, gTTS fallback)
# -----------------------------
def speak_text_pyttsx3(text):
    # pyttsx3 often works for local desktop. Not guaranteed on some servers.
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        return True
    except Exception:
        return False

def speak_text_gtts_bytes(text, lang='en'):
    # Use gTTS to generate mp3 bytes (requires internet)
    try:
        tts = gTTS(text=text, lang=lang)
        bio = io.BytesIO()
        tts.write_to_fp(bio)
        bio.seek(0)
        return bio.read()
    except Exception:
        return None

# -----------------------------
# Safety: check for emergency keywords
# -----------------------------
EMERGENCY_PATTERNS = [
    r"\bkill myself\b", r"\bkill me\b", r"\bsuicid", r"\bend my life\b",
    r"\bhurt myself\b", r"\bwant to die\b"
]

def check_emergency(text):
    txt = text.lower()
    for pat in EMERGENCY_PATTERNS:
        if re.search(pat, txt):
            return True
    return False

# -----------------------------
# Rule-based fallback replies
# -----------------------------
def rule_based_reply(emotion_label, user_text):
    positive_replies = [
        "That's really good to hear! Keep it going. 😊",
        "I’m happy for you — what helped you feel this way?",
        "Great! Keep nurturing that positivity."
    ]
    neutral_replies = [
        "I understand. Would you like to explain more?",
        "Thanks for telling me — I'm here to listen.",
        "It sounds like a calm moment. Want to talk more?"
    ]
    negative_replies = [
        "I'm sorry you're feeling this way. I'm here for you. 💙",
        "That sounds really tough. Would you like a few grounding exercises?",
        "I hear you. Would you like some tips to feel calmer?"
    ]
    if emotion_label in ["joy", "positive", "happy", "trust", "surprise"]:
        return random.choice(positive_replies)
    elif emotion_label in ["sadness", "anger", "fear", "negative", "disgust"]:
        return random.choice(negative_replies)
    else:
        return random.choice(neutral_replies)

# -----------------------------
# Emotion detection wrapper
# -----------------------------
def detect_emotion(emotion_pipe, text):
    # Prefer a multi-class emotion model if available
    try:
        out = emotion_pipe(text)
        # If that pipe returns label & score or list
        if isinstance(out, list):
            # sentiment pipeline fallback returns list of dicts sometimes
            label = out[0]["label"]
            score = out[0].get("score", None)
        elif isinstance(out, dict):
            label = out.get("label", "")
            score = out.get("score", None)
        else:
            # fallback: use TextBlob polarity
            polarity = TextBlob(text).sentiment.polarity
            if polarity > 0.2:
                return "positive", polarity
            elif polarity < -0.2:
                return "negative", polarity
            else:
                return "neutral", polarity
        return label.lower(), score
    except Exception:
        # fallback to simple polarity
        p = TextBlob(text).sentiment.polarity
        if p > 0.2:
            return "positive", p
        elif p < -0.2:
            return "negative", p
        else:
            return "neutral", p

# -----------------------------
# Response generation wrapper
# -----------------------------
def generate_response_genpipe(gen_pipeline, user_message, history):
    """
    If gen_pipeline (DialoGPT) is available, use it with simple conversation context.
    history: list of previous turns (strings). We'll form a dialogue string.
    """
    if gen_pipeline is None:
        return None

    # Create a short context by concatenating last few turns
    max_context = 6
    context = ""
    if history:
        last = history[-max_context:]
        context = " ".join([f"User: {turn}" if i % 2 == 0 else f"Bot: {turn}"
                            for i, turn in enumerate(last)])
    prompt = f"{context} User: {user_message} Bot:"
    try:
        # Generate text - use small max_length to prevent long responses
        gen_out = gen_pipeline(prompt, max_new_tokens=60, do_sample=True, top_k=50, top_p=0.9, num_return_sequences=1)
        text = gen_out[0]["generated_text"]
        # The generated text includes prompt + completion; extract after the last 'Bot:' if that exists
        if "Bot:" in text:
            # keep the portion after the last "Bot:"
            resp = text.split("Bot:")[-1].strip()
        else:
            # fallback: remove prompt
            resp = text.replace(prompt, "").strip()
        # Clean up awkward end tokens
        resp = resp.split("\n")[0].strip()
        return resp
    except Exception:
        return None

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Mental Health Chatbot - Pro", page_icon="💬", layout="centered")
st.title("🤖 AI Mental Health Support Chatbot — Pro")
st.markdown("An empathetic AI companion with emotion detection, voice, and trend visualization. Developed for YUVAi — AI for Social Good.")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # stores tuples: (sender, text)
if "emotion_log" not in st.session_state:
    st.session_state.emotion_log = []  # stores dicts: {"label":..., "score":..., "text":...}
if "gen_pipeline" not in st.session_state:
    st.session_state.gen_pipeline = None
if "emotion_pipe" not in st.session_state:
    st.session_state.emotion_pipe = None
if "tts_mode" not in st.session_state:
    st.session_state.tts_mode = "auto"  # auto -> try pyttsx3, then gTTS

# Load models on first use (shows spinner)
with st.spinner("Loading models (this may take some time the first run)…"):
    if st.session_state.emotion_pipe is None:
        st.session_state.emotion_pipe = load_emotion_model()
    if st.session_state.gen_pipeline is None:
        # Try DialoGPT-medium; you can change to -small if memory is a concern.
        st.session_state.gen_pipeline = load_dialogpt_model("microsoft/DialoGPT-medium")

st.sidebar.header("Settings")
st.sidebar.write("Model & TTS options (defaults recommended).")
st.sidebar.radio("TTS Mode", options=["auto", "pyttsx3", "gtts", "off"], index=0, key="tts_mode")

# Input area
st.subheader("Start a conversation")
user_text = st.text_input("You:", placeholder="Type how you're feeling or what's on your mind...")

# Buttons
col1, col2, col3 = st.columns([1,1,1])
with col1:
    send_btn = st.button("Send")
with col2:
    clear_btn = st.button("Clear Chat")
with col3:
    end_btn = st.button("End & Show Emotion Trend")

# Clear chat
if clear_btn:
    st.session_state.chat_history = []
    st.session_state.emotion_log = []
    st.success("Chat history cleared.")

# When user sends a message
if send_btn and user_text:
    st.session_state.chat_history.append(("You", user_text))
    # Emergency check
    if check_emergency(user_text):
        st.session_state.chat_history.append(("Bot", "I’m really sorry — it sounds like you may be in crisis. If you are in immediate danger or thinking of harming yourself, please contact your local emergency services right away. You can also reach out to your nearest crisis hotline. Would you like resources?"))
        st.session_state.emotion_log.append({"label": "emergency", "score": 1.0, "text": user_text})
    else:
        # detect emotion
        label, score = detect_emotion(st.session_state.emotion_pipe, user_text)
        st.session_state.emotion_log.append({"label": label, "score": score, "text": user_text})
        # generate response by AI model if available
        history_texts = [t for sender, t in st.session_state.chat_history if sender == "You" or sender == "Bot"]
        bot_reply = generate_response_genpipe(st.session_state.gen_pipeline, user_text, history_texts)
        if bot_reply is None:
            # fallback to rule-based
            bot_reply = rule_based_reply(label, user_text)

        # optionally append recommended tip when negative
        tip = None
        if label in ["sadness", "anger", "fear", "negative", "emotional", "depression"]:
            tip = random.choice([
                "Try 4-4-4 breathing: inhale 4s, hold 4s, exhale 4s.",
                "Take a 5-minute walk or stretch to reset your mind.",
                "Write one small thing you are grateful for right now."
            ])
            bot_reply = f"{bot_reply}\n\nTip: {tip}"

        st.session_state.chat_history.append(("Bot", bot_reply))

        # TTS (auto or selected)
        tts_mode = st.session_state.tts_mode
        spoken = False
        if tts_mode == "pyttsx3" or (tts_mode == "auto" and PYTTSX3_AVAILABLE):
            spoken = speak_text_pyttsx3(bot_reply) if PYTTSX3_AVAILABLE else False
        if not spoken and tts_mode in ("gtts", "auto"):
            mp3bytes = speak_text_gtts_bytes(bot_reply)
            if mp3bytes:
                st.audio(mp3bytes, format="audio/mp3")
                spoken = True
        # If tts_mode == off, do not speak

# Display Chat
st.markdown("### Chat")
for sender, msg in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**Bot:** {msg}")

# End & show emotion trend
if end_btn:
    st.success("Conversation ended. Here's your emotion trend during this chat.")
    # Build DataFrame from emotion_log
    if len(st.session_state.emotion_log) == 0:
        st.warning("No emotion data recorded.")
    else:
        df = pd.DataFrame(st.session_state.emotion_log)
        # Provide simple mapping to numeric score for plot
        def map_label_to_score(row):
            lbl = str(row["label"]).lower()
            if lbl in ["joy", "happy", "positive", "surprise", "trust"]:
                return 1
            if lbl in ["neutral", "neutrality"]:
                return 0
            if lbl in ["sadness", "anger", "fear", "negative", "sad", "emotional", "depression"]:
                return -1
            if lbl == "emergency":
                return -2
            try:
                # if numeric score available, use scaled
                sc = row.get("score", None)
                if sc is not None:
                    # map score (0..1) to -1..1 roughly based on label
                    return (sc * 2 - 1)
            except:
                pass
            return 0

        df["score_val"] = df.apply(map_label_to_score, axis=1)
        # plot
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(df.index + 1, df["score_val"], marker="o")
        ax.set_ylim(-2.2, 1.2)
        ax.set_xlabel("Turn (message index)")
        ax.set_ylabel("Emotion score (higher = more positive)")
        ax.set_title("Emotion Trend During Conversation")
        ax.grid(True)
        st.pyplot(fig)

        # Show summary counts
        counts = df["label"].value_counts().to_dict()
        st.write("Emotion counts:", counts)

        # Save emotion log as CSV for user download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download emotion log (CSV)", csv, file_name="emotion_log.csv", mime="text/csv")

        # Optionally: show recommended resources if a lot of negative emotions
        negative_count = sum(1 for lbl in df["label"] if str(lbl).lower() in ["sadness","anger","fear","negative","emotional","depression","emergency"])
        if negative_count > max(1, len(df)//3):
            st.info("We detected several negative-emotion messages. Here are some resources you might find helpful:")
            # Example resources (modify to local/national hotlines as appropriate)
            st.write("- If you are in immediate danger, contact emergency services.")
            st.write("- India (example) National Helpline: 9152987821 (Samaritans/other local)")
            st.write("- Reach out to a trusted person, counselor, or local mental health professional.")
            st.write("- Consider scheduling a session with a mental health professional.")

# Footer
st.write("---")
st.caption("Pro Version: Emotion detection + DialoGPT + TTS + Trend visualization. Developed by Vinjamuri Sai Ganesh")
