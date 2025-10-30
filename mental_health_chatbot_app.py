# -------------------------------------------------------
# AI Mental Health Support Chatbot - Streamlit Version
# Author: Vinjamuri Sai Ganesh
# -------------------------------------------------------

import streamlit as st
from textblob import TextBlob
import random

# -----------------------------
# Function: Analyze Emotion
# -----------------------------
def analyze_emotion(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity  # -1 to +1
    if polarity > 0.2:
        return "positive", "😊"
    elif polarity < -0.2:
        return "negative", "😔"
    else:
        return "neutral", "😐"

# -----------------------------
# Function: Generate Response
# -----------------------------
def chatbot_response(user_input):
    positive_responses = [
        "That's great to hear! Keep spreading positivity.",
        "I’m so glad you’re feeling good today! 😊",
        "Keep up that wonderful energy!"
    ]

    neutral_responses = [
        "Hmm, I see. Do you want to talk more about it?",
        "Okay, tell me what’s on your mind.",
        "I’m here for you — whatever you’d like to share."
    ]

    negative_responses = [
        "I’m sorry you’re feeling that way 😔. You’re not alone.",
        "Take a deep breath. You’re doing your best and that’s enough. 💙",
        "It’s okay to have tough days. Would you like some tips to calm down?"
    ]

    emotion, emoji = analyze_emotion(user_input)

    if emotion == "positive":
        reply = random.choice(positive_responses)
    elif emotion == "negative":
        reply = random.choice(negative_responses)
    else:
        reply = random.choice(neutral_responses)

    return emotion, emoji, reply

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(page_title="AI Mental Health Chatbot", page_icon="💬", layout="centered")

# -----------------------------
# Chatbot Title
# -----------------------------
st.title("🤖 AI Mental Health Support Chatbot")
st.write("Welcome! I’m here to listen and support you. 💙")
st.write("Type your thoughts below — I’ll respond with care and empathy.")

# -----------------------------
# Chat History Initialization
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# User Input
# -----------------------------
user_input = st.text_input("You:", "")

if user_input:
    emotion, emoji, bot_reply = chatbot_response(user_input)

    # Save chat history
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", f"{emoji} {bot_reply}"))

# -----------------------------
# Display Chat History
# -----------------------------
st.markdown("### 🗨️ Chat History")
for sender, message in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**🧑‍💬 You:** {message}")
    else:
        st.markdown(f"**🤖 Bot:** {message}")

# -----------------------------
# Footer
# -----------------------------
st.write("---")
st.caption("Developed by Vinjamuri Sai Ganesh | AI for Social Good 🌍")
