# -------------------------------------------
# AI Mental Health Support Chatbot (v2)
# With Emotion Detection
# Author: Vinjamuri Sai Ganesh
# -------------------------------------------

from textblob import TextBlob
import random

# Supportive responses for different emotions
positive_responses = [
    "I'm so glad to hear that! 😊 Keep spreading positive energy.",
    "That's wonderful! What made your day so good?",
    "Yay! Stay confident and happy!"
]

neutral_responses = [
    "Hmm, I understand. Do you want to talk more about it?",
    "Okay, tell me more if you’d like.",
    "I’m here to listen — how are you feeling overall?"
]

negative_responses = [
    "I'm really sorry you feel that way 😔. Remember, it’s okay to take a break.",
    "You’re doing your best, and that’s enough. 🌼",
    "Tough times don’t last forever. Would you like some relaxation tips?"
]

# Function to detect emotion
def analyze_emotion(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity  # Range: -1 (negative) to +1 (positive)

    if polarity > 0.2:
        return "positive", "😊"
    elif polarity < -0.2:
        return "negative", "😔"
    else:
        return "neutral", "😐"

# Function to generate chatbot response
def chatbot_response(user_input):
    emotion, emoji = analyze_emotion(user_input)

    if emotion == "positive":
        reply = random.choice(positive_responses)
    elif emotion == "negative":
        reply = random.choice(negative_responses)
    else:
        reply = random.choice(neutral_responses)

    return f"{emoji} ({emotion.capitalize()} mood)\nBot: {reply}"

# Chatbot main interaction loop
print("🤖 AI Mental Health Support Chatbot (v2)")
print("Hello! I'm here to listen and support you. Type 'bye' to exit anytime.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'bye':
        print("Bot: Take care! Remember, every emotion is valid. 💙")
        break

    response = chatbot_response(user_input)
    print(response, "\n")
