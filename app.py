
import streamlit as st
from chat import get_response, bot_name

#run command:  streamlit run c:/Users/Jaden/OneDrive/Desktop/coding/pytorch_chatbot/app.py
# Modern red and black theme
st.set_page_config(page_title="JadenBot", layout="centered")

# Apply custom CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #0F0F0F;
        color: #F5F5F5;
    }
    .stTextInput > div > div > input {
        background-color: #1C1C1C;
        color: #F5F5F5;
        border: 1px solid #8B0000;
        border-radius: 8px;
    }
    .stButton > button {
        background-color: #8B0000;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 8px 16px;
    }
    .stChatMessage {
        background-color: #1C1C1C;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🤖 Welcome to JadenBot")

# Display the chat history
for entry in st.session_state.chat_history:
    with st.chat_message(entry["sender"]):
        st.markdown(entry["message"])

# Message input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Append user message
    st.session_state.chat_history.append({"sender": "user", "message": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get response from the bot
    try:
        bot_reply = get_response(user_input)
    except Exception as e:
        print(f"Error: {e}")
        bot_reply = "Sorry, I encountered an error."

    # Append bot response
    st.session_state.chat_history.append({"sender": "assistant", "message": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
