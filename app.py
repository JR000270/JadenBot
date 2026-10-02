
import streamlit as st
from chat import get_response, bot_name

#run command:  streamlit run C:\Users\Jaden\OneDrive\Documents\GitHub\JadenBot/app.py
# Modern red and black theme
st.set_page_config(page_title="JadenBot", layout="centered")

#custom CSS for styling (base colors live in .streamlit/config.toml)
st.markdown("""
    <style>
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

st.title("Welcome to JadenBot!")



st.sidebar.button("New Chat", on_click=lambda: st.session_state.chat_history.clear())

# What the bot understands (mirrors the tags in intents.json)
with st.sidebar.expander("What can I ask?", expanded=True):
    st.markdown(
        """
        **Small talk**
        - Greetings & goodbyes: *"Hey!"*, *"See you later"*
        - How it's doing: *"How are you?"*
        - Thanks & compliments: *"You're awesome"*

        **About Jaden**
        - *"Who is Jaden?"*
        - *"What's your name?"*
        - *"Where are you located?"*

        **Interests**
        - Music & artists: *"What bands do you like?"*
        - Movies: *"What's your favorite movie?"*
        - Dislikes: *"What do you hate?"*

        **Fun & feelings**
        - Jokes: *"Tell me a joke!"*
        - Your mood: *"I feel sad"*, *"I'm feeling great"*
        """
    )
    st.caption("Short, simple questions work best. JadenBot matches your message to one of these topics.")

with st.sidebar.expander("About JadenBot", expanded=False):
    st.markdown(
        """
        JadenBot is a chatbot powered by a neural network model. It can understand and respond to user inputs based on predefined intents and patterns. 
        The bot is designed to engage in small talk, answer questions about itself, discuss interests, and provide some fun interactions.
        It doesn't hold conversation context between messages, so each message is treated independently.
        """
    )


# Fixed-height, scrollable box that holds the conversation
chat_box = st.container(height=500, border=True)

# Display the chat history
with chat_box:
    for entry in st.session_state.chat_history:
        with st.chat_message(entry["sender"], avatar=entry["avatar"]):
            st.markdown(entry["message"])

# Message input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Append user message
    st.session_state.chat_history.append({"sender": "user", "message": user_input, "avatar":"assets/jadenbot_user_avatar.png"})
    with chat_box:
        with st.chat_message("user", avatar="assets/jadenbot_user_avatar.png"):
            st.markdown(user_input)

    # Get response from the bot
    try:
        bot_reply = get_response(user_input)
    except Exception as e:
        print(f"Error: {e}")
        bot_reply = "Sorry, I encountered an error."

    # Append bot response
    st.session_state.chat_history.append({"sender": "assistant", "message": bot_reply, "avatar": "assets/jadenbot_avatar_2.png"})
    with chat_box:
        with st.chat_message("assistant", avatar="assets/jadenbot_avatar_2.png"):
            st.markdown(bot_reply)
