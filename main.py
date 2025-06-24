from backend.core import run_llm
import streamlit as st

st.set_page_config(page_title="📄 Documentation Assistant", page_icon="🤖", layout="centered")

# --- Stylish Header ---
st.markdown("""
    <h1 style='text-align: center; color: #4F8BF9;'>📄 Documentation Helper Bot</h1>
    <p style='text-align: center; font-size: 16px; color: gray;'>
        Ask questions from your technical documentation — get clear, smart answers fast!
    </p>
""", unsafe_allow_html=True)

# --- Reset Button ---
if st.button("🧹 Reset Chat"):
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_answers_history"] = []
    st.session_state["chat_history"] = []
    st.experimental_rerun()

# --- Input ---
prompt = st.text_input("💬 Enter your question", placeholder="e.g. What is a LangChain Chain?")

# --- Initialize session state for memory ---
if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  # stores [("human", msg), ("ai", reply)]

# --- Handle Prompt Submission ---
if prompt:
    with st.spinner("🧠 Generating response..."):
        generated_response = run_llm(
            query=prompt,
            chat_history=st.session_state["chat_history"]
        )

        # Update all histories
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(generated_response["result"])
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response["result"]))

# --- Display Chat History ---
st.markdown("---")
if st.session_state["chat_answers_history"]:
    for user_query, assistant_response in reversed(list(zip(
        st.session_state["user_prompt_history"],
        st.session_state["chat_answers_history"]
    ))):
        with st.chat_message("user"):
            st.markdown(f"**You:** {user_query}")
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

        st.markdown("---")
