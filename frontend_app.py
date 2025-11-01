import streamlit as st
import requests 

#Page Configuration 
st.set_page_config(
    page_title="My RAG Chatbot",
    page_icon="🤖"
)
st.title("🤖 My Simple RAG Chatbot")
st.caption("Chat with your knowledge base. Upload documents to add to the knowledge base!")

FASTAPI_URL = "http://127.0.0.1:8000"
FASTAPI_CHAT_URL = f"{FASTAPI_URL}/chat"
FASTAPI_UPLOAD_URL = f"{FASTAPI_URL}/upload-document"

# File Upload Section
with st.sidebar:
    st.header("📄 Upload Documents")
    st.caption("Add PDF or TXT files to your knowledge base")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt"],
        help="Upload a PDF or TXT file to add to the knowledge base"
    )
    
    if uploaded_file is not None:
        if st.button("Upload and Process", type="primary"):
            try:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(FASTAPI_UPLOAD_URL, files=files)
                    response.raise_for_status()
                    
                    result = response.json()
                    st.success(f"✅ {result['message']}")
                    st.info(f"📊 Added {result['chunks_added']} chunks to the knowledge base")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Error uploading file: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_detail = e.response.json()
                        st.error(f"Details: {error_detail.get('detail', 'Unknown error')}")
                    except:
                        st.error(f"Status: {e.response.status_code}")

#Chat History Management 
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input and Response
# st.chat_input waits for the user to type something
if prompt := st.chat_input("How many vacation days do we get?"):
    
    # 1. Add user's message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Get bot's response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Send the question to the FastAPI backend
                response = requests.post(FASTAPI_CHAT_URL, json={"question": prompt})
                response.raise_for_status() # Raise an error for bad responses (4xx or 5xx)
                
                # Get the answer from the JSON response
                data = response.json()
                bot_answer = data.get("answer", "Sorry, I couldn't find an answer.")
                
            except requests.exceptions.RequestException as e:
                bot_answer = f"Error: Could not connect to the backend. {e}"

        # Display the bot's answer
        st.markdown(bot_answer)
    
    # 3. Add bot's message to history
    st.session_state.messages.append({"role": "assistant", "content": bot_answer})