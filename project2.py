
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os

# ✅ DO NOT call st.set_page_config here (since main file handles it)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# Embedding model
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# Extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Chunking text
def get_text_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return splitter.split_text(text)

# Create vector store
def get_vector_store(text_chunks):
    return FAISS.from_texts(text_chunks, embedding_model)

# Use Google Gemini 
def get_conversation_chain(vector_store):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7,
        convert_system_message_to_human=True,
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vector_store.as_retriever(), memory=memory
    )

# Handle Q&A logic
def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

# Main app logic
def main():
    st.title("🤖 Chat with Multiple PDFs (Gemini-Powered)")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.header("📄 Upload Documents")
        pdf_docs = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

        if st.button("Process PDFs"):
            with st.spinner("Processing..."):
                text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(text)
                vector_store = get_vector_store(chunks)
                st.session_state.conversation = get_conversation_chain(vector_store)
                st.session_state.chat_history = []
                st.success("Documents processed successfully!")

    # Show chat history
    if st.session_state.conversation:
        for i, message in enumerate(st.session_state.chat_history):
            role = "user" if i % 2 == 0 else "assistant"
            with st.chat_message(role):
                st.markdown(message.content)

        # Input box
        user_question = st.chat_input("Ask a question about your documents")
        if user_question:
            with st.chat_message("user"):
                st.markdown(user_question)
            handle_user_input(user_question)
            with st.chat_message("assistant"):
                last_reply = st.session_state.chat_history[-1].content
                st.markdown(last_reply)

if __name__ == "__main__":
    main()
