import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import fitz
import warnings
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

warnings.filterwarnings("ignore")

PDF_FOLDER = "pdf_files"
INDEX_DIR = "faiss_index"
PROCESSED_FILE_LOG = "processed_files.txt"

os.makedirs(PDF_FOLDER, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def split_text(text, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_processed_files():
    if not os.path.exists(PROCESSED_FILE_LOG):
        print("No new files found")
        return set()
    with open(PROCESSED_FILE_LOG, "r") as f:
        print("New files found")
        return set(line.strip() for line in f.readlines())
def save_processed_file(filename):
    with open(PROCESSED_FILE_LOG, "a") as f:
        f.write(filename + "\n")
    print("Saved new PDF files")

def update_faiss_index(chunks, embeddings):
    if os.path.exists(INDEX_DIR):
        vector_store = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        vector_store.add_texts(chunks)
    else:
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local(INDEX_DIR)
    print("Updated FAISS index")
    return vector_store

def setup_qa_system(vector_store, k=3):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    llm = OllamaLLM(model="mistral")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa

@st.cache_resource(show_spinner=False)
def initialize_qa():
    embeddings = get_embeddings()
    processed_files = load_processed_files()

    all_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    new_files = [f for f in all_files if f not in processed_files]

    for pdf_name in new_files:
        pdf_path = os.path.join(PDF_FOLDER, pdf_name)
        text = extract_text_from_pdf(pdf_path)
        chunks = split_text(text)
        update_faiss_index(chunks, embeddings)
        save_processed_file(pdf_name)

    vector_store = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    qa = setup_qa_system(vector_store)
    return qa

st.set_page_config(page_title="PDF QA Chat", layout="centered")
st.markdown("<h2 style='text-align:center;'>📖 Ask Questions About Your PDFs</h2>", unsafe_allow_html=True)

qa = initialize_qa()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask a question about the PDFs...")

if user_input:
    with st.spinner("Thinking..."):
        answer = qa.run(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

