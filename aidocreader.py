import streamlit as st
import faiss
import numpy as np
import PyPDF2
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

llm=OllamaLLM(model="mistral")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
index = faiss.IndexFlatL2(384)

vector_store={}

def extract_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text =""
    for page in pdf_reader.pages:
        text += page.extract_text () + "\n"
    return text

def generate_summary(text):
    global summary_text
    st.write("Generating AI Summary..")
    summary_text=llm.invoke(f"Summarize the following document:\n\n{text[:3000]}")
    return summary_text

def store_in_faiss(text,filename):
    global index,vector_store
    st.write(f"Storing document '{filename}' in FAISS...")

    #split text into chunks
    splitter=CharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    texts=splitter.split_text(text)

    vectors=embeddings.embed_documents(texts)
    vectors=np.array(vectors,dtype=np.float32)

    index. add (vectors)
    vector_store[len(vector_store)] = (filename, texts)
    return "Document stored successfully!"

def retrieve_and_answer(query):
    global index, vector_store
    # Convert query into embedding
    query_vector=np.array(embeddings.embed_query(query),dtype=np.float32).reshape(1,-1)
    # Search FAISS
    D, I = index.search(query_vector,k=2 ) # Retrieve top 2 similar chunks
    context=""
    for idx in I[0]:
        if idx in vector_store:
            context+=" ".join(vector_store[idx][1])+"\n\n"
    
    if not context:
        return "No relevant data found."
    #ai to generate answer
    return llm.invoke(f"Based on the following context, answer the question:\n\n {context}\n\nQuestion: {query}\n Answer:")

def download_summary():
    if summary_text:
        st.download_button(
            label="Download Summary",
            data=summary_text,
            file_name="AI_summary.txt",
            mime="text/plain"
        )
#streamlit web UI
st.title("AI Document Reader & QA Bot")
st.write("Upload a PDF and ask questions based on its content!")
# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF Document", type=["pdf"])

if uploaded_file:
    # Extract text
    text = extract_from_pdf(uploaded_file)

    # Store in FAISS
    store_message = store_in_faiss(text, uploaded_file.name)
    st.write(store_message)

    # Generate AI Summary
    summary = generate_summary(text)
    st.subheader("🧠 AI-Generated Summary")
    st.write(summary)

    # Enable File Download for Summary
    download_summary()

if uploaded_file:
    text = extract_from_pdf(uploaded_file)
    store_message = store_in_faiss(text, uploaded_file.name)
    st.write(store_message)

    # User input for Q&A
    query = st.text_input("❓ Ask a question based on the uploaded document:")
    if query:
        answer = retrieve_and_answer(query)
        st.subheader("🤖 AI Answer:")
        st.write(answer)
