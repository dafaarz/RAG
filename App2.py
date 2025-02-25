import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import os
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "chroma"
DATA_PATH = "data"

prompt_template = """
Answer the question based only on the following context:

{context}

---

Answer the question based only on the above context: {question}
"""

embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
llm_model = ChatOpenAI(model="gpt-4-mini", temperature=0) 
db_vector = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)

def upload_pdf(file):
    os.makedirs(DATA_PATH, exist_ok=True)
    with open(os.path.join(DATA_PATH, file.name), "wb") as f:
        f.write(file.getbuffer())

def load_pdf(directory):
    loader = PyPDFDirectoryLoader(directory)
    docs = loader.load()
    return docs

def text_splits(docs: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True
    )
    return text_splitter.split_documents(docs)

def save_to_chroma(chunks: list[Document]):
    if not chunks:
        st.error("No text found in PDF!")
        return

    embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
    
    if os.path.exists(CHROMA_PATH):
        DB = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)
        DB.add_documents(chunks)
    else:
        DB = Chroma.from_documents(chunks, embedding, persist_directory=CHROMA_PATH)

    DB.persist()
    st.success(f"Added {len(chunks)} chunks to ChromaDB.")

def retrieve_results(text_query):
    result = db_vector.similarity_search_with_score(text_query, k=5)
    if not result or result[0][1] < 0.5:
        st.warning("No relevant documents found!")
        return []
    
    return [doc for doc, score in result]

def answer_results(question, doc):
    if not doc:
        return "I don't know."

    context_text = context_text = "\n\n---\n\n".join([doc.page_content for doc in doc])
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = prompt | llm_model

    return chain.invoke({"question": question, "context": context_text})

st.title("PDF RAG Chatbot")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=False
)

if uploaded_file:
    upload_pdf(uploaded_file)
    documents = load_pdf(DATA_PATH)
    
    if not documents:
        st.error("Failed to extract text from PDF.")
    else:
        chunked_documents = text_splits(documents)
        save_to_chroma(chunked_documents)

question = st.chat_input("Ask a question:")

if question:
    st.chat_message("user").write(question)
    related_documents = retrieve_results(question)
    answer = answer_results(question, related_documents)
    st.chat_message("assistant").write(answer)
