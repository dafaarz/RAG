import os
from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import default_storage
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

CHROMA_PATH = "media/chroma"
DATA_PATH = "media/data"

embedding_model = OllamaEmbeddings(model="nomic-embed-text")
db_vector = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)
llm_model = OllamaLLM(model="mistral")

prompt_template = """
Answer the question based only on the following context:

{context}

---

Answer the question based only on the above context: {question}
"""

def index(request):
    """Renders the chatbot HTML page."""
    return render(request, "index.html") 

def upload_pdf(request):
    #Handle PDF Upload dan Processingnya
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method."}, status=400)

    if "pdf_file" not in request.FILES:
        return JsonResponse({"error": "No file uploaded."}, status=400)

    # Menyimpan PDF dalam suatu directory Django dan pdf_files adalah key untuk aksesnya
    pdf_file = request.FILES["pdf_file"]
    

    # Memastikan Directory data ada, Bikin path directory untuk save data, dan save data (PDF)
    os.makedirs(DATA_PATH, exist_ok=True)
    file_path = os.path.join(DATA_PATH, pdf_file.name)
    default_storage.save(file_path, pdf_file)

    # Extract text dari PDF dan Split data text (untuk code ini split by Char bukan Text/String)
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    if not documents:
        return JsonResponse({"error": "No text extracted from the PDF."}, status=400)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, 
        chunk_overlap=110, 
        length_function=len, 
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)

    # Save data embedding ChromaDB
    if os.path.exists(CHROMA_PATH):
        db_vector.add_documents(chunks)
    else:
        db = Chroma.from_documents(chunks, embedding_model, persist_directory=CHROMA_PATH)
        db.persist()

    return JsonResponse({"message": f"Added {len(chunks)} chunks to ChromaDB."})


def ask_question(request):
    """Handles chatbot queries."""
    if request.method == "GET" and "question" in request.GET:
        question = request.GET["question"]
        result = db_vector.similarity_search_with_score(question, k=5)
        
        if not result or result[0][1] < 0.5:
            return JsonResponse({"answer": "No relevant documents found."})

        docs = [doc for doc, score in result]
        context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm_model
        answer = chain.invoke({"question": question, "context": context_text})

        return JsonResponse({"answer": str(answer)})

    return JsonResponse({"error": "Invalid request. HERE IN ASK QUESTION"}, status=400)
