import os
from django.shortcuts import render,redirect
from django.http import JsonResponse
from django.core.files.storage import default_storage
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required

load_dotenv()

CHROMA_PATH = "media/chroma"
DATA_PATH = "media/data"

embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
llm_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt_template = """
Answer the question based only on the following context:

{context}

---

Answer the question based only on the above context: {question}
"""

#@login_required
def knowledge_select(request):
    #Render page knowledge select
    return render(request,"select_knowledge.html")

#@login_required
def set_knowledge(request):
    #Menyimpan knowledge yang dipilih
    if request.method == "POST":
        selected_knowledge = request.POST.getlist("knowledge")
        request.session["knowledge"] = selected_knowledge
        return redirect("index")
    return redirect("select_knowledge.html")

#@login_required
def index(request):
   #render chatbot UI
    return render(request, "index.html") 

#@login_required
def upload_pdf(request):
    #Handle PDF Upload dan Processingnya
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method."}, status=400)

    if "pdf_file" not in request.FILES:
        return JsonResponse({"error": "No file uploaded."}, status=400)
    
    selected_knowledge = request.session.get("knowledge")

    #ChromaDB user
    user_id = "101-C"
    user_chroma_path = os.path.join(CHROMA_PATH, str(user_id)) 
    os.makedirs(user_chroma_path, exist_ok=True)    

    # Menyimpan PDF dalam suatu directory Django dan pdf_files adalah key untuk aksesnya
    pdf_files = request.FILES.getlist("pdf_file")
    max_pdf_size = 5 * 1024 * 1024
    for file in pdf_files:
        if file.size > max_pdf_size:
            return JsonResponse({"error": f"File {file.name} is too large. Max 5MB per file."}, status=400)
    

    # Memastikan Directory data ada
    os.makedirs(DATA_PATH, exist_ok=True)

    user_pdf_paths = [] #untuk nampung file user yang akan di delete setelah chunking

    #Bikin path directory untuk save data, dan save data (PDF)
    for pdf_file in pdf_files: #Loop setiap PDF yang ada
        unique_filename = f"user_{user_id}_{pdf_file.name}"
        file_path = os.path.join(DATA_PATH, str(unique_filename))
        user_pdf_paths.append(file_path)
        default_storage.save(file_path,pdf_file)

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

    pdfs_chunks = [] #store semua chunk untuk implementasi multiple PDFs
    pdfs_chunks.extend(chunks) #fungsi sama seperti .append tetapi dapat add multiple data

    # Save data embedding ChromaDB
    user_db_vector = Chroma(persist_directory=user_chroma_path, embedding_function=embedding_model)

    if os.path.exists(user_chroma_path):
        user_db_vector.add_documents(pdfs_chunks)
    else:
        db = Chroma.from_documents(pdfs_chunks, embedding_model, persist_directory=user_chroma_path)
        db.persist()

    if "office_rules" in selected_knowledge:
        office_db_vector = Chroma(persist_directory=os.path.join(CHROMA_PATH,"office_rules"),embedding_function=embedding_model)
        office_db_vector.add_documents(pdfs_chunks)

    if "worker_details" in selected_knowledge:
        worker_db_vector = Chroma(persist_directory=os.path.join(CHROMA_PATH,"worker_details"),embedding_function=embedding_model)
        worker_db_vector.add_documents(pdfs_chunks)

    # Delete PDF setelah di proses
    for pdf_path in user_pdf_paths:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

    return JsonResponse({"message": f"Added {len(pdfs_chunks)} chunks from {len(pdf_files)} PDFs to Users ChromaDB."})

#@login_required
def ask_question(request):
    #Ini handle chatbot querry / question
    if request.method == "GET" and "question" in request.GET:
        
        question = request.GET["question"]
        selected_knowledge = request.session.get("knowledge")
        print(selected_knowledge)
        
        user_id = "101-C" #request.user.id DIHARDCODE DULU UNTUK TESTING
        user_chroma_path = os.path.join(CHROMA_PATH, str(user_id)) 
        user_db_vector = Chroma(persist_directory=user_chroma_path, embedding_function=embedding_model)
        
        #Ambil semua result knowledge
        all_results = []
        
        if "user_data" in selected_knowledge:
            all_results.extend(user_db_vector.similarity_search_with_score(question,k=5))
            

        if "office_rules" in selected_knowledge:
            office_db_vector = Chroma(persist_directory=os.path.join(CHROMA_PATH,"office_rules"),embedding_function=embedding_model)
            all_results.extend(office_db_vector.similarity_search_with_score(question,k=5))

            
        if "worker_details" in selected_knowledge:
            worker_db_vector = Chroma(persist_directory=os.path.join(CHROMA_PATH,"worker_details"),embedding_function=embedding_model)
            all_results.extend(worker_db_vector.similarity_search_with_score(question,k=5))
            print("HERREEREERRE : ",all_results)

        # Sort data result berdasarkan similarity score DESC
        all_results.sort(key=lambda x: x[1], reverse=True)

        if not all_results or all_results[0][1] < 0.3:
            return JsonResponse({"answer" : "No relevant docs found"})

        # Mengambil hanya document dari result tanpa score dan mengambil text dari documents
        docs = [doc for doc, score in all_results[:5]]
        context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])
        
        
        # Prompt dan langchain
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm_model
        answer = chain.invoke({"question": question, "context": context_text})
        answer_human = answer.content

        return JsonResponse({"answer": str(answer_human)})

    return JsonResponse({"error": "Invalid request."}, status=400)
