from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
import os

# Paso 1: Cargar el libro PDF como documentos
book_path = "code/3_RAG/3. Manual de Cardiología y Cirugía Cardiovascular.pdf"
loader = PyPDFLoader(book_path)
documentos = loader.load()

# Paso 2: Dividir el texto en fragmentos pequeños (chunking)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documentos)

# Paso 3: Convertir cada fragmento en un vector usando embeddings
embedding = OllamaEmbeddings(model="nomic-embed-text")  # Otra opción es "all-minilm" pero necesita API
vectorstore = FAISS.from_documents(docs, embedding)

# Paso 4: Crear un sistema de recuperación
retriever = vectorstore.as_retriever()

# Paso 5: Crear el modelo RAG (Retriever + LLM)
llm = Ollama(model="llama3")  # Usa cualquier modelo LLM que tengas cargado en Ollama
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# Paso 6: Hacer una pregunta y obtener respuesta con contexto
pregunta = '''¿Cuál de las siguientes enfermedades alcanza mayor letalidad?: 
        1. Ictus.
        2. COVID-19.
        3. Infarto agudo de miocardio.
        4. Encefalopatía espongiforme bovina.'''
output = qa_chain.invoke({"query": pregunta})

print("\n🧠 Respuesta:")
print(output["result"])

print("\n📚 Documentos usados:")
for doc in output["source_documents"]:
    print("-", doc.metadata.get("source", "sin nombre"))

