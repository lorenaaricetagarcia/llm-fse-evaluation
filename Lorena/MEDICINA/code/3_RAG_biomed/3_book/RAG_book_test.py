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
pregunta = '''      "enunciado": "Un varón de 40 años consulta por disnea de esfuerzo  lentamente progresiva desde hace un año. No tiene  hábitos tóxicos ni antecedentes de interés. No ha  tenido dolor torácico. Presenta un soplo sistólico  rudo en foco aórtico, irradiado a ápex cardiaco y  carótidas. El ECG en ritmo sinusal muestra criterios  de hipertrofia ventricular izquierda y en la Rx de  tórax es evidente una raíz de aorta dilatada. Con  estos datos, ¿qué diagnóstico es el más probable?:",
      "opciones": [
        "Válvula aórtica bicúspide estenótica.",
        "Comunicación interventricular perimembranosa.",
        "Insuficiencia aórtica degenerativa.",
        "Miocardiopatía restrictiva."
      ],'''
output = qa_chain.invoke({"query": pregunta})

print("\n🧠 Respuesta:")
print(output["result"])
