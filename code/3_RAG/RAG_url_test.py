from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# Paso 1: Cargar la web
url = "https://es.wikipedia.org/wiki/Infarto_agudo_de_miocardio"
loader = WebBaseLoader(url)
documentos = loader.load()

# Paso 2: Dividir texto
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documentos)

# Paso 3: Vectorizar y almacenar
embedding = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.from_documents(docs, embedding)
retriever = vectorstore.as_retriever()

# Paso 4: Crear QA chain
llm = Ollama(model="llama3")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# Paso 5: Preguntar
pregunta = "¿Qué causa un infarto agudo de miocardio?"
respuesta = qa_chain.invoke({"query": pregunta})

print("\n🧠 Respuesta:")
print(respuesta["result"])
