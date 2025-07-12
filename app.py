# 1. IMPORTAR LIBRERÍAS
import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# --- CONFIGURACIÓN DE LA PÁGINA DE STREAMLIT ---
st.set_page_config(page_title="Asistente Legal ESAP", page_icon="⚖️")

st.title("⚖️ Asistente Legal para la ESAP")
st.markdown("Esta IA responde preguntas basándose **únicamente** en los documentos proporcionados. Escribe tu consulta abajo.")

# --- FUNCIÓN PRINCIPAL PARA CARGAR Y PROCESAR DOCUMENTOS ---
@st.cache_resource
def cargar_base_de_conocimiento():
    # Cargar documentos desde la carpeta 'documentos'
    loader_pdf = DirectoryLoader('documentos/', glob="./*.pdf", loader_cls=PyPDFLoader, show_progress=True)
    loader_docx = DirectoryLoader('documentos/', glob="./*.docx", loader_cls=Docx2txtLoader, show_progress=True)

    documentos = loader_pdf.load() + loader_docx.load()

    if not documentos:
        st.error("Error: No se encontraron archivos en la carpeta 'documentos'. Asegúrate de subirlos a GitHub.")
        st.stop()

    # Dividir los documentos en trozos más pequeños
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    textos = text_splitter.split_documents(documentos)

    # Usar los 'embeddings' de Google para convertir texto a vectores
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=st.secrets["GOOGLE_API_KEY"])

    # Crear la base de datos vectorial en memoria con Chroma
    vectordb = Chroma.from_documents(documents=textos, embedding=embeddings)
    return vectordb.as_retriever(search_kwargs={"k": 3}) # k=3 significa que buscará los 3 trozos más relevantes

# --- LÓGICA PRINCIPAL DE LA APLICACIÓN ---
try:
    # Validar que la API key está configurada en los secretos de Streamlit
    if "GOOGLE_API_KEY" not in st.secrets:
        st.error("ERROR: Debes configurar tu GOOGLE_API_KEY en los secretos de Streamlit.")
        st.stop()

    # Cargar la base de conocimiento y el recuperador
    retriever = cargar_base_de_conocimiento()

    # Configurar el modelo de lenguaje de Google (Gemini)
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=st.secrets["GOOGLE_API_KEY"],
                                 temperature=0.2, convert_system_message_to_human=True)

    # Crear la cadena que une el buscador (retriever) y el cerebro (llm)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    # Interfaz de usuario para la consulta
    prompt = st.text_input("Introduce tu consulta jurídica aquí:", placeholder="Ej: ¿Cuál es el término para responder un derecho de petición?")

    if st.button("Generar Respuesta"):
        if prompt:
            with st.spinner("Buscando en los documentos y generando respuesta... 🧠"):
                respuesta_llm = qa_chain(prompt)
                respuesta = respuesta_llm['result']
                fuentes = respuesta_llm['source_documents']

                st.subheader("✅ Respuesta Generada:")
                st.write(respuesta)

                st.subheader("📂 Fuentes Consultadas:")
                for fuente in fuentes:
                    st.info(f"**Archivo:** {os.path.basename(fuente.metadata.get('source', 'N/A'))}")
        else:
            st.warning("Por favor, introduce una pregunta.")
except Exception as e:
    st.error(f"Ha ocurrido un error inesperado: {e}")