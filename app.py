import os
import warnings
import streamlit as st

from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

warnings.filterwarnings("ignore")

st.set_page_config(page_title="HelpBot 360", page_icon="🤖")
st.title("🤖 HelpBot 360 - Asistente TerraCampo")

# ---------------------------------------------------------
# 1. VALIDACIÓN DE API KEY
# ---------------------------------------------------------
api_key = None
if "GOOGLE_API_KEY" in st.secrets and str(st.secrets["GOOGLE_API_KEY"]).strip():
    api_key = str(st.secrets["GOOGLE_API_KEY"]).strip()

if not api_key:
    api_key = st.sidebar.text_input("Ingresa tu GOOGLE_API_KEY:", type="password")

if not api_key:
    st.warning("⚠️ Debes ingresar una GOOGLE_API_KEY válida para iniciar.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key

# ---------------------------------------------------------
# 2. CARGA DEL ÍNDICE VECTORIAL
# ---------------------------------------------------------
@st.cache_resource
def cargar_retriever():
    archivos_politicas = [
        "01_Vacaciones_Permisos_y_Licencias_TerraCampo.docx",
        "02_Uso_y_Mantenimiento_de_Equipos_TerraCampo.docx",
        "03_Beneficios_y_Nomina_TerraCampo.docx"
    ]

    docs_totales = []
    for archivo in archivos_politicas:
        if os.path.exists(archivo):
            loader = Docx2txtLoader(archivo)
            docs_totales.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    splits = text_splitter.split_documents(docs_totales)

    embeddings_gratuitos = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings_gratuitos)
    return vectorstore.as_retriever(search_kwargs={"k": 2})

retriever = cargar_retriever()

# ---------------------------------------------------------
# 3. CONSTRUCCIÓN DEL GRAFO LANGGRAPH
# ---------------------------------------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key,
    temperature=0
)

def nodo_helpbot(state: State):
    pregunta_usuario = state["messages"][-1].content
    docs_recuperados = retriever.invoke(pregunta_usuario)
    contexto = "\n\n".join([doc.page_content for doc in docs_recuperados])
    
    system_prompt = f"""
    Eres el asistente virtual de TerraCampo S.A.S.
    
    REGLAS DE RESPUESTA DIRECTA:
    1. Responde de forma concisa, clara y DIRECTA al punto.
    2. NUNCA saludes (no digas "Hola", "¡Hola! Claro que sí", etc.).
    3. NUNCA te presentes ni digas "Soy HelpBot 360".
    4. Basa tu respuesta ÚNICAMENTE en este contexto:
    <contexto>
    {contexto}
    </contexto>
    5. Si la información no está en el contexto, di brevemente que no la tienes y sugiere contactar a Recursos Humanos o Mantenimiento.
    """
    mensajes = [SystemMessage(content=system_prompt), HumanMessage(content=pregunta_usuario)]
    respuesta = model.invoke(mensajes)
    return {"messages": [respuesta]}

workflow = StateGraph(State)
workflow.add_node("bot", nodo_helpbot)
workflow.add_edge(START, "bot")
workflow.add_edge("bot", END)
app_graph = workflow.compile()

# ---------------------------------------------------------
# 4. INTERFAZ DE CHAT
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial visual en pantalla
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Escribe tu pregunta sobre las políticas aquí..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando manuales..."):
            try:
                # Se envía ÚNICAMENTE la última pregunta para evitar cargar respuestas anteriores
                resultado = app_graph.invoke({"messages": [HumanMessage(content=prompt)]})
                
                res_content = resultado['messages'][-1].content
                if isinstance(res_content, list):
                    respuesta_texto = "".join([b.get("text", "") if isinstance(b, dict) else str(b) for b in res_content])
                else:
                    respuesta_texto = res_content

                st.write(respuesta_texto)
                st.session_state.messages.append({"role": "assistant", "content": respuesta_texto})
            except Exception as e:
                st.error("❌ Error al procesar la consulta.")
                st.caption(f"Detalle técnico: {e}")
