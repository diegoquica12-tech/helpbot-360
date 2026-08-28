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
# 1. GESTIÓN Y VALIDACIÓN DE API KEY
# ---------------------------------------------------------
api_key = None

if "GOOGLE_API_KEY" in st.secrets and st.secrets["GOOGLE_API_KEY"]:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    api_key = st.sidebar.text_input("Ingresa tu GOOGLE_API_KEY:", type="password")

if not api_key:
    st.info("Por favor ingresa tu GOOGLE_API_KEY en los Secrets de Streamlit o en la barra lateral.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key

# ---------------------------------------------------------
# 2. INICIALIZACIÓN CON CACHÉ Y PARÁMETRO EXPLÍCITO
# ---------------------------------------------------------
@st.cache_resource
def iniciar_bot(key_api):
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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    # Asignación directa de API Key y modelo gemini-2.0-flash
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=key_api,
        temperature=0
    )

    def nodo_helpbot(state: State):
        pregunta_usuario = state["messages"][-1].content
        docs_recuperados = retriever.invoke(pregunta_usuario)
        contexto = "\n\n".join([doc.page_content for doc in docs_recuperados])
        
        system_prompt = f"""
        Eres HelpBot 360, el asistente de TerraCampo S.A.S.
        Responde a la consulta de forma amigable basándote ÚNICAMENTE en este contexto:
        <contexto>
        {contexto}
        </contexto>
        Si la información no está en el contexto, di que no la tienes y sugiere contactar a Recursos Humanos o Mantenimiento.
        """
        mensajes = [SystemMessage(content=system_prompt)] + state["messages"]
        respuesta = model.invoke(mensajes)
        return {"messages": [respuesta]}

    workflow = StateGraph(State)
    workflow.add_node("bot", nodo_helpbot)
    workflow.add_edge(START, "bot")
    workflow.add_edge("bot", END)
    
    return workflow.compile()

# Se inicializa pasando la clave directamente
app_graph = iniciar_bot(api_key)

# ---------------------------------------------------------
# 3. INTERFAZ DE CHAT
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.graph_state = {"messages": []}

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Escribe tu pregunta sobre las políticas aquí..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando manuales..."):
            st.session_state.graph_state["messages"].append(HumanMessage(content=prompt))
            resultado = app_graph.invoke(st.session_state.graph_state)
            
            res_content = resultado['messages'][-1].content
            if isinstance(res_content, list):
                respuesta_texto = "".join([b.get("text", "") if isinstance(b, dict) else str(b) for b in res_content])
            else:
                respuesta_texto = res_content

            st.write(respuesta_texto)
            st.session_state.messages.append({"role": "assistant", "content": respuesta_texto})
