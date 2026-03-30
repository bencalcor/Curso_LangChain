import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# =========================
# Configuración de la página
# =========================
st.set_page_config(page_title="Chatbot Básico", page_icon="🤖", layout="centered")

st.title("🤖 Chatbot Básico con LangChain")
st.markdown(
    "Este es un chatbot de ejemplo construido con LangChain + Streamlit. "
    "Escribe tu mensaje para comenzar."
)

# =========================
# Inicializar session_state
# =========================
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

if "confirm_clear" not in st.session_state:
    st.session_state.confirm_clear = False

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("Configuración")

    if st.button("🗑️ Nueva conversación", use_container_width=True):
        st.session_state.confirm_clear = True

    if st.session_state.confirm_clear:
        st.warning("¿Seguro que deseas borrar la conversación?")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Sí, borrar", use_container_width=True):
                st.session_state.mensajes = []
                st.session_state.confirm_clear = False
                st.rerun()

        with col2:
            if st.button("Cancelar", use_container_width=True):
                st.session_state.confirm_clear = False
                st.rerun()

    st.divider()

    temperature = st.slider("Temperatura", 0.0, 1.0, 0.5, 0.1)
    model_name = st.selectbox("Modelo", ["gpt-3.5-turbo", "gpt-4", "gpt-4o-mini"], index=2)

    personalidad = st.selectbox(
        "Personalidad del Asistente",
        [
            "Útil y amigable",
            "Profesional y formal",
            "Casual y relajado",
            "Experto técnico",
            "Creativo y divertido"
        ]
    )

    system_messages = {
        "Útil y amigable": (
            "Eres un asistente útil y amigable llamado jAIme. "
            "Responde de manera clara, breve y amable."
        ),
        "Profesional y formal": (
            "Eres un asistente profesional y formal. "
            "Proporciona respuestas precisas, bien estructuradas y respetuosas."
        ),
        "Casual y relajado": (
            "Eres un asistente casual y relajado llamado jAIme. "
            "Habla de forma natural, cercana y amigable."
        ),
        "Experto técnico": (
            "Eres un asistente experto técnico. "
            "Da respuestas detalladas, precisas y bien explicadas."
        ),
        "Creativo y divertido": (
            "Eres un asistente creativo y divertido llamado jAIme. "
            "Usa ejemplos, comparaciones y un tono alegre."
        )
    }

    chat_model = ChatOpenAI(model=model_name, temperature=temperature)

# =========================
# ChatPromptTemplate
# =========================
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_messages[personalidad]),
    ("human", "Historial de conversación:\n{historial}\n\nPregunta actual: {mensaje}")
])

# Cadena LCEL
cadena = chat_prompt | chat_model

# =========================
# Función para preparar historial
# =========================
def preparar_historial(historial):
    historial_texto = ""

    for msg in historial[-10:]:
        if isinstance(msg, HumanMessage):
            historial_texto += f"Usuario: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            historial_texto += f"Asistente: {msg.content}\n"

    if not historial_texto.strip():
        historial_texto = "(No hay historial previo)"

    return historial_texto

# =========================
# Mostrar historial existente
# =========================
for msg in st.session_state.mensajes:
    if isinstance(msg, SystemMessage):
        continue

    role = "assistant" if isinstance(msg, AIMessage) else "user"

    with st.chat_message(role):
        st.markdown(msg.content)

# =========================
# Input del usuario
# =========================
pregunta = st.chat_input("Escribe tu mensaje:")

if pregunta:
    with st.chat_message("user"):
        st.markdown(pregunta)

    historial_texto = preparar_historial(st.session_state.mensajes)

    try:
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            for chunk in cadena.stream({
                "mensaje": pregunta,
                "historial": historial_texto
            }):
                if hasattr(chunk, "content") and chunk.content:
                    full_response += chunk.content
                    response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

        st.session_state.mensajes.append(HumanMessage(content=pregunta))
        st.session_state.mensajes.append(AIMessage(content=full_response))

    except Exception as e:
        st.error(f"Error al generar respuesta: {str(e)}")
        st.info("Verifica que tu API Key de OpenAI esté configurada correctamente.")