# =========================
# Carga de bibliotecas
# =========================
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
import streamlit as st


# =========================
# Configuración de la página
# =========================
st.set_page_config(
    page_title="Chatbot Básico",
    page_icon="🤖",
    layout="centered"
)

# =========================
# Estilos personalizados
# =========================
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .subtitle {
        font-size: 1rem;
        color: #B0B3B8;
        margin-bottom: 1.2rem;
    }

    .custom-card {
        background-color: rgba(255, 255, 255, 0.03);
        padding: 1rem 1.2rem;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1rem;
    }

    .stChatMessage {
        border-radius: 14px;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    .sidebar-info {
        font-size: 0.95rem;
        color: #D1D5DB;
        background-color: rgba(255,255,255,0.04);
        padding: 0.8rem 1rem;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.06);
    }
</style>
""", unsafe_allow_html=True)


# =========================
# Inicialización de estados
# =========================
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

if "confirm_clear" not in st.session_state:
    st.session_state.confirm_clear = False


# =========================
# Encabezado principal
# =========================
st.markdown('<div class="main-title">🤖 Chatbot Básico con LangChain</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Este es un Chatbot de Ejemplo construido con LangChain y Streamlit, por Ernie Calderon. Escribe tu mensaje para comenzar.</div>',
    unsafe_allow_html=True
)


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("Configuración")

    if st.button("🗑️ Nueva conversación", use_container_width=True):
        st.session_state.confirm_clear = True

    if st.session_state.confirm_clear:
        st.warning("¿Seguro que deseas borrar la conversación actual?")
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

    temperature = st.slider("Temperatura", 0.0, 1.0, 0.7, 0.1)
    model_name = st.selectbox("Modelo", ["gpt-3.5-turbo", "gpt-4", "gpt-4o-mini"], index=2)

    st.divider()

    total_msgs = len([m for m in st.session_state.mensajes if not isinstance(m, SystemMessage)])
    st.markdown(
        f"""
        <div class="sidebar-info">
            <b>Modelo activo:</b> {model_name}<br>
            <b>Temperatura:</b> {temperature}<br>
            <b>Mensajes en historial:</b> {total_msgs}
        </div>
        """,
        unsafe_allow_html=True
    )

    chat_model = ChatOpenAI(model=model_name, temperature=temperature)


# =========================
# Prompt
# =========================
prompt_template = PromptTemplate(
    input_variables=["mensaje", "historial"],
    template="""
Eres un asistente útil, amigable y claro llamado jAIme.

Historial de conversación:
{historial}

Responde de manera clara, breve y útil a la siguiente pregunta del usuario:
{mensaje}
"""
)

# Crear cadena usando LCEL
cadena = prompt_template | chat_model


# =========================
# Función para convertir historial a texto
# =========================
def convertir_historial_a_texto(historial):
    lineas = []

    for msg in historial:
        if isinstance(msg, SystemMessage):
            continue
        elif isinstance(msg, HumanMessage):
            lineas.append(f"Usuario: {msg.content}")
        elif isinstance(msg, AIMessage):
            lineas.append(f"Asistente: {msg.content}")

    return "\n".join(lineas)


# =========================
# Mensaje inicial si no hay historial
# =========================
if not st.session_state.mensajes:
    st.markdown(
        """
        <div class="custom-card">
            <b>Bienvenido.</b><br>
            Puedes hacer preguntas, pedir explicaciones o conversar libremente con el asistente.
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================
# Renderizar historial existente
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

    # Convertir historial a texto antes de enviarlo al prompt
    historial_texto = convertir_historial_a_texto(st.session_state.mensajes)

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