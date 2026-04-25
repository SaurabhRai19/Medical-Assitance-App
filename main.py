import streamlit as st
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Medical Assistant", page_icon="🩺", layout="centered")

# -------------------- CSS (SAFE STYLING) --------------------
st.markdown("""
<style>

/* Page background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #e9f2ff, #f7fbff);
}

/* Center card */
.block-container {
    background: white;
    padding: 2rem;
    border-radius: 18px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    max-width: 800px;
}

/* Chat message styling */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 10px;
}

/* Assistant */
[data-testid="stChatMessage"][aria-label="assistant"] {
    background-color: #f1f6fb;
}

/* User */
[data-testid="stChatMessage"][aria-label="user"] {
    background-color: #dbeafe;
}

/* Disclaimer */
.disclaimer {
    background: #fff4e5;
    border-left: 5px solid #ffa726;
    padding: 10px;
    border-radius: 10px;
    font-size: 13px;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown("## 🩺 AI Medical Assistant")

st.markdown("""
<div class="disclaimer">
⚠️ This assistant provides general medical information only.
Consult a qualified doctor for professional advice.
</div>
""", unsafe_allow_html=True)

# -------------------- ENV --------------------
load_dotenv()
# -------------------- LOAD SECRETS (UNIFIED) --------------------
def load_secrets():
    try:
        import streamlit as st
        # If running inside Streamlit Cloud
        if "PINECONE_API_KEY" in st.secrets:
            return {
                "PINECONE_API_KEY": st.secrets["PINECONE_API_KEY"],
                "GEMINI_API_KEY": st.secrets["GEMINI_API_KEY"]
            }
    except:
        pass

    # Fallback to .env (local)
    load_dotenv()
    return {
        "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY")
    }


secrets = load_secrets()

# -------------------- SET ENV --------------------
os.environ["PINECONE_API_KEY"] = secrets["PINECONE_API_KEY"]
os.environ["GEMINI_API_KEY"] = secrets["GEMINI_API_KEY"]

# -------------------- INIT RAG --------------------
@st.cache_resource
def init_rag():
    embeddings = download_hugging_face_embeddings()

    docsearch = PineconeVectorStore.from_existing_index(
        index_name="medical-chatbot",
        embedding=embeddings
    )

    retriever = docsearch.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash")

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, qa_chain)

with st.spinner("Loading..."):
    rag_chain = init_rag()

# -------------------- STATE --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------- CHAT DISPLAY --------------------
for msg in st.session_state.messages:
    avatar = "🧑‍🦱" if msg["role"] == "user" else "👩‍⚕️"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# -------------------- INPUT --------------------
user_input = st.chat_input("Describe your symptoms...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user", avatar="🧑‍🦱"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="👩‍⚕️"):
        with st.spinner("Analyzing..."):
            response = rag_chain.invoke({"input": user_input})
            answer = response["answer"]
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})