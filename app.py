import streamlit as st
import tempfile

from groq import Groq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Ask My Docs",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Ask My Docs – RAG PDF Q&A App")


# =========================
# GROQ API
# =========================
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

client = Groq(api_key=GROQ_API_KEY)


# =========================
# SESSION STATE
# =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# =========================
# EMBEDDINGS MODEL
# =========================
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================
# BUILD VECTORSTORE
# =========================
@st.cache_resource
def build_vectorstore(uploaded_files):

    all_docs = []

    for uploaded_file in uploaded_files:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:

            tmp_file.write(uploaded_file.read())

            temp_path = tmp_file.name

        loader = PyPDFLoader(temp_path)

        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = uploaded_file.name

        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(all_docs)

    vectorstore = FAISS.from_documents(
        chunks,
        embedding_model
    )

    return vectorstore


# =========================
# FILE UPLOAD
# =========================
uploaded_files = st.file_uploader(
    "Upload PDF Files",
    type="pdf",
    accept_multiple_files=True
)

vectorstore = None

if uploaded_files:

    with st.spinner("Processing PDFs..."):

        vectorstore = build_vectorstore(uploaded_files)

    st.success("PDFs processed successfully ✅")


# =========================
# RETRIEVER
# =========================
retriever = (
    vectorstore.as_retriever(search_kwargs={"k": 3})
    if vectorstore else None
)


# =========================
# CHAT HISTORY
# =========================
st.subheader("💬 Chat")

for msg in st.session_state.chat_history:

    if msg["role"] == "user":
        st.markdown(f"**🧑 You:** {msg['content']}")

    else:
        st.markdown(f"**🤖 AI:** {msg['content']}")


# =========================
# QUESTION INPUT
# =========================
question = st.text_input(
    "Ask a question about your documents"
)


# =========================
# MAIN LOGIC
# =========================
if question and retriever:

    with st.spinner("Thinking..."):

        docs = retriever.invoke(question)

        context = "\n\n".join([
            doc.page_content for doc in docs
        ])

        prompt = f"""
Answer ONLY using the provided context.

If answer is not found,
say:
"I don't know based on the documents."

Context:
{context}

Question:
{question}
"""

        messages = [
            {
                "role": "system",
                "content": "You answer only from provided context."
            }
        ]

        messages.extend(st.session_state.chat_history)

        messages.append({
            "role": "user",
            "content": prompt
        })

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            stream=True
        )

        answer_box = st.empty()

        full_answer = ""

        for chunk in response:

            if chunk.choices[0].delta.content:

                content = chunk.choices[0].delta.content

                full_answer += content

                answer_box.markdown(full_answer)

    # SAVE CHAT HISTORY
    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": full_answer
    })

    # FINAL ANSWER
    st.subheader("📌 Final Answer")

    st.write(full_answer)

    # SOURCES
    st.subheader("📚 Sources")

    for i, doc in enumerate(docs):

        source = doc.metadata.get("source", "Unknown")

        page = doc.metadata.get("page", "N/A")

        with st.expander(
            f"Source {i+1} | File: {source} | Page: {page}"
        ):
            st.write(doc.page_content)