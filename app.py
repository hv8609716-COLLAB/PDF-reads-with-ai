import os
import streamlit as st
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# ---------- Config ----------
HF_TOKEN = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN"))
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE = 250
CHUNK_OVERLAP = 50
TOP_K = 4

st.set_page_config(page_title="AI Research Assistant", page_icon="🤖", layout="centered")


# ---------- Cached resources ----------
@st.cache_resource
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource
def load_client():
    return InferenceClient(model=LLM_MODEL, token=HF_TOKEN, provider="auto")


embed_model = load_embed_model()
client = load_client()

# ---------- Session state ----------
if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = []
if "doc_embeddings" not in st.session_state:
    st.session_state.doc_embeddings = None
if "full_document_text" not in st.session_state:
    st.session_state.full_document_text = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": ..., "content": ...}
if "pdf_status" not in st.session_state:
    st.session_state.pdf_status = ""


# ---------- Core functions (same logic as original) ----------
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def read_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "

    full_text = text.strip()

    if not full_text:
        return "⚠️ text in pdf not find (scanned/image PDF)", [], None

    chunks = chunk_text(full_text)

    if not chunks:
        return "⚠️ PDF is empty", [], None

    embeddings = embed_model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

    status = f"✅ PDF ready! {len(reader.pages)} pages, {len(chunks)} chunks indexed!"
    return status, chunks, embeddings, full_text


def retrieve_chunks(question, doc_chunks, doc_embeddings, top_k=TOP_K):
    q_embedding = embed_model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    scores = np.dot(doc_embeddings, q_embedding.T).flatten()
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [doc_chunks[i] for i in top_idx]


def is_summary_request(question):
    keywords = ["summarize", "summary", "summarise", "overview", "gist", "tl;dr", "short karo", "sankshep"]
    q_lower = question.lower()
    return any(k in q_lower for k in keywords)


def call_llm(prompt, system_msg="You are a helpful research assistant that answers based only on the given document context."):
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt}
    ]
    response = client.chat_completion(messages=messages, max_tokens=600, temperature=0.3)
    return response.choices[0].message.content


def answer_question(question):
    full_document_text = st.session_state.full_document_text
    doc_chunks = st.session_state.doc_chunks
    doc_embeddings = st.session_state.doc_embeddings

    if not full_document_text:
        return "⚠️ first upload the pdf, then click 'Process PDF'!"

    if not question or not question.strip():
        return "⚠️ ask the question first!"

    try:
        if is_summary_request(question):
            if len(full_document_text.split()) > 3000:
                partial_summaries = []
                for c in doc_chunks[:8]:
                    partial = call_llm(f"Summarize this text in 2-3 sentences:\n\n{c}")
                    partial_summaries.append(partial)
                combined = "\n".join(partial_summaries)
                final_answer = call_llm(
                    f"Combine these partial summaries into one clear, well-structured summary:\n\n{combined}\n\nUser's request: {question}"
                )
            else:
                final_answer = call_llm(f"Document:\n\n{full_document_text}\n\nTask: {question}")
        else:
            context_chunks = retrieve_chunks(question, doc_chunks, doc_embeddings)
            context = "\n\n---\n\n".join(context_chunks)
            prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say so clearly.

Context:
{context}

Question: {question}

Answer:"""
            final_answer = call_llm(prompt)

    except Exception as e:
        final_answer = f"❌ Error: {str(e)}"

    return final_answer


# ---------- UI ----------
st.title("🤖 AI Research Assistant")
st.markdown("**by Harsh Vardhan**")
st.markdown("Upload the PDF and ask anything about it — summary, explanation, specific facts!")

st.subheader("📄 PDF Upload")
uploaded_pdf = st.file_uploader("Choose a PDF file", type=["pdf"])

col1, col2 = st.columns([1, 3])
with col1:
    process_clicked = st.button("Process PDF 📂")
with col2:
    if st.session_state.pdf_status:
        st.info(st.session_state.pdf_status)

if process_clicked:
    if uploaded_pdf is None:
        st.session_state.pdf_status = "❌ PDF select first!"
    else:
        with st.spinner("Reading and indexing PDF..."):
            result = read_pdf(uploaded_pdf)
            if len(result) == 3:
                # error case: status, [], None
                status, chunks, embeddings = result
                st.session_state.pdf_status = status
            else:
                status, chunks, embeddings, full_text = result
                st.session_state.pdf_status = status
                st.session_state.doc_chunks = chunks
                st.session_state.doc_embeddings = embeddings
                st.session_state.full_document_text = full_text
    st.rerun()

st.divider()
st.subheader("💬 Chat")

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
question = st.chat_input("e.g. summarize this document / what is the main topic? / explain section 2...")

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = answer_question(question)
            st.markdown(answer)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
