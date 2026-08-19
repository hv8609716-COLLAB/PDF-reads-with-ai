import os
import gradio as gr
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# ---------- Config ----------
HF_TOKEN = os.environ.get("HF_TOKEN")  
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"   
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE = 250       
CHUNK_OVERLAP = 50     
TOP_K = 4              

# ---------- Global state ----------
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
client = InferenceClient(model=LLM_MODEL, token=HF_TOKEN)

doc_chunks = []
doc_embeddings = None
full_document_text = ""


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


def read_pdf(file):
    global doc_chunks, doc_embeddings, full_document_text

    if file is None:
        return "❌ PDF select first!"

    reader = PdfReader(file.name)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "

    full_document_text = text.strip()

    if not full_document_text:
        return "⚠️ text in pdf not find (scanned/image PDF )"

    doc_chunks = chunk_text(full_document_text)

    if not doc_chunks:
        return "⚠️ PDF is empty"

    doc_embeddings = embed_model.encode(doc_chunks, convert_to_numpy=True, normalize_embeddings=True)

    return f"✅ PDF ready! {len(reader.pages)} pages, {len(doc_chunks)} chunks indexed!"


def retrieve_chunks(question, top_k=TOP_K):
    q_embedding = embed_model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    scores = np.dot(doc_embeddings, q_embedding.T).flatten()  # cosine similarity
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


def answer_question(question, history):
    global full_document_text

    if history is None:
        history = []

    if not full_document_text:
        return history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "⚠️ first upload the pdf 'Process PDF' click it!"}
        ]

    if not question or not question.strip():
        return history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "⚠️ ask the Question first!"}
        ]

    try:
        if is_summary_request(question):
            # Lambe document ke liye chunk-wise map-reduce summary
            if len(full_document_text.split()) > 3000:
                partial_summaries = []
                for c in doc_chunks[:8]:  # bohot lambe doc ke liye limit
                    partial = call_llm(f"Summarize this text in 2-3 sentences:\n\n{c}")
                    partial_summaries.append(partial)
                combined = "\n".join(partial_summaries)
                final_answer = call_llm(f"Combine these partial summaries into one clear, well-structured summary:\n\n{combined}\n\nUser's request: {question}")
            else:
                final_answer = call_llm(f"Document:\n\n{full_document_text}\n\nTask: {question}")
        else:
            context_chunks = retrieve_chunks(question)
            context = "\n\n---\n\n".join(context_chunks)
            prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say so clearly.

Context:
{context}

Question: {question}

Answer:"""
            final_answer = call_llm(prompt)

    except Exception as e:
        final_answer = f"❌ Error: {str(e)}"

    return history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": final_answer}
    ]


with gr.Blocks(title="AI Research Assistant") as demo:
    gr.Markdown("# 🤖 AI Research Assistant\n**by Harsh Vardhan**\n upload the pdf and ask anything about it — summary, explanation, specific facts!")

    with gr.Row():
        pdf_input = gr.File(
            label="📄 PDF Upload",
            file_types=[".pdf"]
        )
        status = gr.Textbox(
            label="Status",
            interactive=False
        )

    process_btn = gr.Button("Process PDF 📂")
    process_btn.click(fn=read_pdf, inputs=pdf_input, outputs=status)

    chatbot = gr.Chatbot(label="💬 Chat", height=400)
    question = gr.Textbox(
        label="❓ Question",
        placeholder="e.g. summarize this document / what is the main topic? / explain section 2..."
    )
    ask_btn = gr.Button("Ask 🔍")

    ask_btn.click(fn=answer_question, inputs=[question, chatbot], outputs=chatbot).then(
        lambda: "", None, question
    )
    question.submit(fn=answer_question, inputs=[question, chatbot], outputs=chatbot).then(
        lambda: "", None, question
    )

demo.launch()
