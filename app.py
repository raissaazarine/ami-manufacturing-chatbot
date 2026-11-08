"""Manufacturing chatbot over the Komatsu product guide, with citable sources.

    pip install -r requirements.txt
    echo "OPENAI_API_KEY=sk-..." > .env
    streamlit run app.py
"""
from __future__ import annotations

import base64
import json
import os
import re

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI, OpenAIError

from trial import PDF_PATH, load_source_documents

load_dotenv(find_dotenv())

OPENROUTER_URL = "https://openrouter.ai/api/v1"
# OpenRouter keys start with "sk-or-" and need OpenRouter's URL and "openai/" model names.
USES_OPENROUTER = os.getenv("OPENAI_API_KEY", "").startswith("sk-or-")
MODEL_PREFIX = "openai/" if USES_OPENROUTER else ""
CHAT_MODEL = os.getenv("OPENAI_MODEL", MODEL_PREFIX + "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", MODEL_PREFIX + "text-embedding-3-small")
TOP_K = 4  # documents retrieved by similarity
MAX_CONTEXT_DOCS = 8  # models named in a question are added on top of TOP_K
CHUNK_SIZE = 1500  # characters; model spec documents already fit and stay whole
EMBEDDING_BATCH = 100
HISTORY_TURNS = 6  # recent chat messages sent along with each question

QNA_PROMPT = """Anda adalah asisten engineering manufaktur alat berat.
Jawab hanya berdasarkan KONTEKS di bawah, dan sebutkan sumbernya seperti [hal. 19].
Jika informasinya tidak ada di konteks, katakan dengan jelas bahwa dokumen tidak memuatnya;
jangan mengisi angka dari pengetahuan umum. Boleh menyebut model terdekat yang ada di konteks,
dengan menegaskan bahwa itu model yang berbeda. Jawab dalam bahasa yang dipakai penanya.

KONTEKS:
{context}"""

CONDENSE_PROMPT = """Tulis ulang pertanyaan terakhir pengguna menjadi satu pertanyaan yang berdiri
sendiri, memakai riwayat percakapan untuk melengkapi rujukan seperti "model itu".
Keluarkan pertanyaannya saja."""

ANALYSIS_PROMPT = """Anda membantu engineer menganalisa masalah pada alat berat.
Dokumen yang tersedia hanya katalog spesifikasi: tidak ada data kerusakan, manual servis, atau
pengalaman engineer dan customer. Jadi semua penyebab yang Anda sebut adalah hipotesis umum,
bukan temuan dari data. Jawab dalam Bahasa Indonesia dengan format:

**Ringkasan masalah**
**Hipotesis penyebab** (urutkan dari yang paling mungkin, sebut alasannya)
**Langkah pemeriksaan** (yang paling aman dan mudah lebih dulu)
**Spesifikasi terkait** (hanya dari data di bawah, sebut halamannya; tulis "tidak ada" bila tidak relevan)
**Data yang dibutuhkan** untuk memastikan penyebabnya

Jika ada foto, jelaskan hanya apa yang benar-benar terlihat. Akhiri dengan mengingatkan agar
diagnosa dikonfirmasi teknisi berwenang dan manual servis resmi.

SPESIFIKASI {model}:
{context}"""

EVALUATION_PROMPT = """Nilai jawaban chatbot RAG berikut. Beri skor 0 sampai 1 untuk:
- faithfulness: seberapa banyak klaim jawaban yang didukung KONTEKS
- answer_relevance: seberapa langsung jawaban menjawab PERTANYAAN
- context_relevance: seberapa relevan KONTEKS terhadap PERTANYAAN
Keluarkan JSON saja dengan ketiga kunci itu.

PERTANYAAN: {question}

KONTEKS:
{context}

JAWABAN: {answer}"""


def chunk_text(text):
    """Split long text on line breaks; a single longer line becomes its own chunk."""
    chunks, current = [], ""
    for line in text.split("\n"):
        if current and len(current) + len(line) + 1 > CHUNK_SIZE:
            chunks.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        chunks.append(current)
    return chunks


def read_excel_documents(file):
    documents = []
    for sheet, frame in pd.read_excel(file, sheet_name=None).items():
        documents.append({
            "source": file.name,
            "sheet": sheet,
            "type": "sheet",
            "text": frame.to_string(index=False),
        })
    return documents


def read_upload(file):
    if file.name.lower().endswith(".pdf"):
        return load_source_documents(file)  # needs only .name and an open file
    return read_excel_documents(file)


def split_documents(documents):
    pieces = []
    for document in documents:
        for chunk in chunk_text(document["text"]):
            pieces.append({**document, "text": chunk})
    return pieces


def citation(document):
    if "sheet" in document:
        return f"{document['source']}, sheet {document['sheet']}"
    return f"{document['source']}, hal. {document['page']}"


def context_block(documents):
    return "\n\n".join(f"[{citation(doc)}]\n{doc['text']}" for doc in documents)


@st.cache_data(show_spinner=False)
def load_booklet():
    return load_source_documents(PDF_PATH)


@st.cache_data(show_spinner=False)
def parse_upload(file_id, _file):
    # file_id is the cache key; UploadedFile itself is not hashable.
    return read_upload(_file)


@st.cache_data(show_spinner=False)
def embed_texts(texts, _client):
    vectors = []
    for start in range(0, len(texts), EMBEDDING_BATCH):
        batch = list(texts[start:start + EMBEDDING_BATCH])
        response = _client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vectors += [item.embedding for item in response.data]
    matrix = np.array(vectors, dtype=float)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def named_in_question(document, question):
    """True when the question names this model, in full or without its suffix."""
    model = document.get("model")
    if not model:
        return False
    names = {model.split(" (")[0], model.split("-")[0]}
    return any(
        re.search(rf"(?<![\w-]){re.escape(name)}(?![\w])", question, re.IGNORECASE)
        for name in names
    )


def retrieve(question, documents, client):
    """Exact model names first, because embeddings match part codes poorly."""
    named = [doc for doc in documents if named_in_question(doc, question)]
    matrix = embed_texts(tuple(doc["text"] for doc in documents), client)
    query = embed_texts((question,), client)[0]
    ranked = [documents[i] for i in np.argsort(matrix @ query)[::-1][:TOP_K]]
    selected = named + [doc for doc in ranked if doc not in named]
    return selected[:MAX_CONTEXT_DOCS]


def complete(client, messages, **options):
    response = client.chat.completions.create(
        model=CHAT_MODEL, temperature=0, messages=messages, **options
    )
    return response.choices[0].message.content or ""


def condense_question(question, history, client):
    if not history:
        return question
    transcript = "\n".join(f"{msg['role']}: {msg['content']}" for msg in history)
    return complete(client, [
        {"role": "system", "content": CONDENSE_PROMPT},
        {"role": "user", "content": f"{transcript}\nuser: {question}"},
    ]).strip() or question


def answer_question(question, documents, client):
    history = st.session_state["chat_history"][-HISTORY_TURNS:]
    plain_history = [{"role": msg["role"], "content": msg["content"]} for msg in history]
    standalone = condense_question(question, plain_history, client)
    sources = retrieve(standalone, documents, client)
    answer = complete(client, [
        {"role": "system", "content": QNA_PROMPT.format(context=context_block(sources))},
        *plain_history,
        {"role": "user", "content": question},
    ])
    return answer, sources


def evaluate_answer(question, answer, sources, client):
    """Score an answer with the LLM; a failure here must not lose the answer."""
    context = context_block(sources)
    row = {"Pertanyaan": question, "Jawaban": answer, "Sumber": ", ".join(citation(d) for d in sources)}
    try:
        scores = json.loads(complete(
            client,
            [{"role": "user", "content": EVALUATION_PROMPT.format(
                question=question, context=context, answer=answer)}],
            response_format={"type": "json_object"},
        ))
        row["Faithfulness"] = float(scores["faithfulness"])
        row["Answer Relevance"] = float(scores["answer_relevance"])
        row["Context Relevance"] = float(scores["context_relevance"])
    except (OpenAIError, ValueError, KeyError, TypeError) as error:
        row["Error"] = str(error)
    return row


def analyze_problem(model, symptoms, photo, documents, client):
    specs = [doc for doc in documents if doc.get("model") == model]
    content = [{"type": "text", "text": f"Model: {model}\nGejala: {symptoms}"}]
    if photo:
        encoded = base64.b64encode(photo.getvalue()).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{photo.type};base64,{encoded}"},
        })
    return complete(client, [
        {"role": "system", "content": ANALYSIS_PROMPT.format(model=model, context=context_block(specs))},
        {"role": "user", "content": content},
    ])


def show_qna(documents, client):
    st.subheader("Fitur QnA")
    question = st.chat_input("Masukkan pertanyaan:", key="chat_input_qna")

    if question:
        try:
            with st.spinner("Mencari jawaban di dokumen..."):
                answer, sources = answer_question(question, documents, client)
                evaluation = evaluate_answer(question, answer, sources, client)
        except OpenAIError as error:
            st.error(f"Gagal menghubungi OpenAI: {error}")
        else:
            st.session_state["chat_history"] += [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
            st.session_state["last_sources"] = sources
            st.session_state["evaluation_metrics"].append(evaluation)

    if st.session_state["last_sources"]:
        st.sidebar.write("Dokumen Sumber:")
        for doc in st.session_state["last_sources"]:
            with st.sidebar.expander(citation(doc)):
                st.text(doc["text"])

    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.button("Clear History", key="clear_history_qna"):
        st.session_state["chat_history"] = []
        st.session_state["evaluation_metrics"] = []
        st.session_state["last_sources"] = []
        st.rerun()


def show_analysis(documents, client):
    st.subheader("Fitur Analisa Masalah Produk")
    st.warning(
        "Dokumen saat ini hanya berisi spesifikasi, tanpa data kerusakan atau pengalaman engineer. "
        "Hasilnya berupa hipotesis umum, bukan perbandingan dengan kasus nyata."
    )
    models = sorted({doc["model"] for doc in documents if doc.get("model")})
    model = st.selectbox("Model:", models, key="analysis_model")
    symptoms = st.text_area("Gejala / masalah:", key="analysis_symptoms")
    photo = st.file_uploader(
        "Foto produk bermasalah (opsional):", type=["jpg", "jpeg", "png"], key="analysis_photo"
    )
    if photo:
        st.image(photo, width=300)

    if st.button("Analisa", key="submit_analysis"):
        if not symptoms.strip():
            st.error("Harap isi gejala atau masalahnya.")
            return
        try:
            with st.spinner("Menganalisa, harap tunggu..."):
                result = analyze_problem(model, symptoms, photo, documents, client)
        except OpenAIError as error:
            st.error(f"Gagal menghubungi OpenAI: {error}")
        else:
            st.markdown(result)


def show_evaluation():
    st.subheader("Evaluation Metrics")
    rows = st.session_state["evaluation_metrics"]
    if not rows:
        st.write("Belum ada data. Ajukan pertanyaan di fitur QnA terlebih dahulu.")
        return
    st.caption(
        "Skor dinilai oleh LLM dan bersifat indikatif. Context Recall tidak dihitung "
        "karena butuh jawaban acuan yang disusun manusia."
    )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def main():
    st.set_page_config(page_title="Manufacturing Chatbot", layout="wide")
    st.sidebar.title("Manufacturing Chatbot")
    st.sidebar.write("Pilih fitur yang ingin dipakai")
    feature = st.sidebar.selectbox(
        "Pilih fitur:", ["QnA", "Analisa Masalah Produk", "Evaluation Metrics"]
    )

    for key in ("chat_history", "evaluation_metrics", "last_sources"):
        st.session_state.setdefault(key, [])

    if feature == "Evaluation Metrics":
        show_evaluation()
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY belum diisi. Buat file .env di folder project: OPENAI_API_KEY=sk-...")
        return
    base_url = os.getenv("OPENAI_BASE_URL") or (OPENROUTER_URL if USES_OPENROUTER else None)
    client = OpenAI(api_key=api_key, base_url=base_url)

    documents = load_booklet()
    st.sidebar.header("Dokumen")
    st.sidebar.caption(f"Aktif: {PDF_PATH.name}")
    upload = st.sidebar.file_uploader("Tambah file Excel atau PDF", type=["xlsx", "pdf"], key="file_uploader")
    if upload:
        try:
            extra = parse_upload(upload.file_id, upload)
        except Exception as error:  # unreadable uploads come in many library-specific types
            st.sidebar.error(f"Gagal membaca file: {error}")
        else:
            if extra:
                documents = documents + extra
                st.sidebar.success(f"{upload.name}: {len(extra)} dokumen ditambahkan")
            else:
                st.sidebar.error("Tidak ada teks yang terbaca (kemungkinan PDF hasil scan).")

    if feature == "QnA":
        show_qna(split_documents(documents), client)
    else:
        show_analysis(documents, client)


main()
