import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from openai import RateLimitError
from transformers import pipeline
import tempfile
import time

st.set_page_config(page_title="AI Document Q&A", page_icon="📄", layout="centered")

st.title("📄 AI-Powered Document Question Answering System")

st.write("Upload any PDF and ask questions in English, Tamil, or mixed-language")

uploaded_file = st.file_uploader("Upload your any PDF", type=["pdf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        pdf_path = temp_file.name

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    chunks = text_splitter.split_documents(documents)

    embeddings=HuggingFaceEmbeddings(model_name="sentence-transfomers/all-MiniLM-L6-v2")

    # --- Fix: build the vector store in small batches with retry/backoff
    # so a temporary rate limit (or a burst of tokens) doesn't kill the app.
    def build_vector_store(chunks, embeddings, batch_size=20, max_retries=5):
        vector_store = None
        progress = st.progress(0, text="Embedding document...")
        total = len(chunks)
        for i in range(0, total, batch_size):
            batch = chunks[i:i + batch_size]
            for attempt in range(max_retries):
                try:
                    if vector_store is None:
                        vector_store = Chroma.from_documents(documents=batch, embedding=embeddings)
                    else:
                        vector_store.add_documents(batch)
                    break
                except RateLimitError:
                    wait = 2 ** attempt
                    st.warning(f"Rate limited by OpenAI, retrying in {wait}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait)
            else:
                st.error(
                    "Failed to embed the document after multiple retries.\n\n"
                    "This usually means your OpenAI account has no available quota. "
                    "Check platform.openai.com → Settings → Billing to confirm you have credits, "
                    "then try again."
                )
                st.stop()
            progress.progress(min((i + batch_size) / total, 1.0), text="Embedding document...")
        progress.empty()
        return vector_store

    vector_store = build_vector_store(chunks, embeddings)

    # Fix: was "search_kwarges" (typo) -> "search_kwargs"
    retriever = vector_store.as_retriever(search_kwargs={"k": 8})

    st.success("PDF processed successfully!")

    st.write("Uploaded PDF:", uploaded_file.name)

    question = st.text_input("🔎 Ask a question about the PDF:")

    if st.button("Ask"):
        if question.strip():
            search_prompt = f"""
You are a query understanding assistant for a PDF question-answering system.

The user may ask questions in english, tamil, tanglish or mixed language.

Understand the user's meaning, not just the exact words.

Handle:
-spelling mistakes
-different sentence structures
-short forms
-subject abbreviations
-related words
-english and tamil mixed questions

If the user uses a short subject name, match it with the complete subject name in the document.

For example:
"English exam" may refer to "English communication"
if the PDF contains that subject.

If the question asks about an exam, consider the subject name, exam name, date, day, time.

Return ONLY the best english search query.
Do not answer the question.
Do not explain anything

User question:
{question}

English search query:
"""
            search_query = llm.invoke(search_prompt)

            if hasattr(search_query, "content"):
                search_query = search_query.content
            else:
                search_query = str(search_query)

            search_query = search_query.strip()

            relevant_documents = retriever.invoke(search_query)

            relevant_context = "\n\n".join(document.page_content for document in relevant_documents)

            sources = set()

            for document in relevant_documents:
                page_number = document.metadata.get("page", 0) + 1
                sources.add(f"Page {page_number}")

            prompt = f"""
You are a precise document question-answering assistant.

Use ONLY information from the retrieved document context.

PDF CONTEXT:
{relevant_context}

USER QUESTION:
{question}

Give the actual answer to the user's question based on the pdf context.

Answer the user's question using the pdf context.

Important rules:
1. DO NOT repeat the user's question as the answer
2. Do not describe what the user asked.
3. Give the actual information found in the pdf.
4. If the user asks what the document contains, summarize the main content of the document.
5. If the user asks to explain something, explain that topic using the pdf.
6. Answer in the same language style as the user's question.
7. If the user writes tanglish, answer in simple tanglish.
8. Do not invent information.
9. If the requested information is not available in the PDF context, say:
"I could not find the answer in the Document"
10. Never repeat the user's question as the answer.

FINAL ANSWER:
"""

            st.write("Generating answer....")

            answer = llm.invoke(prompt)

            if hasattr(answer, "content"):
                answer = answer.content
            else:
                answer = str(answer)

            st.subheader("🤖 Answer")

            st.info(answer.strip())

            if sources:
                st.subheader("📚 Source")

                st.write(f"Document: {uploaded_file.name}")

                st.write("Pages used:")

                for source in sorted(sources):
                    st.write(f"- {source}")
            else:
                st.write("No sources found.")
        else:
            st.warning("Please enter a question.")
