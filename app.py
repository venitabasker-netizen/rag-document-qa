import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import tempfile

st.set_page_config(page_title="AI Document Q&A", page_icon="📄", layout="centered")

st.title("📄 AI-Powered Document Question Answering System")

st.write("Upload any PDF and ask questions in English, Tamil, or mixed-language")

# --- Gemini API key comes from Streamlit Secrets (set this in Streamlit Cloud settings) ---
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

uploaded_file = st.file_uploader("Upload your any PDF", type=["pdf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        pdf_path = temp_file.name

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    chunks = text_splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY,
    )

    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)

    retriever = vector_store.as_retriever(search_kwargs={"k": 8})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GEMINI_API_KEY,
    )

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
