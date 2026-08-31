import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma 
import tempfile

st.set_page_config(page_title="AI Document Q&A",page_icon="📜",layout="centered")

st.title("📜AI-Powered Document Question Answering System")

st.write("Upload any PDF and ask questions in English,Tamil, or mixed-language")

uploaded_file=st.file_uploader("uploaded your any PDF",type=["pdf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf")as temp_file:temp_file.write(uploaded_file.getvalue())
    pdf_path=temp_file.name

    loader=PyPDFLoader(pdf_path)

    documents=loader.load()

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

    chunks=text_splitter.split_documents(documents)

    embeddings=HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

    vector_store=Chroma.from_documents(documents=chunks,embedding=embeddings)

    retriever=vector_store.as_retriever(search_kwarges={"k":8})

    llm =ChatOpenAI(model="gpt-40-mini")

    st.success("PDF processed successfully!")

    st.write("Uploaded PDF:",uploaded_file.name)

    question=st.text_input("💭Ask a question about the PDF:") 

    if st.button("Ask"):

       if question.strip():

            searh_prompt=f"""
you are a query understanding assistant for a PDF question-answering system.

The user may ask questions in english,tamil,tanglish or mixed language.

understand the user's meaning,not just the exact words.

Handle:
-spelling mistakes
-different sentence structures
-short forms
-subject abbreviations
-related words
-english and tamil mixed questions

if the user uses a short subject names,match it with the complete subject name in the PDF.

for example:
"English exam" may refer to "English communication"
if the PDF contains that subject.

if the question asks about an exam,consider the subject name,exam name,date.day,time,and schedule.

return ONLY the best english search query.
do not answer the question.
do not explain anything
    
User question:
{question}

English search query:
 """

            search_query=llm.invoke(search_prompt)

            if hasattr(search_query,"content"):
                search_query=search_query.content
            else:
                search_query=str(search_query)

            search_query=search_query.strip()
    
            relevant_documents=retriever.invoke(search_query)

            relevant_context="\n\n".join(document.page_content for document in relevant_documents)

            sources=set()

            for document in relevant_documents:

                page_number=document.metadata.get("page",0)+1

                sources.add(f"Page{page_number}")

            prompt=f"""

you are a precise document question-answering assistant.

use ONLY information form the retrieved document context.

PDF CONTEXT:
{relevant_context}

USER QUESTION:
{question}

Give the actual answer to the user's question based on the pdf context.

Asnwer the user's question using the pdf context.

important rules:
1.DO NOT repeat the user's question as the answer
2.do not describe what the user askwe.
3.give the actual information found in the pdf.
4.if the user asks whar the document contains,summarize the main content of the doucment.
5.if the user asks to expplain something,explain that topic using the pdf.
6.answer in the same language style as the user's question.
7.if the user writes tanglish,answer in simple tanglish.
8.do not invent information.
9.if the requested information is not available in the PDF context,say:
I could not find the answer in the Document
10.Never repeat the user' question as the answer.


FIINAL ANSWER:
"""

            st.write("Generating answer....")

            answer=llm.invoke(prompt)

            if hasattr(answer,"content"):
                answer=answer.content
            else:  
                answer=str(answer)

            st.subheader("🤖Answer")

            st.info(answer.strip())

            if sources:

                st.subheader("📚Source")

                st.write(f"Document:{uploaded_file.name}")

                st.write("Pages used:")

                for source in sorted(sources):

                    st.write(f".{source}")

            else:

                st.warning("Please enter a question.")
    





   
