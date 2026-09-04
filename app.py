
<!DOCTYPE html>
<html lang="en">
<head>
    <base target="_self">
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DocQA — Document Question Answering</title>
    <meta name="description" content="A RAG-based Document QA system using FastAPI, Streamlit, ChromaDB, and OpenAI. Upload PDFs, ask questions, get grounded answers with citations.">
    <script src="https://cdn.tailwindcss.com">
    </script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400;14..32,500;14..32,600;14..32,700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #4f46e5;
            --primary-light: #818cf8;
            --primary-dark: #3730a3;
            --surface: #0f172a;
            --surface-alt: #1e293b;
            --surface-card: #1e293b;
            --text: #f1f5f9;
            --text-muted: #94a3b8;
            --border: #334155;
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--surface);
            color: var(--text);
            min-height: 100vh;
        }

        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: var(--surface);
        }
        ::-webkit-scrollbar-thumb {
            background: var(--border);
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: var(--text-muted);
        }

        /* Code blocks */
        pre {
            background: #0b1120 !important;
            border: 1px solid var(--border);
            border-radius: 0.75rem;
            padding: 1.25rem;
            overflow-x: auto;
            font-size: 0.8rem;
            line-height: 1.6;
            color: #e2e8f0;
        }

        code {
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
        }

        .code-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.5rem 1rem;
            background: #0b1120;
            border: 1px solid var(--border);
            border-bottom: none;
            border-radius: 0.75rem 0.75rem 0 0;
            font-size: 0.75rem;
            color: var(--text-muted);
        }

        .code-header .dot-group {
            display: flex;
            gap: 6px;
        }
        .code-header .dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }
        .dot-red {
            background: #ef4444;
        }
        .dot-yellow {
            background: #f59e0b;
        }
        .dot-green {
            background: #22c55e;
        }

        /* Chat bubbles */
        .chat-msg {
            max-width: 82%;
            padding: 0.875rem 1.125rem;
            border-radius: 1.125rem;
            line-height: 1.6;
            font-size: 0.925rem;
            animation: fadeIn 0.3s ease;
        }

        .chat-msg.user {
            background: var(--primary);
            color: #fff;
            border-bottom-right-radius: 0.25rem;
            align-self: flex-end;
        }

        .chat-msg.assistant {
            background: var(--surface-alt);
            color: var(--text);
            border: 1px solid var(--border);
            border-bottom-left-radius: 0.25rem;
            align-self: flex-start;
        }

        .chat-msg .citation {
            display: inline-block;
            font-size: 0.7rem;
            background: rgba(79, 70, 229, 0.2);
            color: var(--primary-light);
            padding: 0.15rem 0.5rem;
            border-radius: 4px;
            margin: 0.1rem 0.15rem;
            border: 1px solid rgba(79, 70, 229, 0.3);
            cursor: default;
        }

        .chat-msg .citation:hover {
            background: rgba(79, 70, 229, 0.35);
        }

        .chat-msg .not-found {
            color: var(--warning);
            font-weight: 500;
        }

        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(8px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        /* Pulse dot for streaming simulation */
        .pulse-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            background: var(--primary-light);
            border-radius: 50%;
            animation: pulse 1.2s ease-in-out infinite;
        }
        @keyframes pulse {
            0%,
            100% {
                opacity: 1;
                transform: scale(1);
            }
            50% {
                opacity: 0.4;
                transform: scale(0.75);
            }
        }

        /* Tab styling */
        .tab-btn {
            padding: 0.625rem 1.25rem;
            font-size: 0.85rem;
            font-weight: 500;
            border: none;
            background: transparent;
            color: var(--text-muted);
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
        }
        .tab-btn:hover {
            color: var(--text);
        }
        .tab-btn.active {
            color: var(--primary-light);
            border-bottom-color: var(--primary-light);
        }

        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }

        /* File upload zone */
        .upload-zone {
            border: 2px dashed var(--border);
            border-radius: 1rem;
            padding: 2.5rem 1.5rem;
            text-align: center;
            transition: all 0.3s;
            cursor: pointer;
        }
        .upload-zone:hover,
        .upload-zone.dragover {
            border-color: var(--primary);
            background: rgba(79, 70, 229, 0.05);
        }
        .upload-zone .icon {
            font-size: 2.5rem;
            margin-bottom: 0.75rem;
            color: var(--text-muted);
        }
        .upload-zone .label {
            font-weight: 600;
            color: var(--text);
        }
        .upload-zone .hint {
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
        }

        /* Status badge */
        .badge {
            display: inline-flex;
            align-items: center;
            gap: 0.375rem;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 500;
        }
        .badge-success {
            background: rgba(34, 197, 94, 0.15);
            color: var(--success);
        }
        .badge-warning {
            background: rgba(245, 158, 11, 0.15);
            color: var(--warning);
        }
        .badge-info {
            background: rgba(79, 70, 229, 0.15);
            color: var(--primary-light);
        }

        /* Skeleton loading */
        .skeleton {
            background: linear-gradient(90deg, var(--surface-alt) 25%, #2d3a4e 50%, var(--surface-alt) 75%);
            background-size: 200% 100%;
            animation: shimmer 1.5s infinite;
            border-radius: 0.5rem;
        }
        @keyframes shimmer {
            0% {
                background-position: 200% 0;
            }
            100% {
                background-position: -200% 0;
            }
        }

        /* Responsive tweaks */
        @media (max-width: 640px) {
            .chat-msg {
                max-width: 92%;
                font-size: 0.85rem;
            }
            .upload-zone {
                padding: 1.5rem 1rem;
            }
        }
    </style>
</head>
<body>

    <header class="border-b border-[#334155] bg-[#0f172a]/80 backdrop-blur-md sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="flex items-center justify-between h-16">
                <div class="flex items-center gap-3">
                    <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white font-bold text-sm">DQ</div>
                    <div>
                        <h1 class="text-lg font-semibold text-white leading-tight">DocQA</h1>
                        <span class="text-xs text-slate-400">RAG · FastAPI · Streamlit</span>
                    </div>
                </div>
                <div class="flex items-center gap-3">
                    <span class="badge badge-success hidden sm:inline-flex">
                        <span class="w-1.5 h-1.5 rounded-full bg-green-400"></span>
                        All Systems Ready
                    </span>
                    <button id="resetBtn" class="text-sm text-slate-400 hover:text-white transition px-3 py-1.5 rounded-lg hover:bg-slate-800/50 border border-transparent hover:border-slate-700">
                        ⟲ Reset Demo
                    </button>
                </div>
            </div>
        </div>
    </header>

    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <!-- Tab Navigation -->
        <div class="flex gap-1 border-b border-[#334155] mb-8 overflow-x-auto">
            <button class="tab-btn active" data-tab="demo">💬 Live Demo</button>
            <button class="tab-btn" data-tab="code">📁 Full Source Code</button>
            <button class="tab-btn" data-tab="docs">📖 Architecture & Docs</button>
        </div>

        <!-- ===== TAB 1: DEMO ===== -->
        <div id="tab-demo" class="tab-content active">
            <div class="grid lg:grid-cols-5 gap-6">
                <!-- Left sidebar — Upload & Settings -->
                <div class="lg:col-span-2 space-y-5">
                    <!-- Upload Card -->
                    <div class="bg-[#1e293b] rounded-2xl border border-[#334155] p-5">
                        <h2 class="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-4">📄 Document Upload</h2>
                        <div id="uploadZone" class="upload-zone">
                            <div class="icon">📂</div>
                            <div class="label">Drop PDF here or click to browse</div>
                            <div class="hint">Supports .pdf up to 25 MB</div>
                            <input type="file" id="fileInput" accept=".pdf" class="hidden">
                        </div>
                        <div id="uploadStatus" class="mt-3 hidden">
                            <div class="flex items-center justify-between text-sm">
                                <span class="text-slate-300">📎 <span id="fileName">report.pdf</span></span>
                                <span id="fileBadge" class="badge badge-info">Processing…</span>
                            </div>
                            <div class="mt-2 w-full bg-slate-700/50 rounded-full h-1.5 overflow-hidden">
                                <div id="progressBar" class="h-full bg-indigo-500 rounded-full transition-all duration-700" style="width:0%"></div>
                            </div>
                            <div id="chunkInfo" class="mt-2 text-xs text-slate-500 hidden">
                                Extracted <span id="chunkCount">0</span> chunks · <span id="embStatus">Embedding…</span>
                            </div>
                        </div>
                    </div>

                    <!-- Status Panel -->
                    <div class="bg-[#1e293b] rounded-2xl border border-[#334155] p-5">
                        <h2 class="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-3">⚙️ System Status</h2>
                        <div class="space-y-2.5 text-sm">
                            <div class="flex items-center justify-between">
                                <span class="text-slate-400">Vector Store</span>
                                <span class="badge badge-success">● ChromaDB Ready</span>
                            </div>
                            <div class="flex items-center justify-between">
                                <span class="text-slate-400">Embedding Model</span>
                                <span class="badge badge-success">● text-embedding-3-small</span>
                            </div>
                            <div class="flex items-center justify-between">
                                <span class="text-slate-400">LLM</span>
                                <span class="badge badge-success">● GPT-4o-mini</span>
                            </div>
                            <div class="flex items-center justify-between">
                                <span class="text-slate-400">Documents</span>
                                <span id="docCount" class="badge badge-info">1 loaded</span>
                            </div>
                        </div>
                    </div>

                    <!-- Sample Questions -->
                    <div class="bg-[#1e293b] rounded-2xl border border-[#334155] p-5">
                        <h2 class="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-3">💡 Try Asking</h2>
                        <div id="sampleQuestions" class="space-y-1.5">
                            <button class="sample-q w-full text-left text-sm text-slate-400 hover:text-white hover:bg-slate-800/50 px-3 py-2 rounded-lg transition border border-transparent hover:border-slate-700">
                                "What is the main topic of this document?"
                            </button>
                            <button class="sample-q w-full text-left text-sm text-slate-400 hover:text-white hover:bg-slate-800/50 px-3 py-2 rounded-lg transition border border-transparent hover:border-slate-700">
                                "Summarize the key findings."
                            </button>
                            <button class="sample-q w-full text-left text-sm text-slate-400 hover:text-white hover:bg-slate-800/50 px-3 py-2 rounded-lg transition border border-transparent hover:border-slate-700">
                                "What does the document say about AI safety?"
                            </button>
                        </div>
                    </div>
                </div>

                <!-- Right — Chat Interface -->
                <div class="lg:col-span-3">
                    <div class="bg-[#1e293b] rounded-2xl border border-[#334155] flex flex-col h-[600px] lg:h-[700px]">
                        <!-- Chat header -->
                        <div class="px-5 py-3.5 border-b border-[#334155] flex items-center justify-between">
                            <div class="flex items-center gap-2.5">
                                <div class="w-7 h-7 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white text-xs font-bold">AI</div>
                                <div>
                                    <span class="text-sm font-medium text-white">RAG Assistant</span>
                                    <span class="text-xs text-slate-500 ml-2">grounded · cite-sourced</span>
                                </div>
                            </div>
                            <span id="chatStatus" class="text-xs text-slate-500 flex items-center gap-1.5">
                                <span class="w-1.5 h-1.5 rounded-full bg-green-400"></span>
                                Ready
                            </span>
                        </div>

                        <!-- Messages -->
                        <div id="chatMessages" class="flex-1 overflow-y-auto p-5 space-y-4">
                            <div class="chat-msg assistant">
                                👋 Hello! I'm your document assistant. Upload a PDF above, then ask me anything about its contents. I'll answer with <strong>citations</strong> from the source.
                            </div>
                            <div class="chat-msg assistant">
                                <span class="text-xs text-slate-500">📎 To get started, click the upload zone and select a PDF document.</span>
                            </div>
                        </div>

                        <!-- Input -->
                        <div class="px-5 py-3.5 border-t border-[#334155]">
                            <form id="chatForm" class="flex gap-2">
                                <input
                                type="text"
                                id="chatInput"
                                placeholder="Ask a question about your document…"
                                autocomplete="off"
                                class="flex-1 bg-slate-800/60 border border-slate-700 rounded-xl px-4 py-2.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 transition"
                                >
                                <button type="submit" id="sendBtn" class="bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-700 disabled:cursor-not-allowed text-white rounded-xl px-4 py-2.5 text-sm font-medium transition flex items-center gap-1.5">
                                    <span>Send</span>
                                    <span>➤</span>
                                </button>
                            </form>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- ===== TAB 2: CODE ===== -->
        <div id="tab-code" class="tab-content">
            <div class="space-y-8">
                <!-- File explorer style tabs -->
                <div class="flex flex-wrap gap-1 border-b border-[#334155] pb-2">
                    <button class="code-tab-btn text-sm px-3 py-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800/50 transition active" data-code="backend">🐍 backend.py</button>
                    <button class="code-tab-btn text-sm px-3 py-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800/50 transition" data-code="streamlit">🖥️ streamlit_app.py</button>
                    <button class="code-tab-btn text-sm px-3 py-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800/50 transition" data-code="rag">🧠 rag_engine.py</button>
                    <button class="code-tab-btn text-sm px-3 py-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800/50 transition" data-code="config">⚙️ config.py</button>
                </div>

                <!-- Backend -->
                <div class="code-tab-content" data-code="backend">
                    <div class="code-header">
                        <span class="dot-group">
                            <span class="dot dot-red"></span>
                            <span class="dot dot-yellow"></span>
                            <span class="dot dot-green"></span>
                        </span>
                        <span class="text-slate-500">fastapi — main.py</span>
                        <span class="text-slate-600 text-xs">FastAPI · ChromaDB · OpenAI</span>
                    </div>
                    <pre><code># ─── main.py — FastAPI Backend ─────────────────────────────────────────────
        import os
        import uuid
        from pathlib import Path
        from typing import List, Optional

        from fastapi import FastAPI, UploadFile, File, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
        import chromadb
        from chromadb.config import Settings
        from openai import OpenAI
        import PyPDF2
        import tiktoken

        from config import settings
        from rag_engine import RAGEngine

        app = FastAPI(title="DocQA API", version="1.0.0")
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # ── State ──────────────────────────────────────────────────────────────
        vectorstore = chromadb.PersistentClient(
            path=settings.CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        collection = vectorstore.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            embedding_function=None,  # We'll handle embeddings manually
        )
        openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        rag = RAGEngine(openai_client, collection, settings)

        # ── Schemas ────────────────────────────────────────────────────────────
        class AskRequest(BaseModel):
            question: str
            top_k: int = 5

        class AskResponse(BaseModel):
            answer: str
            citations: List[dict]
            sources: List[str]

        class UploadResponse(BaseModel):
            filename: str
            chunk_count: int
            doc_id: str
            message: str

        # ── Helpers ────────────────────────────────────────────────────────────
        def extract_text_from_pdf(path: Path) -> str:
            text = ""
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""
            return text

        def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
            tokens = tiktoken.get_encoding("cl100k_base").encode(text)
            chunks = []
            i = 0
            while i < len(tokens):
                chunk_tokens = tokens[i : i + chunk_size]
                chunks.append(tiktoken.get_encoding("cl100k_base").decode(chunk_tokens))
                i += chunk_size - overlap
            return chunks

        # ── Endpoints ──────────────────────────────────────────────────────────
        @app.post("/upload", response_model=UploadResponse)
        async def upload_pdf(file: UploadFile = File(...)):
            if not file.filename.endswith(".pdf"):
                raise HTTPException(400, "Only PDF files are supported.")

            doc_id = str(uuid.uuid4())
            save_path = Path(settings.UPLOAD_DIR) / f"{doc_id}_{file.filename}"
            save_path.parent.mkdir(parents=True, exist_ok=True)

            content = await file.read()
            save_path.write_bytes(content)

            # Extract & chunk
            raw_text = extract_text_from_pdf(save_path)
            chunks = chunk_text(raw_text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)

            # Embed & store
            embeddings = []
            metadatas = []
            ids = []
            for idx, chunk in enumerate(chunks):
                resp = openai_client.embeddings.create(
                    model=settings.EMBED_MODEL,
                    input=chunk,
                )
                embeddings.append(resp.data[0].embedding)
                metadatas.append({
                    "doc_id": doc_id,
                    "filename": file.filename,
                    "page_number": idx // 3 + 1,  # approximate page
                    "chunk_index": idx,
                })
                ids.append(f"{doc_id}_chunk_{idx}")

            collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
                ids=ids,
            )

            return UploadResponse(
                filename=file.filename,
                chunk_count=len(chunks),
                doc_id=doc_id,
                message="Document processed and indexed successfully.",
            )

        @app.post("/ask", response_model=AskResponse)
        async def ask_question(req: AskRequest):
            if not req.question.strip():
                raise HTTPException(400, "Question cannot be empty.")

            # Search
            query_emb = openai_client.embeddings.create(
                model=settings.EMBED_MODEL,
                input=req.question,
            ).data[0].embedding

            results = collection.query(
                query_embeddings=[query_emb],
                n_results=req.top_k,
            )

            if not results["documents"] or not results["documents"][0]:
                return AskResponse(
                    answer="Information not found in the document.",
                    citations=[],
                    sources=[],
                )

            # Build context
            context_parts = []
            citations = []
            seen_sources = set()
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                source = f"{meta['filename']} (p. {meta['page_number']})"
                context_parts.append(f"[Source: {source}]\n{doc}")
                citations.append({
                    "filename": meta["filename"],
                    "page": meta["page_number"],
                    "relevance_score": round(1 - dist, 3),
                })
                seen_sources.add(source)

            context = "\n\n".join(context_parts)

            # Generate answer
            system_prompt = """You are a precise document QA assistant. Answer the user's question
        using ONLY the provided context. If the context does not contain the answer,
        respond exactly: "Information not found in the document."
        Always cite your sources inline using [Source: filename (p. X)]."""

            response = openai_client.chat.completions.create(
                model=settings.LLM_MODEL,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {req.question}"},
                ],
            )

            answer = response.choices[0].message.content

            return AskResponse(
                answer=answer,
                citations=citations,
                sources=list(seen_sources),
            )

        @app.get("/health")
        async def health():
            return {"status": "ok", "collection_size": collection.count()}
        </code></pre>
                </div>

                <!-- Streamlit -->
                <div class="code-tab-content hidden" data-code="streamlit">
                    <div class="code-header">
                        <span class="dot-group">
                            <span class="dot dot-red"></span>
                            <span class="dot dot-yellow"></span>
                            <span class="dot dot-green"></span>
                        </span>
                        <span class="text-slate-500">streamlit_app.py</span>
                        <span class="text-slate-600 text-xs">Streamlit · Chat UI · File Upload</span>
                    </div>
                    <pre><code># ─── streamlit_app.py — Chat Frontend ─────────────────────────────────────
        import os
        import requests
        import streamlit as st
        from pathlib import Path

        # ── Config ─────────────────────────────────────────────────────────────
        API_BASE = os.getenv("API_BASE", "http://localhost:8000")
        st.set_page_config(
            page_title="DocQA · RAG Assistant",
            page_icon="📄",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # ── Session State ──────────────────────────────────────────────────────
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "doc_id" not in st.session_state:
            st.session_state.doc_id = None
        if "filename" not in st.session_state:
            st.session_state.filename = None

        # ── Sidebar ────────────────────────────────────────────────────────────
        with st.sidebar:
            st.title("📄 DocQA")
            st.caption("RAG-powered document assistant")

            st.divider()
            st.subheader("1. Upload a PDF")
            uploaded_file = st.file_uploader(
                "Choose a PDF file",
                type="pdf",
                label_visibility="collapsed",
            )

            if uploaded_file and st.button("📤 Process Document"):
                with st.spinner("Extracting text, chunking, and embedding…"):
                    files = {"file": (uploaded_file.name, uploaded_file.read(), "application/pdf")}
                    resp = requests.post(f"{API_BASE}/upload", files=files)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state.doc_id = data["doc_id"]
                        st.session_state.filename = data["filename"]
                        st.success(f"✅ {data['chunk_count']} chunks indexed")
                    else:
                        st.error(f"Upload failed: {resp.text}")

            if st.session_state.filename:
                st.info(f"📎 Active: **{st.session_state.filename}**")
            else:
                st.warning("No document loaded. Please upload a PDF.")

            st.divider()
            st.subheader("2. Ask Questions")
            st.caption("Questions are answered using only the document context.")

            if st.button("🧹 Clear Conversation"):
                st.session_state.messages = []
                st.rerun()

        # ── Chat Interface ─────────────────────────────────────────────────────
        st.subheader("💬 Chat with your Document")

        # Display messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and "citations" in msg:
                    with st.expander("📚 View Sources"):
                        for c in msg["citations"]:
                            st.caption(f"📄 {c['filename']} — p. {c['page']} (score: {c['relevance_score']})")

        # Chat input
        if prompt := st.chat_input("Ask something about your document…"):
            if not st.session_state.doc_id:
                st.warning("Please upload a document first.")
            else:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Searching document & generating answer…"):
                        try:
                            resp = requests.post(
                                f"{API_BASE}/ask",
                                json={"question": prompt, "top_k": 5},
                            )
                            if resp.status_code == 200:
                                data = resp.json()
                                answer = data["answer"]
                                citations = data["citations"]
                                sources = data["sources"]

                                st.markdown(answer)
                                if sources:
                                    with st.expander("📚 Sources"):
                                        for s in sources:
                                            st.caption(f"📄 {s}")

                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": answer,
                                    "citations": citations,
                                    "sources": sources,
                                })
                            else:
                                st.error(f"API error: {resp.text}")
                        except Exception as e:
                            st.error(f"Connection error: {e}")
        </code></pre>
                </div>

                <!-- RAG Engine -->
                <div class="code-tab-content hidden" data-code="rag">
                    <div class="code-header">
                        <span class="dot-group">
                            <span class="dot dot-red"></span>
                            <span class="dot dot-yellow"></span>
                            <span class="dot dot-green"></span>
                        </span>
                        <span class="text-slate-500">rag_engine.py</span>
                        <span class="text-slate-600 text-xs">Retrieval-Augmented Generation Core</span>
                    </div>
                    <pre><code># ─── rag_engine.py — Retrieval & Generation ───────────────────────────────
        from typing import List, Tuple, Optional
        from dataclasses import dataclass

        from openai import OpenAI
        import chromadb
        import numpy as np

        @dataclass
        class ChunkResult:
            text: str
            filename: str
            page: int
            score: float

        class RAGEngine:
            """Core RAG logic: embed query → retrieve → build context → generate."""

            def __init__(
                self,
                openai_client: OpenAI,
                collection: chromadb.Collection,
                settings,
            ):
                self.client = openai_client
                self.collection = collection
                self.settings = settings

            def embed(self, text: str) -> List[float]:
                """Get embedding vector for text."""
                resp = self.client.embeddings.create(
                    model=self.settings.EMBED_MODEL,
                    input=text,
                )
                return resp.data[0].embedding

            def retrieve(
                self, query: str, top_k: int = 5
            ) -> Tuple[List[ChunkResult], List[str]]:
                """Retrieve relevant chunks from vector store."""
                query_vec = self.embed(query)
                results = self.collection.query(
                    query_embeddings=[query_vec],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"],
                )

                if not results["documents"] or not results["documents"][0]:
                    return [], []

                chunks = []
                sources = set()
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):
                    chunks.append(ChunkResult(
                        text=doc,
                        filename=meta["filename"],
                        page=meta["page_number"],
                        score=round(1 - dist, 4),
                    ))
                    sources.add(f"{meta['filename']} (p. {meta['page_number']})")

                return chunks, list(sources)

            def build_context(self, chunks: List[ChunkResult]) -> str:
                """Build a grounded context string from retrieved chunks."""
                parts = []
                for c in chunks:
                    parts.append(f"[Source: {c.filename} (p. {c.page})]\n{c.text}")
                return "\n\n".join(parts)

            def generate(self, question: str, context: str) -> str:
                """Generate an answer strictly grounded in the context."""
                system_prompt = (
                    "You are a precise document QA assistant. Answer the user's question "
                    "using ONLY the provided context. If the context does not contain the answer, "
                    "respond exactly: 'Information not found in the document.'\n"
                    "Always cite your sources inline using [Source: filename (p. X)]."
                )
                user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

                response = self.client.chat.completions.create(
                    model=self.settings.LLM_MODEL,
                    temperature=0.1,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                return response.choices[0].message.content

            def answer(self, question: str, top_k: int = 5) -> dict:
                """End-to-end: retrieve → build context → generate."""
                chunks, sources = self.retrieve(question, top_k)
                if not chunks:
                    return {
                        "answer": "Information not found in the document.",
                        "citations": [],
                        "sources": [],
                    }

                context = self.build_context(chunks)
                answer_text = self.generate(question, context)

                citations = [
                    {"filename": c.filename, "page": c.page, "relevance_score": c.score}
                    for c in chunks
                ]

                return {
                    "answer": answer_text,
                    "citations": citations,
                    "sources": sources,
                }
        </code></pre>
                </div>

                <!-- Config -->
                <div class="code-tab-content hidden" data-code="config">
                    <div class="code-header">
                        <span class="dot-group">
                            <span class="dot dot-red"></span>
                            <span class="dot dot-yellow"></span>
                            <span class="dot dot-green"></span>
                        </span>
                        <span class="text-slate-500">config.py</span>
                        <span class="text-slate-600 text-xs">Pydantic Settings · Environment</span>
                    </div>
                    <pre><code># ─── config.py — Application Settings ─────────────────────────────────────
        from pydantic_settings import BaseSettings
        from typing import Optional


        class Settings(BaseSettings):
            # ── OpenAI ─────────────────────────────────────────────────────────
            OPENAI_API_KEY: str
            EMBED_MODEL: str = "text-embedding-3-small"
            LLM_MODEL: str = "gpt-4o-mini"

            # ── Vector Store ────────────────────────────────────────────────────
            CHROMA_PATH: str = "./chroma_db"
            COLLECTION_NAME: str = "documents"

            # ── Chunking ───────────────────────────────────────────────────────
            CHUNK_SIZE: int = 512
            CHUNK_OVERLAP: int = 64

            # ── Paths ──────────────────────────────────────────────────────────
            UPLOAD_DIR: str = "./uploads"

            # ── Server ──────────────────────────────────────────────────────────
            HOST: str = "0.0.0.0"
            PORT: int = 8000

            class Config:
                env_file = ".env"
                env_file_encoding = "utf-8"


        settings = Settings()  # type: ignore[call-arg]
        </code></pre>
                </div>

                <!-- Install & Run instructions -->
                <div class="bg-[#1e293b] rounded-2xl border border-[#334155] p-6">
                    <h3 class="text-base font-semibold text-white mb-3">🚀 Quick Start</h3>
                    <div class="grid sm:grid-cols-2 gap-4 text-sm">
                        <div class="bg-slate-800/50 rounded-xl p-4">
                            <span class="text-indigo-400 font-medium">1. Install</span>
                            <pre class="mt-2 text-xs bg-slate-900/50 p-3 rounded-lg">pip install fastapi uvicorn streamlit \
        chromadb openai PyPDF2 tiktoken \
        python-multipart pydantic-settings</pre>
                        </div>
                        <div class="bg-slate-800/50 rounded-xl p-4">
                            <span class="text-indigo-400 font-medium">2. Environment</span>
                            <pre class="mt-2 text-xs bg-slate-900/50 p-3 rounded-lg"># .env file
        OPENAI_API_KEY=sk-...
        EMBED_MODEL=text-embedding-3-small
        LLM_MODEL=gpt-4o-mini</pre>
                        </div>
                        <div class="bg-slate-800/50 rounded-xl p-4">
                            <span class="text-indigo-400 font-medium">3. Run Backend</span>
                            <pre class="mt-2 text-xs bg-slate-900/50 p-3 rounded-lg">uvicorn main:app --reload --port 8000</pre>
                        </div>
                        <div class="bg-slate-800/50 rounded-xl p-4">
                            <span class="text-indigo-400 font-medium">4. Run UI</span>
                            <pre class="mt-2 text-xs bg-slate-900/50 p-3 rounded-lg">streamlit run streamlit_app.py</pre>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- ===== TAB 3: DOCS ===== -->
        <div id="tab-docs" class="tab-content">
            <div class="max-w-4xl space-y-8">
                <!-- Architecture -->
                <div class="bg-[#1e293b] rounded-2xl border border-[#334155] p-6">
                    <h2 class="text-lg font-semibold text-white mb-4">🏗️ System Architecture</h2>
                    <div class="grid md:grid-cols-4 gap-4 text-center text-sm">
                        <div class="bg-slate-800/60 rounded-xl p-4 border border-slate-700">
                            <div class="text-2xl mb-2">📄</div>
                            <div class="font-medium text-white">User Upload</div>
                            <div class="text-slate-500 text-xs mt-1">PDF → text extraction</div>
                        </div>
                        <div class="bg-slate-800/60 rounded-xl p-4 border border-indigo-500/30">
                            <div class="text-2xl mb-2">✂️</div>
                            <div class="font-medium text-white">Chunking</div>
                            <div class="text-slate-500 text-xs mt-1">512 tokens · 64 overlap</div>
                        </div>
                        <div class="bg-slate-800/60 rounded-xl p-4 border border-indigo-500/30">
                            <div class="text-2xl mb-2">🧠</div>
                            <div class="font-medium text-white">Embedding</div>
                            <div class="text-slate-500 text-xs mt-1">text-embedding-3-small</div>
                        </div>
                        <div class="bg-slate-800/60 rounded-xl p-4 border border-slate-700">
                            <div class="text-2xl mb-2">🗃️</div>
                            <div class="font-medium text-white">ChromaDB</div>
                            <div class="text-slate-500 text-xs mt-1">persistent vector store</div>
                        </div>
                    </div>
                    <div class="flex items-center justify-center my-3">
                        <span class="text-slate-600 text-sm">⬇️</span>
                    </div>
                    <div class="grid md:grid-cols-3 gap-4 text-center text-sm">
                        <div class="bg-slate-800/60 rounded-xl p-4 border border-indigo-500/30">
                            <div class="text-2xl mb-2">🔍</div>
                            <div class="font-medium text-white">Retrieval</div>
                            <div class="text-slate-500 text-xs mt-1">cosine similarity · top-k</div>
                        </div>
                        <div class="bg-slate-800/60 rounded-xl p-4 border border-indigo-500/30">
                            <div class="text-2xl mb-2">🤖</div>
                            <div class="font-medium text-white">Generation</div>
                            <div class="text-slate-500 text-xs mt-1">GPT-4o-mini · 0.1 temp</div>
                        </div>
                        <div class="bg-slate-800/60 rounded-xl p-4 border border-slate-700">
                            <div class="text-2xl mb-2">💬</div>
                            <div class="font-medium text-white">Answer + Citations</div>
                            <div class="text-slate-500 text-xs mt-1">grounded · sourced</div>
                        </div>
                    </div>
                </div>

                <!-- Guardrails -->
                <div class="bg-[#1e293b] rounded-2xl border border-[#334155] p-6">
                    <h2 class="text-lg font-semibold text-white mb-4">🛡️ Guardrails & Citations</h2>
                    <div class="space-y-3 text-sm text-slate-300">
                        <div class="flex items-start gap-3">
                            <span class="text-green-400 mt-0.5">✓</span>
                            <div><strong class="text-white">Citation Enforcement:</strong> Every answer includes inline source references with filename and page number. The LLM is instructed to cite sources for every claim.</div>
                        </div>
                        <div class="flex items-start gap-3">
                            <span class="text-green-400 mt-0.5">✓</span>
                            <div><strong class="text-white">Grounded Answers:</strong> The system prompt explicitly restricts the LLM to use only the provided context. If the answer isn't in the context, it must respond with "Information not found in the document."</div>
                        </div>
                        <div class="flex items-start gap-3">
                            <span class="text-green-400 mt-0.5">✓</span>
                            <div><strong class="text-white">Low Temperature:</strong> Temperature is set to 0.1 to minimize hallucination and ensure consistent, conservative outputs.</div>
                        </div>
                        <div class="flex items-start gap-3">
                            <span class="text-green-400 mt-0.5">✓</span>
                            <div><strong class="text-white">Relevance Scoring:</strong> Each retrieved chunk includes a relevance score (1 − cosine distance) so users can assess confidence.</div>
                        </div>
                        <div class="flex items-start gap-3">
                            <span class="text-green-400 mt-0.5">✓</span>
                            <div><strong class="text-white">Source Transparency:</strong> The UI displays all sources used to generate each answer in an expandable section.</div>
                        </div>
                    </div>
                </div>

                <!-- API Endpoints -->
                <div class="bg-[#1e293b] rounded-2xl border border-[#334155] p-6">
                    <h2 class="text-lg font-semibold text-white mb-4">📡 API Reference</h2>
                    <div class="space-y-4 text-sm">
                        <div class="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
                            <div class="flex items-center gap-2 mb-1">
                                <span class="badge badge-success">POST</span>
                                <code class="text-indigo-300 font-medium">/upload</code>
                            </div>
                            <p class="text-slate-400">Upload a PDF file. Returns <code>doc_id</code>, <code>chunk_count</code>, and a success message.</p>
                        </div>
                        <div class="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
                            <div class="flex items-center gap-2 mb-1">
                                <span class="badge badge-info">POST</span>
                                <code class="text-indigo-300 font-medium">/ask</code>
                            </div>
                            <p class="text-slate-400">Send a question. Returns <code>answer</code>, <code>citations</code> (filename, page, score), and <code>sources</code> list.</p>
                        </div>
                        <div class="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
                            <div class="flex items-center gap-2 mb-1">
                                <span class="badge badge-info">GET</span>
                                <code class="text-indigo-300 font-medium">/health</code>
                            </div>
                            <p class="text-slate-400">Health check. Returns status and collection size.</p>
                        </div>
                    </div>
                </div>

                <!-- Folder Structure -->
                <div class="bg-[#1e293b] rounded-2xl border border-[#334155] p-6">
                    <h2 class="text-lg font-semibold text-white mb-4">📁 Project Structure</h2>
                    <pre class="text-sm bg-slate-900/50 p-4 rounded-xl border border-slate-700/50"><code>docqa/
        ├── main.py              # FastAPI server (endpoints)
        ├── streamlit_app.py     # Streamlit chat UI
        ├── rag_engine.py        # Retrieval & generation logic
        ├── config.py            # Pydantic settings (.env)
        ├── requirements.txt     # Dependencies
        ├── .env                 # API keys (git-ignored)
        ├── chroma_db/           # Persistent vector store (auto-created)
        └── uploads/             # Uploaded PDFs (auto-created)</code></pre>
                </div>
            </div>
        </div>
    </main>

    <footer class="border-t border-[#334155] mt-12 py-6 text-center text-xs text-slate-600">
        <span>DocQA · Built with FastAPI + Streamlit + ChromaDB + OpenAI · RAG Pattern</span>
    </footer>

    <script>
        // ─────────────────────────────────────────────────────────────────────────
        // 1. DATA LAYER
        // ─────────────────────────────────────────────────────────────────────────

        const sampleQuestions = [
            "What is the main topic of this document?",
            "Summarize the key findings.",
            "What does the document say about AI safety?",
            "Who are the authors?",
            "What methodology was used?",
        ];

        // Simulated QA pairs based on the "loaded" document
        const qaData = [{
            q: "what is the main topic of this document?",
            a: "The document is a comprehensive technical report on **Large Language Model Alignment and Safety**, covering reinforcement learning from human feedback (RLHF), red-teaming methodologies, and evaluation benchmarks for AI systems.",
            citations: [
                { filename: "llm_safety_report_2024.pdf", page: 1, score: 0.96 },
                { filename: "llm_safety_report_2024.pdf", page: 3, score: 0.88 }
            ]
        }, {
            q: "summarize the key findings.",
            a: "The key findings are: (1) RLHF significantly improves model truthfulness by 34% compared to supervised fine-tuning alone. (2) Red-teaming revealed 12 critical failure modes in long-tail knowledge scenarios. (3) Current evaluation benchmarks cover only 62% of identified safety-relevant dimensions. (4) Multi-turn dialogue attacks are 2.7× more effective than single-turn attacks at eliciting harmful responses. [Source: llm_safety_report_2024.pdf (p. 1)]",
            citations: [
                { filename: "llm_safety_report_2024.pdf", page: 1, score: 0.94 },
                { filename: "llm_safety_report_2024.pdf", page: 5, score: 0.91 },
                { filename: "llm_safety_report_2024.pdf", page: 8, score: 0.85 }
            ]
        }, {
            q: "what does the document say about ai safety?",
            a: "The document defines AI safety across three pillars: **robustness** (performance under distribution shift), **alignment** (value alignment with human intent), and **control** (human oversight mechanisms). It recommends a layered defense approach combining input/output classifiers, constitutional AI constraints, and human-in-the-loop verification for high-stakes deployments. [Source: llm_safety_report_2024.pdf (p. 3)]",
            citations: [
                { filename: "llm_safety_report_2024.pdf", page: 3, score: 0.93 },
                { filename: "llm_safety_report_2024.pdf", page: 4, score: 0.87 }
            ]
        }, {
            q: "who are the authors?",
            a: "The report was authored by researchers from the Alignment Research Center and Stanford's Center for AI Safety: Dr. Amelia Chen (lead), Prof. David Liang, Dr. Sarah Okediji, and Marcus Rivera. [Source: llm_safety_report_2024.pdf (p. 1)]",
            citations: [
                { filename: "llm_safety_report_2024.pdf", page: 1, score: 0.92 }
            ]
        }, {
            q: "what methodology was used?",
            a: "The study employed a mixed-methods approach: (1) Quantitative evaluation across 14 safety benchmarks using standardized adversarial test suites. (2) Qualitative red-teaming sessions with 28 domain experts. (3) A novel 'stress-test' framework that probes model behavior under iterative multi-turn pressure. All experiments were conducted on three base model families (LLaMA-3, Mistral, and GPT-4-mini) to ensure generalizability. [Source: llm_safety_report_2024.pdf (p. 2)]",
            citations: [
                { filename: "llm_safety_report_2024.pdf", page: 2, score: 0.95 },
                { filename: "llm_safety_report_2024.pdf", page: 6, score: 0.89 }
            ]
        }];

        const defaultResponses = [
            "That's an interesting question! Based on the document, I can provide insights on this topic. Could you be more specific about what aspect you'd like to explore?",
            "The document discusses several related points. Let me highlight the most relevant ones: first, it emphasizes the importance of rigorous evaluation. Second, it provides empirical evidence from multiple benchmark tests. Third, it outlines practical recommendations for deployment. [Source: llm_safety_report_2024.pdf (p. 3)]",
        ];

        // ─────────────────────────────────────────────────────────────────────────
        // 2. STATE
        // ─────────────────────────────────────────────────────────────────────────

        let uploaded = false;
        let chatHistory = [];
        let isProcessing = false;

        // ─────────────────────────────────────────────────────────────────────────
        // 3. RENDER FUNCTIONS
        // ─────────────────────────────────────────────────────────────────────────

        function renderMessage(msg) {
            const role = msg.role;
            const content = msg.content;
            const citations = msg.citations || [];

            let html = '';
            if (role === 'user') {
                html = `<div class="chat-msg user">${escapeHtml(content)}</div>`;
            } else {
                let processed = content;
                // Replace citation markers with styled badges
                processed = processed.replace(
                    /\[Source:\s*([^\]]+)\]/g,
                    '<span class="citation" title="Click to view source">📚 $1</span>'
                );
                // If "not found" in answer, highlight
                if (processed.toLowerCase().includes('information not found')) {
                    processed =
                        `<span class="not-found">⚠️</span> ${processed}`;
                }
                html = `<div class="chat-msg assistant">${processed}</div>`;
                if (citations.length > 0) {
                    html += `<div class="ml-2 mt-1 text-xs text-slate-500 flex flex-wrap gap-1.5">`;
                    const seen = new Set();
                    citations.forEach(c => {
                        const key = `${c.filename}-${c.page}`;
                        if (!seen.has(key)) {
                            seen.add(key);
                            html +=
                                `<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-slate-800/80 border border-slate-700/50 text-slate-400">📄 ${c.filename} <span class="text-slate-600">p.${c.page}</span> <span class="text-indigo-400/70 text-[10px]">${c.score}</span></span>`;
                        }
                    });
                    html += `</div>`;
                }
            }
            return html;
        }

        function renderChat() {
            const container = document.getElementById('chatMessages');
            container.innerHTML = chatHistory.map(m => renderMessage(m)).join('');
            container.scrollTop = container.scrollHeight;
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // ─────────────────────────────────────────────────────────────────────────
        // 4. QA LOGIC
        // ─────────────────────────────────────────────────────────────────────────

        function findAnswer(question) {
            const q = question.toLowerCase().trim();
            for (const item of qaData) {
                if (q.includes(item.q) || item.q.includes(q)) {
                    return item;
                }
            }
            // Fuzzy match by keywords
            const keywords = q.split(/\s+/).filter(w => w.length > 3);
            let best = null;
            let bestScore = 0;
            for (const item of qaData) {
                const words = item.q.split(/\s+/);
                let score = 0;
                for (const kw of keywords) {
                    if (words.some(w => w.includes(kw) || kw.includes(w))) score += 1;
                }
                if (score > bestScore) {
                    bestScore = score;
                    best = item;
                }
            }
            if (best && bestScore >= 1) return best;

            // Check for "not found" patterns
            const notFoundPatterns = ['not found', 'no information', 'doesn\'t mention', 'not discussed', 'not covered'];
            if (notFoundPatterns.some(p => q.includes(p))) {
                return {
                    a: "Information not found in the document.",
                    citations: []
                };
            }

            return null;
        }

        function simulateAnswer(question) {
            return new Promise((resolve) => {
                const delay = 800 + Math.random() * 1200;
                setTimeout(() => {
                    const match = findAnswer(question);
                    if (match) {
                        resolve({
                            answer: match.a,
                            citations: match.citations,
                        });
                    } else {
                        // Fallback with default
                        const fallback = defaultResponses[Math.floor(Math.random() * defaultResponses.length)];
                        resolve({
                            answer: fallback,
                            citations: [
                                { filename: "llm_safety_report_2024.pdf", page: 2, score: 0.72 },
                                { filename: "llm_safety_report_2024.pdf", page: 5, score: 0.68 }
                            ]
                        });
                    }
                }, delay);
            });
        }

        // ─────────────────────────────────────────────────────────────────────────
        // 5. EVENT HANDLING (Delegation)
        // ─────────────────────────────────────────────────────────────────────────

        document.addEventListener('DOMContentLoaded', function() {
            // ── Tab switching ──────────────────────────────────────────────────
            document.querySelectorAll('.tab-btn').forEach(btn => {
                btn.addEventListener('click', function(e) {
                    const tab = this.dataset.tab;
                    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove(
                    'active'));
                    this.classList.add('active');
                    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove(
                        'active'));
                    document.getElementById(`tab-${tab}`).classList.add('active');
                });
            });

            // ── Code tabs ──────────────────────────────────────────────────────
            document.querySelectorAll('.code-tab-btn').forEach(btn => {
                btn.addEventListener('click', function(e) {
                    const code = this.dataset.code;
                    document.querySelectorAll('.code-tab-btn').forEach(b => b.classList.remove(
                        'active'));
                    this.classList.add('active');
                    document.querySelectorAll('.code-tab-content').forEach(t => t.classList.add(
                        'hidden'));
                    document.querySelector(`.code-tab-content[data-code="${code}"]`).classList.remove(
                        'hidden');
                });
            });

            // ── File Upload ────────────────────────────────────────────────────
            const uploadZone = document.getElementById('uploadZone');
            const fileInput = document.getElementById('fileInput');
            const uploadStatus = document.getElementById('uploadStatus');
            const fileName = document.getElementById('fileName');
            const fileBadge = document.getElementById('fileBadge');
            const progressBar = document.getElementById('progressBar');
            const chunkInfo = document.getElementById('chunkInfo');
            const chunkCount = document.getElementById('chunkCount');
            const embStatus = document.getElementById('embStatus');
            const docCount = document.getElementById('docCount');
            const chatStatus = document.getElementById('chatStatus');

            uploadZone.addEventListener('click', () => fileInput.click());

            uploadZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadZone.classList.add('dragover');
            });
            uploadZone.addEventListener('dragleave', () => {
                uploadZone.classList.remove('dragover');
            });
            uploadZone.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadZone.classList.remove('dragover');
                if (e.dataTransfer.files.length) {
                    fileInput.files = e.dataTransfer.files;
                    handleFile(e.dataTransfer.files[0]);
                }
            });

            fileInput.addEventListener('change', function() {
                if (this.files.length) handleFile(this.files[0]);
            });

            function handleFile(file) {
                if (!file.name.toLowerCase().endsWith('.pdf')) {
                    alert('Please upload a PDF file.');
                    return;
                }
                if (file.size > 25 * 1024 * 1024) {
                    alert('File is too large. Maximum size is 25 MB.');
                    return;
                }

                uploaded = true;
                fileName.textContent = file.name;
                uploadStatus.classList.remove('hidden');
                fileBadge.textContent = '⏳ Extracting text…';
                fileBadge.className = 'badge badge-warning';
                progressBar.style.width = '20%';

                // Simulate processing steps
                setTimeout(() => {
                    progressBar.style.width = '45%';
                    fileBadge.textContent = '✂️ Chunking…';
                }, 400);

                setTimeout(() => {
                    progressBar.style.width = '70%';
                    fileBadge.textContent = '🧠 Embedding…';
                }, 900);

                setTimeout(() => {
                    const totalChunks = Math.floor(20 + Math.random() * 30);
                    progressBar.style.width = '100%';
                    fileBadge.textContent = '✅ Ready';
                    fileBadge.className = 'badge badge-success';
                    chunkInfo.classList.remove('hidden');
                    chunkCount.textContent = totalChunks;
                    embStatus.textContent = '✅ 768-dim vectors stored';
                    docCount.textContent = '1 loaded';
                    chatStatus.innerHTML =
                        `<span class="w-1.5 h-1.5 rounded-full bg-green-400"></span> Ready — ${file.name}`;

                    // Add system message
                    chatHistory.push({
                        role: 'assistant',
                        content: `✅ **${file.name}** processed successfully! ${totalChunks} chunks embedded and indexed. You can now ask questions about the document.`,
                        citations: []
                    });
                    renderChat();
                }, 1600);
            }

            // ── Sample Questions ──────────────────────────────────────────────
            document.querySelectorAll('.sample-q').forEach(btn => {
                btn.addEventListener('click', function() {
                    const text = this.textContent.trim().replace(/^"/, '').replace(/"$/, '');
                    document.getElementById('chatInput').value = text;
                    document.getElementById('chatForm').dispatchEvent(new Event('submit'));
                });
            });

            // ── Chat Form ──────────────────────────────────────────────────────
            const chatForm = document.getElementById('chatForm');
            const chatInput = document.getElementById('chatInput');
            const sendBtn = document.getElementById('sendBtn');

            chatForm.addEventListener('submit', async function(e) {
                e.preventDefault();
                const question = chatInput.value.trim();
                if (!question || isProcessing) return;

                // If no doc uploaded, warn but still answer with general knowledge
                if (!uploaded) {
                    // Allow it but note
                }

                // Add user message
                chatHistory.push({ role: 'user', content: question });
                chatInput.value = '';
                sendBtn.disabled = true;
                isProcessing = true;
                chatStatus.innerHTML = `<span class="pulse-dot"></span> Thinking…`;
                renderChat();

                // Simulate typing indicator
                const typingMsg = {
                    role: 'assistant',
                    content: `<span class="pulse-dot"></span> Searching document & generating answer…`,
                    citations: []
                };
                chatHistory.push(typingMsg);
                renderChat();

                // Get answer
                const result = await simulateAnswer(question);

                // Remove typing indicator
                chatHistory.pop();

                // Add real answer
                chatHistory.push({
                    role: 'assistant',
                    content: result.answer,
                    citations: result.citations,
                });

                sendBtn.disabled = false;
                isProcessing = false;
                chatStatus.innerHTML =
                    `<span class="w-1.5 h-1.5 rounded-full bg-green-400"></span> Ready`;
                renderChat();
            });

            // ── Reset Button ──────────────────────────────────────────────────
            document.getElementById('resetBtn').addEventListener('click', function() {
                chatHistory = [{
                    role: 'assistant',
                    content: "👋 Hello! I'm your document assistant. Upload a PDF above, then ask me anything about its contents. I'll answer with **citations** from the source."
                }, {
                    role: 'assistant',
                    content: "📎 To get started, click the upload zone and select a PDF document."
                }];
                uploaded = false;
                document.getElementById('uploadStatus').classList.add('hidden');
                document.getElementById('progressBar').style.width = '0%';
                document.getElementById('chunkInfo').classList.add('hidden');
                document.getElementById('docCount').textContent = '0 loaded';
                document.getElementById('chatStatus').innerHTML =
                    `<span class="w-1.5 h-1.5 rounded-full bg-green-400"></span> Ready`;
                document.getElementById('fileInput').value = '';
                renderChat();
            });

            // ── Initial render ────────────────────────────────────────────────
            renderChat();

            // ── Keyboard shortcut ─────────────────────────────────────────────
            document.addEventListener('keydown', function(e) {
                if (e.key === '/' && !e.target.matches('input, textarea')) {
                    e.preventDefault();
                    chatInput.focus();
                }
            });
        });
    </script>
</body>
</html>
