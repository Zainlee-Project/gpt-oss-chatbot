import os
import json
from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple

import gradio as gr
import requests
import numpy as np


MODEL_NAME = os.getenv("MODEL_NAME", "gpt-oss:20b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")


def _http_post(url: str, payload: dict, stream: bool = False, timeout: Tuple[int, int] = (10, 600)):
    """Thin wrapper around requests.post to centralize timeouts and error handling."""
    return requests.post(url, json=payload, stream=stream, timeout=timeout)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("Vectors must have the same shape for cosine similarity")
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def embed_text(text: str) -> Optional[np.ndarray]:
    """Generate a single embedding using Ollama's embeddings API."""
    if not text:
        return None
    url = f"{OLLAMA_URL}/api/embeddings"
    payload = {"model": EMBED_MODEL, "prompt": text}
    try:
        resp = _http_post(url, payload, stream=False, timeout=(10, 120))
        resp.raise_for_status()
        data = resp.json()
        vector = np.array(data.get("embedding", []), dtype=np.float32)
        if vector.size == 0:
            return None
        return _normalize(vector)
    except requests.RequestException:
        return None


def embed_texts(texts: List[str]) -> List[np.ndarray]:
    """Embed a list of texts. Falls back gracefully on failures."""
    vectors: List[np.ndarray] = []
    for text in texts:
        vector = embed_text(text)
        if vector is None:
            vector = np.zeros(1, dtype=np.float32)  # placeholder to keep alignment
        vectors.append(vector)
    return vectors


def read_text_from_file(file_path: str) -> str:
    """Extract text from a supported file type.

    Supported: .txt, .md, .json, .csv, .py, .pdf, .docx
    PDFs and DOCX are optional; if parser is unavailable, return empty string.
    """
    lower = file_path.lower()
    try:
        if lower.endswith((".txt", ".md", ".py", ".csv", ".json")):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        if lower.endswith(".pdf"):
            try:
                from pypdf import PdfReader  # type: ignore
            except Exception:
                return ""
            try:
                reader = PdfReader(file_path)
                pages = [p.extract_text() or "" for p in reader.pages]
                return "\n".join(pages)
            except Exception:
                return ""
        if lower.endswith(".docx"):
            try:
                import docx  # type: ignore
            except Exception:
                return ""
            try:
                d = docx.Document(file_path)
                return "\n".join(p.text for p in d.paragraphs)
            except Exception:
                return ""
    except Exception:
        return ""
    return ""


def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks by characters.

    This is a pragmatic approach that avoids tokenizers while producing
    reasonably sized chunks for RAG.
    """
    text = (text or "").strip()
    if not text:
        return []
    if max_chars <= 0:
        return [text]
    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == length:
            break
        start = max(0, end - overlap)
    return chunks


@dataclass
class IndexedChunk:
    content: str
    metadata: dict
    embedding: np.ndarray


class InMemoryVectorStore:
    """A minimal in-memory vector store with cosine similarity search."""

    def __init__(self) -> None:
        self._chunks: List[IndexedChunk] = []
        self._dim: Optional[int] = None

    @property
    def size(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks.clear()
        self._dim = None

    def add_chunks(self, chunks: List[str], metadata: dict) -> int:
        added = 0
        for idx, chunk in enumerate(chunks):
            vector = embed_text(chunk)
            if vector is None:
                continue
            if self._dim is None:
                self._dim = vector.size
            if vector.size != self._dim:
                # Skip inconsistent embeddings
                continue
            self._chunks.append(
                IndexedChunk(content=chunk, metadata={**metadata, "chunk_index": idx}, embedding=vector)
            )
            added += 1
        return added

    def search(self, query: str, top_k: int = 4) -> List[IndexedChunk]:
        if not self._chunks:
            return []
        query_vec = embed_text(query)
        if query_vec is None or (self._dim is not None and query_vec.size != self._dim):
            return []
        scored: List[Tuple[float, IndexedChunk]] = []
        for ch in self._chunks:
            score = _cosine_similarity(query_vec, ch.embedding)
            scored.append((score, ch))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[: max(1, top_k)]]


def stream_from_ollama(
    messages: List[dict],
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
) -> Generator[str, None, None]:
    
    url = f"{OLLAMA_URL}/api/chat"
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": True,
        "options": {
            "num_predict": max_tokens if max_tokens > 0 else -1,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
        },
    }

    try:
        with _http_post(url, payload, stream=True, timeout=(10, 600)) as response:
            response.raise_for_status()
            full_text = ""
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                try:
                    data = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                if data.get("error"):
                    yield f"[Ollama error] {data['error']}"
                    return

                msg = data.get("message", {})
                content_chunk = msg.get("content")
                if content_chunk:
                    full_text += content_chunk
                    yield full_text

                if data.get("done"):
                    break
    except requests.RequestException as exc:
        yield (
            f"Could not reach Ollama at {OLLAMA_URL}. "
            f"Ensure Ollama is running and the model '{MODEL_NAME}' is pulled.\nError: {exc}"
        )


def build_message_history(
    history_messages: List[dict],
    system_prompt: str,
) -> List[dict]:
    messages: List[dict] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.extend(history_messages)
    return messages


def respond(
    history_messages: List[dict],
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    seed: int,
):
    messages = build_message_history(history_messages, system_prompt)

    for partial in stream_from_ollama(
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
    ):
        yield partial


def make_augmented_messages(
    history_messages: List[dict],
    system_prompt: str,
    user_message: str,
    vector_store: Optional[InMemoryVectorStore],
    use_files: bool,
    top_k: int,
) -> List[dict]:
    """Create messages augmented with retrieved context if enabled."""
    if use_files and vector_store and vector_store.size > 0:
        hits = vector_store.search(user_message, top_k=top_k)
        if hits:
            context_blocks = []
            for i, h in enumerate(hits, start=1):
                source = h.metadata.get("source", "uploaded-file")
                idx = h.metadata.get("chunk_index", 0)
                context_blocks.append(f"[Source: {source} | Chunk: {idx}]\n{h.content}")
            context_text = "\n\n---\n\n".join(context_blocks)
            augmented_user = (
                "You have access to the following context extracted from user-uploaded files. "
                "Use it to answer the question. If the context is insufficient, say you don't know.\n\n"
                f"[Context]\n{context_text}\n\n[Question]\n{user_message}"
            )
            msgs = build_message_history(history_messages[:-1] + [{"role": "user", "content": augmented_user}], system_prompt)
            return msgs
    # Default: no augmentation
    return build_message_history(history_messages, system_prompt)


def ui() -> gr.Blocks:
    default_system = (
        "You are an intelligent, polite, and highly professional AI assistant."
        + "Provide context-aware, accurate, and solution-oriented answers in 2–3 concise sentences."
        + "Maintain a respectful, diplomatic tone, using clear and precise language without unnecessary elaboration."
        + "Always prioritize clarity, relevance, and actionable value in your responses."
    )

    with gr.Blocks(title="GPT-OSS:20B (Ollama)") as demo:
        gr.Markdown(
            f"**Model**: `{MODEL_NAME}`  |  **Backend**: `{OLLAMA_URL}`\n\n"
            "Chat with the locally running Ollama model."
        )

        chat = gr.Chatbot(height=520, type="messages")

        # RAG controls
        with gr.Accordion("File search (RAG)", open=False):
            with gr.Row():
                files = gr.Files(label="Upload files", file_count="multiple", type="filepath")
                use_rag = gr.Checkbox(label="Use uploaded files for answers", value=True)
            with gr.Row():
                top_k = gr.Slider(label="Max context chunks", minimum=1, maximum=8, value=4, step=1)
                clear_vs = gr.Button("Clear indexed data")
            rag_status = gr.Markdown(value="No files indexed yet.")
            vs_state = gr.State(InMemoryVectorStore())

        with gr.Row():
            msg_box = gr.Textbox(
                label="Your message",
                placeholder="Type a message and press Enter...",
            )

        with gr.Accordion("Advanced settings", open=False):
            system_prompt = gr.Textbox(
                label="System prompt",
                value=default_system,
                lines=3,
            )
            with gr.Row():
                temperature = gr.Slider(
                    label="Temperature",
                    minimum=0.0,
                    maximum=1.5,
                    value=0.2,
                    step=0.05,
                )
                top_p = gr.Slider(
                    label="Top P",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.95,
                    step=0.01,
                )
            with gr.Row():
                max_tokens = gr.Slider(
                    label="Max new tokens (0 = unlimited)",
                    minimum=0,
                    maximum=8192,
                    value=512,
                    step=32,
                )
                seed = gr.Number(
                    label="Seed (for reproducibility)", value=0, precision=0
                )

        def on_submit(user_message, chat_history, sp, temp, max_tok, tp, sd, vs_obj, use_files, k):
            chat_history = list(chat_history or [])
            chat_history.append({"role": "user", "content": user_message})
            # Build augmented messages for the model call
            messages = make_augmented_messages(
                history_messages=chat_history,
                system_prompt=sp,
                user_message=user_message,
                vector_store=vs_obj,
                use_files=bool(use_files),
                top_k=int(k or 4),
            )
            stream = stream_from_ollama(
                messages=messages,
                temperature=temp,
                top_p=tp,
                max_tokens=max_tok,
                seed=int(sd),
            )
            chat_history.append({"role": "assistant", "content": ""})

            for partial in stream:
                chat_history[-1]["content"] = partial
                yield chat_history

        def index_files(file_paths, vs_obj: InMemoryVectorStore):
            vs = vs_obj or InMemoryVectorStore()
            paths = [p for p in (file_paths or []) if isinstance(p, str) and os.path.exists(p)]
            if not paths:
                return vs, "No valid files provided."
            total_chunks = 0
            total_added = 0
            for path in paths:
                text = read_text_from_file(path)
                if not text:
                    continue
                chunks = chunk_text(text)
                total_chunks += len(chunks)
                added = vs.add_chunks(chunks, metadata={"source": os.path.basename(path)})
                total_added += added
            status = (
                f"Indexed {len(paths)} file(s). Created {total_chunks} chunk(s); "
                f"{total_added} embedded. Current store size: {vs.size}.\n"
                f"Embed model: `{EMBED_MODEL}`"
            )
            return vs, status

        def clear_index(vs_obj: InMemoryVectorStore):
            vs = vs_obj or InMemoryVectorStore()
            vs.clear()
            return vs, "Index cleared. No files indexed yet."

        msg_box.submit(
            fn=on_submit,
            inputs=[msg_box, chat, system_prompt, temperature, max_tokens, top_p, seed, vs_state, use_rag, top_k],
            outputs=[chat],
        )

        # File events
        files.upload(fn=index_files, inputs=[files, vs_state], outputs=[vs_state, rag_status])
        clear_vs.click(fn=clear_index, inputs=[vs_state], outputs=[vs_state, rag_status])

        gr.ClearButton(components=[chat], value="Clear chat")
        gr.Markdown("Tip: adjust settings in the accordion if responses are too long or too random.")

    return demo


if __name__ == "__main__":
    demo = ui()
    port_str = os.getenv("GRADIO_SERVER_PORT") or os.getenv("PORT")
    port = int(port_str) if port_str and port_str.isdigit() else None
    demo.queue().launch(server_name="0.0.0.0", server_port=port, inbrowser=True)


