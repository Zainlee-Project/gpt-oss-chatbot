import os
import json
from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime, timezone, timedelta

import gradio as gr
import requests
import numpy as np


MODEL_NAME = os.getenv("MODEL_NAME", "gpt-oss:20b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")


def _http_post(url: str, payload: dict, stream: bool = False, timeout: Tuple[int, int] = (10, 600)):
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


def embed_text(text: str, retries: int = 2) -> Optional[np.ndarray]:
    if not text or not text.strip():
        return None
    
    url = f"{OLLAMA_URL}/api/embeddings"
    payload = {"model": EMBED_MODEL, "prompt": text.strip()}
    
    for attempt in range(retries + 1):
        try:
            resp = _http_post(url, payload, stream=False, timeout=(5, 30))
            resp.raise_for_status()
            data = resp.json()
            
            if "embedding" not in data or not data["embedding"]:
                continue
                
            vector = np.array(data["embedding"], dtype=np.float32)
            if vector.size == 0:
                continue
                
            return _normalize(vector)
            
        except requests.RequestException as e:
            if attempt == retries:
                return None
            time.sleep(1)
    
    return None


def embed_texts(texts: List[str], max_workers: int = 3) -> List[np.ndarray]:
    if not texts:
        return []
    
    vectors: List[Optional[np.ndarray]] = [None] * len(texts)
    
    def embed_with_index(idx_text_pair):
        idx, text = idx_text_pair
        return idx, embed_text(text)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(embed_with_index, (i, text)): i 
            for i, text in enumerate(texts)
        }
        
        for future in as_completed(future_to_idx):
            try:
                idx, vector = future.result()
                vectors[idx] = vector
            except Exception:
                pass
    
    result_vectors: List[np.ndarray] = []
    for vector in vectors:
        if vector is None:
            result_vectors.append(np.zeros(1, dtype=np.float32))
        else:
            result_vectors.append(vector)
    
    return result_vectors


def read_text_from_file(file_path: str) -> str:
    lower = file_path.lower()
    try:
        if lower.endswith((".txt", ".md", ".py", ".csv", ".json")):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        if lower.endswith(".pdf"):
            try:
                from pypdf import PdfReader
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
                import docx
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


def chunk_text(text: str, max_chars: int = 3000, overlap: int = 300) -> List[str]:
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

def get_weather(city: str) -> str:
    latitude = 25.2048
    longitude = 55.2708
    try:
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,weather_code",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
                "timezone": "Asia/Dubai",
                "forecast_days": "3",
            },
            timeout=(5, 10),
        )
        response.raise_for_status()
        payload = response.json()
        current = payload.get("current") or {}
        daily = payload.get("daily") or {}
        
        temperature_c = current.get("temperature_2m")
        feels_like_c = current.get("apparent_temperature")
        humidity = current.get("relative_humidity_2m")
        wind_speed = current.get("wind_speed_10m")
        weather_code = current.get("weather_code")

        def describe(code: Optional[int]) -> str:
            if code is None:
                return "Unknown"
            code_map = {
                0: "Clear", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
                45: "Fog", 48: "Depositing rime fog", 51: "Light drizzle",
                53: "Moderate drizzle", 55: "Dense drizzle", 61: "Slight rain",
                63: "Moderate rain", 65: "Heavy rain", 80: "Rain showers",
                95: "Thunderstorm",
            }
            return code_map.get(int(code), "Unknown")

        current_desc = describe(weather_code)
        current_parts = []
        if temperature_c is not None:
            current_parts.append(f"{temperature_c}°C")
        if feels_like_c is not None:
            current_parts.append(f"feels {feels_like_c}°C")
        if humidity is not None:
            current_parts.append(f"humidity {humidity}%")
        if wind_speed is not None:
            current_parts.append(f"wind {wind_speed} km/h")
        current_metrics = ", ".join(current_parts) if current_parts else "n/a"
        
        result = f"Dubai, UAE: {current_desc}, {current_metrics}"
        
        daily_codes = daily.get("weather_code", [])
        daily_max_temps = daily.get("temperature_2m_max", [])
        daily_min_temps = daily.get("temperature_2m_min", [])
        daily_precipitation = daily.get("precipitation_sum", [])
        
        if len(daily_codes) >= 3 and len(daily_max_temps) >= 3:
            forecast_parts = []
            days = ["Today", "Tomorrow", "Day after"]
            for i in range(min(3, len(daily_codes))):
                day_desc = describe(daily_codes[i] if i < len(daily_codes) else None)
                max_temp = daily_max_temps[i] if i < len(daily_max_temps) else None
                min_temp = daily_min_temps[i] if i < len(daily_min_temps) else None
                precip = daily_precipitation[i] if i < len(daily_precipitation) else None
                
                day_info = f"{days[i]}: {day_desc}"
                if max_temp is not None and min_temp is not None:
                    day_info += f", {min_temp}°C-{max_temp}°C"
                if precip is not None and precip > 0:
                    day_info += f", rain {precip}mm"
                forecast_parts.append(day_info)
            
            if forecast_parts:
                result += ". Forecast: " + "; ".join(forecast_parts)
        
        return result
    except Exception:
        return "Dubai, UAE: Weather data unavailable"


def get_unix_time(city: str) -> str:
    try:
        r = requests.get("https://worldtimeapi.org/api/timezone/Asia/Dubai", timeout=(5, 10))
        r.raise_for_status()
        data = r.json()
        u = data.get("unixtime")
        if isinstance(u, int):
            return str(u)
        if isinstance(u, float):
            return str(int(u))
    except Exception:
        pass
    return str(int(time.time()))


def change_time(city: str, time: str) -> str:
    try:
        ts = float(time)
    except Exception:
        return ""
    if ts > 1_000_000_000_000:
        ts = ts / 1000.0
    key = (city or "").strip().lower()
    if "dubai" in key or "uae" in key or "united arab emirates" in key:
        try:
            from zoneinfo import ZoneInfo
            dt = datetime.fromtimestamp(ts, ZoneInfo("Asia/Dubai"))
            return dt.strftime("%H:%M")
        except Exception:
            dt = datetime.utcfromtimestamp(ts) + timedelta(hours=4)
            return (dt.replace(tzinfo=timezone(timedelta(hours=4))).strftime("%H:%M"))
    try:
        from zoneinfo import ZoneInfo
        dt = datetime.fromtimestamp(ts, ZoneInfo("UTC"))
        return dt.strftime("%H:%M")
    except Exception:
        dt = datetime.utcfromtimestamp(ts).replace(tzinfo=timezone.utc)
        return dt.strftime("%H:%M")



@dataclass
class IndexedChunk:
    content: str
    metadata: dict
    embedding: np.ndarray


class InMemoryVectorStore:
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
        if not chunks:
            return 0
        
        vectors = embed_texts(chunks, max_workers=3)
        
        added = 0
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            if vector is None or vector.size == 1 and vector[0] == 0:
                continue
            if self._dim is None:
                self._dim = vector.size
            if vector.size != self._dim:
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
        
        if query_vec is None:
            query_words = set(query.lower().split())
            fallback_results = []
            for chunk in self._chunks[:top_k * 2]: 
                chunk_words = set(chunk.content.lower().split())
                if query_words & chunk_words:
                    fallback_results.append(chunk)
            if fallback_results:
                return fallback_results[:top_k]
            return []
            
        if self._dim is not None and query_vec.size != self._dim:
            return []
            
        scored: List[Tuple[float, IndexedChunk]] = []
        for ch in self._chunks:
            score = _cosine_similarity(query_vec, ch.embedding)
            scored.append((score, ch))
            
        scored.sort(key=lambda x: x[0], reverse=True)
        top_results = [c for _, c in scored[: max(1, top_k)]]
        return top_results


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
    MAX_CONTEXT_CHARS_PER_CHUNK = 900
    
    if use_files and vector_store and vector_store.size > 0:
        hits = vector_store.search(user_message, top_k=top_k)
        
        if hits:
            context_blocks = []
            for i, h in enumerate(hits, start=1):
                source = h.metadata.get("source", "uploaded-file")
                idx = h.metadata.get("chunk_index", 0)
                trimmed = (h.content or "").strip()
                if len(trimmed) > MAX_CONTEXT_CHARS_PER_CHUNK:
                    trimmed = trimmed[:MAX_CONTEXT_CHARS_PER_CHUNK] + " …"
                context_blocks.append(f"[Source: {source} | Chunk: {idx}]\n{trimmed}")
                
            context_text = "\n\n---\n\n".join(context_blocks)
            augmented_user = (
                "You have access to the following context extracted from user-uploaded files. "
                "Use it to answer the question. If the context is insufficient, say you don't know.\n\n"
                f"[Context]\n{context_text}\n\n[Question]\n{user_message}"
            )
            msgs = build_message_history(history_messages[:-1] + [{"role": "user", "content": augmented_user}], system_prompt)
            return msgs
    return build_message_history(history_messages, system_prompt)


def ui() -> gr.Blocks:
    weather_info = get_weather("Dubai")
    dubai_unix = get_unix_time("Dubai")
    dubai_time = change_time("Dubai", dubai_unix)
    print(f"{weather_info} | Dubai time: {dubai_time}")
    default_system = (
        "You are an intelligent, polite, and highly professional AI assistant with access to real-time data. "
        "Always start responses with a friendly greeting like 'Hi!' "
        "When you receive current information in your knowledge base, use it confidently as live, real-time data. "
        "Provide context-aware, accurate, and solution-oriented answers in 2-3 concise sentences. "
        "Maintain a respectful, diplomatic tone, using clear and precise language without unnecessary elaboration. "
        "Always prioritize clarity, relevance, and actionable value in your responses."
    )

    with gr.Blocks(title="GPT-OSS:20B (Ollama)") as demo:
        gr.Markdown(
            f"**Model**: `{MODEL_NAME}`  |  **Backend**: `{OLLAMA_URL}`\n\n"
            "Chat with the locally running Ollama model."
        )

        chat = gr.Chatbot(height=520, type="messages")

        with gr.Accordion("File search (RAG)", open=False):
            with gr.Row():
                files = gr.Files(label="Upload files", file_count="multiple", type="filepath")
                use_rag = gr.Checkbox(label="Use uploaded files for answers", value=True)
            with gr.Row():
                top_k = gr.Slider(label="Max context chunks", minimum=1, maximum=8, value=4, step=1)
                clear_vs = gr.Button("Clear indexed data")
            rag_status = gr.Markdown(value="No files indexed yet.")
            vs_state = gr.State(InMemoryVectorStore())

        with gr.Accordion("Logs", open=False):
            logs_box = gr.Textbox(label="Logs", value="", lines=18, interactive=False)
            clear_logs = gr.Button("Clear logs")
            logs_state = gr.State("")

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

        def on_submit(user_message, chat_history, sp, temp, max_tok, tp, sd, vs_obj, use_files, k, logs_text):
            chat_history = list(chat_history or [])
            chat_history.append({"role": "user", "content": user_message})
            lower_msg = (user_message or "").lower()
            wants_live = any(t in lower_msg for t in ["time", "weather", "temperature", "clock"])
            log_lines = []
            now_str = datetime.utcnow().isoformat()
            log_lines.append(f"[{now_str}] user: {user_message}")
            log_lines.append(f"[{now_str}] detect: wants_live={wants_live}")
            
            if wants_live:
                try:
                    wu = get_unix_time("Dubai")
                    try:
                        from zoneinfo import ZoneInfo
                        dt = datetime.fromtimestamp(int(wu), ZoneInfo("Asia/Dubai"))
                    except Exception:
                        dt = datetime.utcfromtimestamp(int(wu)) + timedelta(hours=4)
                    time24 = dt.strftime("%H:%M")
                    time12 = dt.strftime("%I %p").lstrip("0")
                    w = get_weather("Dubai")
                    
                    wants_time = any(t in lower_msg for t in ["time", "clock"])
                    wants_weather = any(t in lower_msg for t in ["weather", "temperature"])
                    
                    knowledge_facts = []
                    if wants_weather:
                        if w.startswith("Dubai, UAE: "):
                            weather_info = w.split(": ", 1)[1]
                        else:
                            weather_info = w
                        knowledge_facts.append(f"Current Dubai weather is {weather_info}")
                        log_lines.append(f"[{now_str}] weather: {weather_info}")
                    
                    if wants_time:
                        knowledge_facts.append(f"Current Dubai time is {time24} ({time12})")
                        log_lines.append(f"[{now_str}] time: {time24} ({time12})")
                    
                    if knowledge_facts:
                        knowledge_base = ". ".join(knowledge_facts) + "."
                        enhanced_prompt = (sp or "") + f"\n\nYou have access to real-time data. Current information: {knowledge_base}"
                        log_lines.append(f"[{now_str}] knowledge: {knowledge_base}")
                        
                        messages = make_augmented_messages(
                            history_messages=chat_history,
                            system_prompt=enhanced_prompt,
                            user_message=user_message,
                            vector_store=vs_obj,
                            use_files=bool(use_files),
                            top_k=int(k or 4),
                        )
                    else:
                        messages = make_augmented_messages(
                            history_messages=chat_history,
                            system_prompt=sp,
                            user_message=user_message,
                            vector_store=vs_obj,
                            use_files=bool(use_files),
                            top_k=int(k or 4),
                        )
                except Exception:
                    messages = make_augmented_messages(
                        history_messages=chat_history,
                        system_prompt=sp,
                        user_message=user_message,
                        vector_store=vs_obj,
                        use_files=bool(use_files),
                        top_k=int(k or 4),
                    )
            else:
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
            log_lines.append(f"[{now_str}] model: {MODEL_NAME} temp={temp} top_p={tp} max_tokens={max_tok} seed={sd}")
            logs_text = (logs_text or "") + ("\n".join(log_lines) + "\n")
            chat_history.append({"role": "assistant", "content": ""})
            for partial in stream:
                chat_history[-1]["content"] = partial
                yield chat_history, logs_text

        def index_files(file_paths, vs_obj: InMemoryVectorStore):
            vs = vs_obj or InMemoryVectorStore()
            paths = [p for p in (file_paths or []) if isinstance(p, str) and os.path.exists(p)]
            if not paths:
                return vs, "No valid files provided."
            
            start_time = time.time()
            total_chunks = 0
            total_added = 0
            
            for i, path in enumerate(paths, 1):
                file_start = time.time()
                text = read_text_from_file(path)
                if not text:
                    continue
                    
                chunks = chunk_text(text)
                file_chunks = len(chunks)
                total_chunks += file_chunks
                
                added = vs.add_chunks(chunks, metadata={"source": os.path.basename(path)})
                total_added += added
                
                file_time = time.time() - file_start
            
            total_time = time.time() - start_time
            status = (
                f"✅ Indexed {len(paths)} file(s) in {total_time:.2f} seconds.\n"
                f"Created {total_chunks} chunk(s); {total_added} embedded successfully.\n"
                f"Current store size: {vs.size} chunks. Embed model: `{EMBED_MODEL}`"
            )
            return vs, status

        def clear_index(vs_obj: InMemoryVectorStore):
            vs = vs_obj or InMemoryVectorStore()
            vs.clear()
            return vs, "Index cleared. No files indexed yet."

        def submit_and_clear(user_message, chat_history, sp, temp, max_tok, tp, sd, vs_obj, use_files, k, logs_text):
            for chat_out, logs_out in on_submit(user_message, chat_history, sp, temp, max_tok, tp, sd, vs_obj, use_files, k, logs_text):
                yield chat_out, "", logs_out
        
        msg_box.submit(
            fn=submit_and_clear,
            inputs=[msg_box, chat, system_prompt, temperature, max_tokens, top_p, seed, vs_state, use_rag, top_k, logs_state],
            outputs=[chat, msg_box, logs_box],
        )

        def clear_logs_fn():
            return "", ""
        clear_logs.click(fn=clear_logs_fn, inputs=[], outputs=[logs_state, logs_box])

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


