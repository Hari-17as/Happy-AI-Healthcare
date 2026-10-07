import tkinter as tk
from tkinter import scrolledtext, messagebox
import pickle
import re
import threading
import queue
import requests
import numpy as np


# =========================================================
# SETTINGS
# =========================================================

MODEL_NAME = "qwen3:8b"
OLLAMA_URL = "http://localhost:11434/api/chat"

RAG_INDEX = r"C:\Happy\Medical_Knowledge\_rag_database\medical_index.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOP_RESULTS = 12

# How many previous (question, answer) turns to remember and
# feed back to the model so replies feel like one continuous
# conversation instead of a series of one-off Q&As.
HISTORY_TURNS = 4

REQUEST_TIMEOUT = 300


# =========================================================
# LOAD MEDICAL DATABASE (never let this crash the app silently)
# =========================================================

def load_medical_database():
    """
    Loads the RAG index and the embedding model.
    Returns (embeddings, chunks, embedder, error_message).
    If something goes wrong, error_message is a human-readable
    string and the other values are None — the caller decides
    how to show that to the user instead of the app just dying
    with a traceback.
    """

    try:
        print()
        print("Loading Happy's medical knowledge...")

        with open(RAG_INDEX, "rb") as f:
            database = pickle.load(f)

        embeddings = np.asarray(database["embeddings"], dtype=np.float32)
        chunks = database["chunks"]

        print(f"Medical chunks loaded: {len(chunks):,}")
        print("Loading medical search model...")

        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(EMBEDDING_MODEL)

        print("Happy is ready.")
        print()

        return embeddings, chunks, embedder, None

    except FileNotFoundError:
        return None, None, None, (
            f"I couldn't find my medical knowledge file at:\n{RAG_INDEX}\n\n"
            "Please check the path in RAG_INDEX and try again."
        )

    except (KeyError, TypeError, pickle.UnpicklingError):
        return None, None, None, (
            "My medical knowledge file seems to be in a format I don't "
            "recognize. It may be corrupted or built with a different "
            "version of the indexing tool."
        )

    except Exception as e:
        return None, None, None, (
            f"Something went wrong while starting up Happy:\n{e}"
        )


# =========================================================
# CONVERSATIONAL QUESTIONS (word-boundary matching, not
# naive substring matching — "hi" no longer matches "this",
# "hip", "history", etc.)
# =========================================================

CONVERSATION_PATTERNS = [
    r"\bwho are you\b",
    r"\bwhat are you\b",
    r"\bwhat('?s| is) your name\b",
    r"\byour name\b",
    r"\bwho is happy\b",
    r"\btell me about yourself\b",
    r"\bhello+\b",
    r"\bhi+\b",
    r"\bhlo\b",
    r"\bhey+\b",
    r"\bgood morning\b",
    r"\bgood afternoon\b",
    r"\bgood evening\b",
    r"\bgood night\b",
    r"\bhow are you\b",
    r"\bhow r u\b",
    r"\bhow'?s it going\b",
    r"\bthanks\b",
    r"\bthank you\b",
    r"\bty\b",
    r"\bbye\b",
    r"\bgoodbye\b",
    r"\bsee you\b",
]

CONVERSATION_REGEX = re.compile("|".join(CONVERSATION_PATTERNS), re.IGNORECASE)


def is_conversation(question):
    return bool(CONVERSATION_REGEX.search(question.strip()))


# =========================================================
# SHARED: CALL OLLAMA SAFELY
# =========================================================

def call_ollama(messages, temperature, num_predict):
    """
    Makes the request to Ollama and always returns a plain string —
    either the model's answer, or a warm, human error message.
    Never raises.
    """

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        answer = data.get("message", {}).get("content", "")

        if not answer:
            answer = data.get("response", "")

        answer = str(answer).strip()

        # Qwen3 sometimes wraps hidden reasoning in <think> tags even
        # when not wanted — strip it so the user never sees raw scratch
        # work if the model ignores the "don't show reasoning" instruction.
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

        if not answer:
            return "Sorry, I drew a blank there — could you ask that again?"

        return answer

    except requests.exceptions.ConnectionError:
        return (
            "I can't reach my local model right now. Could you check that "
            "Ollama is running, and then try again?"
        )

    except requests.exceptions.Timeout:
        return "That one took me too long to think through — mind trying again?"

    except (ValueError, KeyError):
        return "I got back something I couldn't quite understand. Let's try that again."

    except Exception as e:
        return f"Something went wrong on my end just now.\n\nDetails: {e}"


# =========================================================
# HAPPY'S PERSONALITY RESPONSE (casual conversation)
# =========================================================

def ask_conversation(question, history):

    system_prompt = """
You are Happy, a warm, caring personal healthcare companion.

You talk the way a kind, emotionally present friend would — naturally,
briefly, and never like a textbook or a customer-service script.

Vary your phrasing; don't reuse the same stock greeting every time.
Match the user's tone: if they're chatty, be a little chatty back;
if they're brief, keep it brief.

If asked who you are, explain simply that you're Happy, their personal
healthcare companion — someone they can talk to about health questions,
but not a replacement for a qualified doctor. Keep that explanation short
and only give it once unless asked again.

Never mention that you are using a medical library, retrieval, RAG,
embeddings, or anything technical about how you work.
"""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})

    return call_ollama(messages, temperature=0.6, num_predict=300)


# =========================================================
# MEDICAL SEARCH
# =========================================================

def search_medical_knowledge(question, embeddings, chunks, embedder):

    query_embedding = embedder.encode(question, normalize_embeddings=True)

    scores = np.dot(embeddings, query_embedding)

    best_indices = np.argsort(scores)[::-1]

    results = []
    used_chunks = set()

    for index in best_indices:

        chunk = chunks[index]

        if not isinstance(chunk, dict):
            continue

        text = str(chunk.get("text", "")).strip()
        book = str(chunk.get("book", "Unknown source"))
        page = str(chunk.get("page", "Unknown"))

        if not text:
            continue

        chunk_id = (book, page, text[:100])

        if chunk_id in used_chunks:
            continue

        used_chunks.add(chunk_id)

        results.append({
            "text": text,
            "book": book,
            "page": page,
            "score": float(scores[index]),
        })

        if len(results) >= TOP_RESULTS:
            break

    return results


# =========================================================
# MEDICAL QUESTION
# =========================================================

def ask_medical(question, with_proof, history, embeddings, chunks, embedder):

    try:
        results = search_medical_knowledge(question, embeddings, chunks, embedder)
    except Exception as e:
        return f"I ran into a problem searching my medical library.\n\nDetails: {e}"

    if not results:
        return "I don't have enough information in my medical library to answer that confidently."

    context_parts = []
    for result in results:
        context_parts.append(
            "BOOK: " + result["book"] +
            "\nPAGE: " + result["page"] +
            "\nCONTENT:\n" + result["text"]
        )

    context = "\n\n====================\n\n".join(context_parts)

    if with_proof:
        source_instruction = """
After the answer, on new lines, give the single most relevant source
using exactly this format:

Source: [book name]
Page: [page number]

Do not show your reasoning, RAG details, or similarity scores.
Do not dump the raw source text.
"""
    else:
        source_instruction = """
Do not mention sources, books, pages, RAG, retrieval, embeddings,
similarity scores, or any technical details.
Do not show your reasoning — give only the natural final answer.
"""

    system_prompt = f"""
You are Happy, a warm and caring healthcare companion talking with someone
who trusts you with their health questions.

Speak the way a calm, attentive human would in conversation — clear,
brief, and kind, not clinical or robotic. It's fine to acknowledge how
the person might be feeling before giving the facts, especially for
worrying symptoms.

{source_instruction}

Use the medical information supplied below as your primary source.

IMPORTANT:
- When the question asks for a list, classification, branches, steps,
  causes, or features, look through ALL supplied medical information
  before answering — don't stop after the first item you find.
- Give the complete answer when the supplied information supports it.
- Do not invent information that isn't supported by the supplied material.
- If the information is insufficient, say plainly:
  "I don't have enough information in my medical library to answer that confidently."
- For anything that sounds potentially urgent or life-threatening, gently
  but clearly encourage the person to seek immediate professional medical
  help or local emergency services.
- Use correct medical terminology, and explain harder terms simply.
- You are an AI healthcare companion, not a replacement for a qualified doctor.

MEDICAL INFORMATION:

{context}
"""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})

    return call_ollama(messages, temperature=0.2, num_predict=700)


# =========================================================
# MAIN HAPPY BRAIN
# =========================================================

def ask_happy(question, with_proof, history, embeddings, chunks, embedder):

    try:
        if is_conversation(question):
            return ask_conversation(question, history)

        return ask_medical(question, with_proof, history, embeddings, chunks, embedder)

    except Exception as e:
        # Absolute last line of defense — Happy should never just crash
        # or go silent on the user, whatever happens.
        return (
            "I hit an unexpected snag answering that, but I'm still here. "
            f"Could you try rephrasing it?\n\n(Details: {e})"
        )


# =========================================================
# HAPPY GUI
# =========================================================

class HappyGUI:

    def __init__(self, root, embeddings, chunks, embedder, startup_error):

        self.root = root
        self.embeddings = embeddings
        self.chunks = chunks
        self.embedder = embedder

        # Short rolling memory of the conversation so replies feel
        # continuous rather than one-off.
        self.history = []

        # Queue used to safely pass results back from the worker
        # thread to the Tkinter main thread.
        self.result_queue = queue.Queue()

        self._typing_dots = 0
        self._typing_job = None

        root.title("Happy — Healthcare Companion")
        root.geometry("850x650")
        root.minsize(650, 500)

        self.chat = scrolledtext.ScrolledText(
            root,
            wrap=tk.WORD,
            font=("Segoe UI", 12),
            state=tk.DISABLED,
        )
        self.chat.pack(fill=tk.BOTH, expand=True, padx=12, pady=(12, 6))

        bottom = tk.Frame(root)
        bottom.pack(fill=tk.X, padx=12, pady=12)

        self.entry = tk.Entry(bottom, font=("Segoe UI", 12))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.entry.bind("<Return>", self.send_message)

        self.send_button = tk.Button(
            bottom, text="Send", font=("Segoe UI", 11), command=self.send_message
        )
        self.send_button.pack(side=tk.RIGHT)

        if startup_error:
            self.add_message(
                "Happy",
                "Hi, I'm Happy. " + startup_error +
                "\n\nI can still chat, but I won't be able to answer "
                "medical questions until that's fixed.",
            )
        else:
            self.add_message("Happy", "Hi! I'm Happy. How can I help you today?")

        self.entry.focus()

    # =====================================================
    # DISPLAY MESSAGE
    # =====================================================

    def add_message(self, speaker, message):
        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, f"{speaker}: {message}\n\n")
        self.chat.see(tk.END)
        self.chat.config(state=tk.DISABLED)

    def start_typing_indicator(self):
        self.chat.config(state=tk.NORMAL)
        self._typing_start_index = self.chat.index(tk.END)
        self.chat.insert(tk.END, "Happy is thinking...\n\n")
        self.chat.see(tk.END)
        self.chat.config(state=tk.DISABLED)
        self._animate_typing_indicator()

    def _animate_typing_indicator(self):
        self._typing_dots = (self._typing_dots + 1) % 4
        dots = "." * self._typing_dots
        line_start = f"{self._typing_start_index.split('.')[0]}.0"
        line_end = f"{self._typing_start_index.split('.')[0]}.end"

        self.chat.config(state=tk.NORMAL)
        self.chat.delete(line_start, line_end)
        self.chat.insert(line_start, f"Happy is thinking{dots}")
        self.chat.config(state=tk.DISABLED)

        self._typing_job = self.root.after(400, self._animate_typing_indicator)

    def stop_typing_indicator(self):
        if self._typing_job is not None:
            self.root.after_cancel(self._typing_job)
            self._typing_job = None

        line_num = self._typing_start_index.split(".")[0]
        self.chat.config(state=tk.NORMAL)
        self.chat.delete(f"{line_num}.0", f"{int(line_num) + 2}.0")
        self.chat.config(state=tk.DISABLED)

    # =====================================================
    # SEND MESSAGE
    # =====================================================

    def send_message(self, event=None):

        question = self.entry.get().strip()

        if not question:
            return

        self.entry.delete(0, tk.END)
        self.add_message("You", question)

        with_proof = False
        if question.lower().startswith("/withproof"):
            with_proof = True
            question = question[len("/withproof"):].strip()

        if not question:
            self.add_message("Happy", "Sure — what would you like me to check?")
            return

        self.send_button.config(state=tk.DISABLED)
        self.entry.config(state=tk.DISABLED)
        self.start_typing_indicator()

        # Run the (potentially slow) model call on a background thread
        # so the window never freezes while Happy is "thinking".
        worker = threading.Thread(
            target=self._worker_ask,
            args=(question, with_proof),
            daemon=True,
        )
        worker.start()

        self.root.after(100, self._poll_result)

    def _worker_ask(self, question, with_proof):
        try:
            answer = ask_happy(
                question, with_proof, list(self.history),
                self.embeddings, self.chunks, self.embedder,
            )
        except Exception as e:
            answer = f"I hit an unexpected problem, but I'm still here.\n\nDetails: {e}"

        self.result_queue.put((question, answer))

    def _poll_result(self):
        try:
            question, answer = self.result_queue.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_result)
            return

        self.stop_typing_indicator()
        self.add_message("Happy", answer)

        # Remember this exchange for conversational continuity.
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        self.history = self.history[-(HISTORY_TURNS * 2):]

        self.send_button.config(state=tk.NORMAL)
        self.entry.config(state=tk.NORMAL)
        self.entry.focus()


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    embeddings, chunks, embedder, startup_error = load_medical_database()

    root = tk.Tk()

    if startup_error:
        # Let the user know loudly once, but still let the app run so
        # they can at least talk to Happy and fix the config.
        messagebox.showwarning("Happy — Setup issue", startup_error)

    app = HappyGUI(root, embeddings, chunks, embedder, startup_error)

    root.mainloop()