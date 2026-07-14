"""
Teequ Agent - ChatGPT-style UI Version
----------------------------------------------
Medical store inventory agent that only responds when the query contains
its wake word "Teequ". Reads/writes data from ONE Excel file.

Run with:  streamlit run store_app.py

Everywhere you MUST change something is marked: # >>> CHANGE
"""

import io
import os
import json
import difflib
import hashlib
import base64

import pandas as pd
import streamlit as st
from gtts import gTTS
from groq import Groq

# ============ CONFIG - CHANGE THESE ============

# >>> CHANGE: put your key in an environment variable instead of hardcoding it.
# PowerShell (run once per terminal session before `streamlit run ...`):
#   $env:GROQ_API_KEY = "your_real_key_here"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")

if not GROQ_API_KEY:
    st.error(
        "GROQ_API_KEY environment variable is not set. "
        "In PowerShell run:  $env:GROQ_API_KEY = \"your_key_here\"  "
        "then restart `streamlit run store_app.py`."
    )
    st.stop()

groq_client = Groq(api_key=GROQ_API_KEY)

GROQ_MODEL = "llama-3.3-70b-versatile"        # >>> CHANGE: swap chat model if you want
GROQ_STT_MODEL = "whisper-large-v3"           # >>> CHANGE: Groq-hosted speech-to-text model

EXCEL_PATH = "inventory.xlsx"                 # >>> CHANGE: path to your real inventory file

AGENT_NAME = "teequ"                          # >>> CHANGE: wake word/phrase (checked in lowercase)
# >>> CHANGE: add every spelling/script variant Whisper might transcribe when you say the
# wake word in a different language. Whisper writes the name in whatever script matches
# the language being spoken (e.g. Urdu speech -> Urdu letters), so an English-only check
# will never match Urdu/Punjabi speech even if it "sounds" the same.
WAKE_WORD_VARIANTS = ["teequ", "ٹیکو", "ٹیقو", "تیکو"]
AGENT_DISPLAY = AGENT_NAME.title()

OFF_MESSAGE = {
    "en": f'Agent is off. Say "{AGENT_DISPLAY}" to activate it.',
    "ur": "ایجنٹ بند ہے۔ فعال کرنے کے لیے \"ٹیکو\" کہیں۔",
    "pa": "ایجنٹ بند اے۔ چالو کرن لئی \"ٹیکو\" آکھو۔",
}

OUT_OF_SCOPE_MESSAGE = {
    "en": "I only handle stock, price, and sales questions for our medical store.",
    "ur": "میں صرف اسٹور کی اسٹاک، قیمت اور سیل سے متعلق سوالات میں مدد کر سکتا ہوں۔",
    "pa": "میں صرف اسٹور دے اسٹاک، قیمت تے سیل نال متعلق سوالاں وچ مدد کر سکنا واں۔",
}

# ============ AUTO-CREATE EXCEL FILE IF IT DOESN'T EXIST ============

SAMPLE_PRODUCTS = [
    {"Product Name": "Panadol (Paracetamol 500mg)", "Stock Quantity": 320, "Unit Price": 5.0,
     "Sales Today": 18, "Sales This Week": 96, "Last Restock Date": "2026-07-05"},
    {"Product Name": "Brufen (Ibuprofen 400mg)", "Stock Quantity": 45, "Unit Price": 8.0,
     "Sales Today": 6, "Sales This Week": 40, "Last Restock Date": "2026-07-01"},
    {"Product Name": "Augmentin (Amoxicillin 625mg)", "Stock Quantity": 12, "Unit Price": 250.0,
     "Sales Today": 2, "Sales This Week": 15, "Last Restock Date": "2026-06-28"},
    {"Product Name": "ORS Sachet", "Stock Quantity": 500, "Unit Price": 15.0,
     "Sales Today": 30, "Sales This Week": 180, "Last Restock Date": "2026-07-07"},
]

if not os.path.exists(EXCEL_PATH):
    pd.DataFrame(SAMPLE_PRODUCTS).to_excel(EXCEL_PATH, index=False)


@st.cache_data
def load_inventory():
    return pd.read_excel(EXCEL_PATH)


inventory_df = load_inventory()

# ============ STEP 1: Voice -> Text ============

def transcribe_audio(audio_bytes: bytes) -> str:
    """Send recorded audio to Groq's hosted Whisper model for transcription."""
    if not audio_bytes or len(audio_bytes) < 500:
        # Very small / empty buffers usually mean the mic didn't actually capture anything.
        raise ValueError(
            "Recording seems empty (no audio captured). "
            "Check your microphone permission and try recording again for at least 1-2 seconds."
        )
    transcription = groq_client.audio.transcriptions.create(
        file=("audio.wav", audio_bytes),
        model=GROQ_STT_MODEL,
    )
    return transcription.text


# ============ STEP 2: Wake word check (fuzzy, tolerant of mishearing) ============

def _fuzzy_contains(text_lower: str, phrase: str, threshold: float = 0.72) -> bool:
    if phrase in text_lower:
        return True
    words = text_lower.split()
    phrase_words = phrase.split()
    n = len(phrase_words)
    for i in range(len(words) - n + 1):
        chunk = " ".join(words[i:i + n])
        if difflib.SequenceMatcher(None, chunk, phrase).ratio() >= threshold:
            return True
    return False


def contains_wake_word(text: str) -> bool:
    text_lower = text.lower()
    return any(_fuzzy_contains(text_lower, variant.lower()) for variant in WAKE_WORD_VARIANTS)


def strip_wake_word(text: str) -> str:
    text_lower = text.lower()
    for variant in WAKE_WORD_VARIANTS:
        variant_lower = variant.lower()
        if variant_lower in text_lower:
            return text_lower.replace(variant_lower, "").strip(" ,.-")
        words = text_lower.split()
        phrase_words = variant_lower.split()
        n = len(phrase_words)
        for i in range(len(words) - n + 1):
            chunk = " ".join(words[i:i + n])
            if difflib.SequenceMatcher(None, chunk, variant_lower).ratio() >= 0.72:
                return " ".join(words[:i] + words[i + n:]).strip(" ,.-")
    return text_lower.strip(" ,.-")


# ============ STEP 3: Understand the query (intent + language) ============

INTENT_PROMPT = """Read this medical store query and respond ONLY with valid JSON (no other text):
{{
  "product_name": "extracted product name, or null if not mentioned",
  "query_type": "stock" or "price" or "sales" or "restock_date" or "general" or "greeting" or "out_of_scope",
  "language": "en" or "ur" or "pa"
}}

Use "greeting" for hellos, "salam", small talk like "how are you", or casual chit-chat with no
real store question. Use "out_of_scope" only for things unrelated to the store (weather, jokes,
politics, etc.) that are NOT a greeting.

Query: "{query}"
"""


def parse_intent(query_text: str) -> dict:
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": INTENT_PROMPT.format(query=query_text)}],
    )
    raw = response.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


# ============ STEP 4: Retrieve data from Excel ============

def retrieve_product(product_name):
    if not product_name:
        return None
    names = inventory_df["Product Name"].tolist()
    matches = difflib.get_close_matches(product_name, names, n=1, cutoff=0.3)
    if not matches:
        return None
    row = inventory_df[inventory_df["Product Name"] == matches[0]].iloc[0]
    return row.to_dict()


# ============ STEP 5: Compose the final reply ============

RESPONSE_PROMPT = """You are "{agent}", a helpful medical store assistant. Reply in {language}
(en=English, ur=Urdu, pa=Punjabi) using ONLY the data given below. Be concise (2-4 sentences).

User's question: "{query}"

Data available:
{data}

If data is empty or missing, clearly say the information is not available - do not make anything up.
"""


GREETING_PROMPT = """You are "{agent}", a friendly voice assistant for a medical store. The user just
greeted you or made small talk (e.g. "salam", "how are you"). Reply warmly in {language}
(en=English, ur=Urdu, pa=Punjabi), 1-2 sentences, and briefly mention you can help with stock,
price, sales, or restock questions.

User said: "{query}"
"""


def compose_greeting_response(query_text, language):
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=120,
        messages=[{"role": "user", "content": GREETING_PROMPT.format(
            agent=AGENT_DISPLAY, query=query_text, language=language
        )}],
    )
    return response.choices[0].message.content.strip()


def compose_response(query_text, product_data, language):
    data_str = json.dumps(product_data, default=str) if product_data else "No matching product found."
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": RESPONSE_PROMPT.format(
            agent=AGENT_DISPLAY, query=query_text, data=data_str, language=language
        )}],
    )
    return response.choices[0].message.content.strip()


# ============ STEP 6: Text -> Voice (optional) ============

GTTS_LANG_MAP = {"en": "en", "ur": "ur", "pa": "pa"}


def text_to_speech(text, language):
    try:
        tts = gTTS(text=text, lang=GTTS_LANG_MAP.get(language, "en"))
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf
    except Exception:
        return None


# ============ SHARED: process a query end-to-end and store it in chat history ============

def process_query_and_reply(raw_query_text: str):
    """Wake-word check -> intent -> data lookup -> reply -> TTS. Appends to chat_history.
    Returns (reply_text, reply_audio_bytes_or_None, language)."""
    st.session_state.chat_history.append({"role": "user", "kind": "user", "text": raw_query_text})

    if not contains_wake_word(raw_query_text):
        msg = OFF_MESSAGE["en"]
        st.session_state.chat_history.append({"role": "agent", "kind": "warning", "text": msg})
        return msg, None, "en"

    cleaned_query = strip_wake_word(raw_query_text)
    try:
        with st.spinner(f"{AGENT_DISPLAY} is checking..."):
            intent = parse_intent(cleaned_query)
            language = intent.get("language", "en")
            if intent.get("query_type") == "out_of_scope":
                reply = OUT_OF_SCOPE_MESSAGE.get(language, OUT_OF_SCOPE_MESSAGE["en"])
            elif intent.get("query_type") == "greeting":
                reply = compose_greeting_response(cleaned_query, language)
            else:
                product_data = retrieve_product(intent.get("product_name"))
                reply = compose_response(cleaned_query, product_data, language)

        audio_buf = text_to_speech(reply, language)
        reply_audio_bytes = audio_buf.read() if audio_buf else None
        st.session_state.chat_history.append({
            "role": "agent", "kind": "agent", "text": reply, "audio": reply_audio_bytes,
        })
        return reply, reply_audio_bytes, language
    except json.JSONDecodeError:
        msg = "Sorry, I had trouble understanding that request. Please rephrase and try again."
        st.session_state.chat_history.append({"role": "agent", "kind": "error", "text": msg})
        return msg, None, "en"
    except Exception as e:
        msg = f"An error occurred: {e}"
        st.session_state.chat_history.append({"role": "agent", "kind": "error", "text": msg})
        return msg, None, "en"


def autoplay_audio_html(audio_bytes: bytes, fmt: str = "audio/mp3"):
    """Embed an <audio autoplay> tag so the reply plays immediately, like an actual call."""
    if not audio_bytes:
        return
    b64 = base64.b64encode(audio_bytes).decode()
    st.markdown(
        f'<audio autoplay="true" controls style="width:100%; margin: 6px 0;">'
        f'<source src="data:{fmt};base64,{b64}" type="{fmt}">'
        f"</audio>",
        unsafe_allow_html=True,
    )


# ============ STREAMLIT UI (ChatGPT-style) ============

st.set_page_config(page_title="Call, Voice & Text Agent for Inventory Stock", page_icon="💊", layout="centered")

# ---- Custom CSS: dark ChatGPT-like theme with colored chat bubbles ----
st.markdown("""
<style>
    .stApp {
        background-color: #0f1117;
    }
    .main .block-container {
        max-width: 780px;
        padding-top: 2rem;
    }
    h1 {
        color: #ffffff !important;
        font-weight: 700;
    }
    .agent-subtitle {
        color: #9ca3af;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .chat-bubble-user {
        background: linear-gradient(135deg, #6366f1, #4f46e5);
        color: #ffffff;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 85%;
        margin-left: auto;
        font-size: 0.98rem;
        line-height: 1.5;
        box-shadow: 0 2px 8px rgba(79,70,229,0.35);
    }
    .chat-bubble-agent {
        background-color: #1e2130;
        color: #e5e7eb;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 85%;
        margin-right: auto;
        font-size: 0.98rem;
        line-height: 1.5;
        border-left: 3px solid #22c55e;
    }
    .chat-bubble-warning {
        background-color: #3f2d1a;
        color: #fbbf24;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 85%;
        margin-right: auto;
        border-left: 3px solid #f59e0b;
    }
    .chat-bubble-error {
        background-color: #3f1a1a;
        color: #f87171;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 85%;
        margin-right: auto;
        border-left: 3px solid #ef4444;
    }
    .sender-label {
        font-size: 0.75rem;
        color: #6b7280;
        margin-bottom: 2px;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    div[data-testid="stRadio"] > label {
        color: #d1d5db !important;
    }
    .stButton>button {
        background: linear-gradient(135deg, #6366f1, #4f46e5);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1.4rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #4f46e5, #4338ca);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("💊 Call, Voice & Text Agent for Inventory Stock")
st.markdown(
    f'<p class="agent-subtitle">Say or type <b>"{AGENT_DISPLAY}"</b> followed by your question. '
    f'Handles stock, price, sales &amp; restock questions for the medical store.</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.caption("🔧 Debug info (confirms you're running the latest file)")
    st.code(f"Wake word variants: {WAKE_WORD_VARIANTS}", language=None)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of dicts: {role, kind, text}

# ---- Render existing chat history ----
for msg in st.session_state.chat_history:
    role = msg["role"]
    kind = msg.get("kind", "normal")
    label = "You" if role == "user" else AGENT_DISPLAY
    css_class = {
        "user": "chat-bubble-user",
        "agent": "chat-bubble-agent",
        "warning": "chat-bubble-warning",
        "error": "chat-bubble-error",
    }.get(kind if role != "user" else "user", "chat-bubble-agent")
    st.markdown(f'<div class="sender-label">{label}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="{css_class}">{msg["text"]}</div>', unsafe_allow_html=True)
    if msg.get("audio"):
        st.audio(msg["audio"], format="audio/mp3")

st.divider()

input_mode = st.radio("Input method", ["💬 Text", "🎤 Voice Message", "📞 Voice Call"], horizontal=True)

# ---------------------------------------------------------------------------
# MODE 1: Text
# ---------------------------------------------------------------------------
if input_mode == "💬 Text":
    query_text = st.text_input(f'Type your question (must include "{AGENT_DISPLAY}")', key="text_input_box")
    if query_text and st.button("Send", type="primary"):
        process_query_and_reply(query_text)
        st.rerun()

# ---------------------------------------------------------------------------
# MODE 2: Voice Message (single turn, auto-replies once transcribed - no Send click needed)
# ---------------------------------------------------------------------------
elif input_mode == "🎤 Voice Message":
    voice_source = st.radio(
        "How do you want to provide voice?",
        ["🎙️ Record with mic", "📁 Upload audio file"],
        horizontal=True,
    )

    audio_bytes = None

    if voice_source == "🎙️ Record with mic":
        audio = st.audio_input("Record your question")
        if audio is not None:
            audio_bytes = audio.getvalue()
    else:
        uploaded_audio = st.file_uploader(
            "Upload a short recording (wav, mp3, m4a, ogg)",
            type=["wav", "mp3", "m4a", "ogg", "webm"],
        )
        if uploaded_audio is not None:
            audio_bytes = uploaded_audio.getvalue()
            st.audio(audio_bytes)

    transcription_error = None

    if audio_bytes is not None:
        audio_hash = hashlib.md5(audio_bytes).hexdigest()
        if st.session_state.get("last_voice_hash") == audio_hash:
            st.caption("✅ This recording was already sent. Record or upload a new one to send another message.")
        else:
            with st.spinner("Transcribing..."):
                try:
                    query_text = transcribe_audio(audio_bytes)
                    if not query_text or not query_text.strip():
                        transcription_error = (
                            "Transcription returned empty text. Please speak clearly and "
                            "make sure the recording is at least 1-2 seconds long."
                        )
                    else:
                        st.session_state.last_voice_hash = audio_hash
                        st.info(f"**Heard:** {query_text}")
                        process_query_and_reply(query_text)   # <-- auto-replies immediately, no button needed
                        st.rerun()
                except Exception as e:
                    transcription_error = f"{type(e).__name__}: {e}"

    if transcription_error:
        st.error(f"🎤 Voice input problem: {transcription_error}")
        with st.expander("Troubleshooting tips"):
            st.markdown(
                "- Make sure your browser has **microphone permission** allowed for this site "
                "(click the lock/site-info icon in the address bar).\n"
                "- Use **Chrome or Edge** — some browsers don't support the recorder widget well.\n"
                "- Access the app via `http://localhost:8501` rather than an IP address.\n"
                "- Record for at least 1-2 seconds; a very short/empty clip fails transcription.\n"
                "- Check Windows Settings → Privacy & Security → Microphone → allow apps/browser access."
            )

# ---------------------------------------------------------------------------
# MODE 3: Voice Call (continuous turn-based call until "End Call")
# ---------------------------------------------------------------------------
else:
    st.markdown(
        "<p style='color:#9ca3af; font-size:0.85rem;'>Note: true hands-free listening isn't possible in a "
        "browser app without extra streaming infrastructure. Each turn you record → release → "
        f"{AGENT_DISPLAY} auto-transcribes, answers, and speaks the reply — then a fresh recorder "
        "appears for your next turn, so it feels like a continuous call.</p>",
        unsafe_allow_html=True,
    )

    if "call_active" not in st.session_state:
        st.session_state.call_active = False
    if "call_turn" not in st.session_state:
        st.session_state.call_turn = 0
    if "call_last_hash" not in st.session_state:
        st.session_state.call_last_hash = None
    if "call_greeted" not in st.session_state:
        st.session_state.call_greeted = False

    if not st.session_state.call_active:
        if st.button("📞 Start Call", type="primary"):
            st.session_state.call_active = True
            st.session_state.call_turn = 0
            st.session_state.call_last_hash = None
            st.session_state.call_greeted = False
            st.rerun()
    else:
        st.success(f'🟢 Call active — say "{AGENT_DISPLAY}" followed by your question, then wait for the reply.')

        if not st.session_state.call_greeted:
            greeting = f"Hello! {AGENT_DISPLAY} here. How can I help you with the store today?"
            st.markdown('<div class="sender-label">' + AGENT_DISPLAY + '</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-bubble-agent">{greeting}</div>', unsafe_allow_html=True)
            greet_buf = text_to_speech(greeting, "en")
            if greet_buf:
                autoplay_audio_html(greet_buf.read())
            st.session_state.call_greeted = True

        rec_key = f"call_audio_{st.session_state.call_turn}"
        audio = st.audio_input("🎙️ Your turn", key=rec_key)

        if audio is not None:
            audio_bytes = audio.getvalue()
            audio_hash = hashlib.md5(audio_bytes).hexdigest()
            if audio_hash != st.session_state.call_last_hash:
                st.session_state.call_last_hash = audio_hash
                try:
                    with st.spinner("Listening..."):
                        heard_text = transcribe_audio(audio_bytes)

                    if heard_text and heard_text.strip():
                        st.markdown('<div class="sender-label">You said</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="chat-bubble-user">{heard_text}</div>', unsafe_allow_html=True)

                        reply, reply_audio, language = process_query_and_reply(heard_text)

                        st.markdown('<div class="sender-label">' + AGENT_DISPLAY + '</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="chat-bubble-agent">{reply}</div>', unsafe_allow_html=True)
                        if reply_audio:
                            autoplay_audio_html(reply_audio)
                        else:
                            st.caption("(Voice reply not available for this language — text shown above.)")

                        st.session_state.call_turn += 1  # fresh recorder widget for the next turn
                    else:
                        st.warning("Didn't catch that clearly — please try recording again.")
                except Exception as e:
                    st.error(f"Error during call: {type(e).__name__}: {e}")

        if st.button("📴 End Call"):
            st.session_state.call_active = False
            st.session_state.call_greeted = False
            st.rerun()
    if st.button("🗑️ Clear chat"):
        st.session_state.chat_history = []
        st.rerun()