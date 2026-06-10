import streamlit as st
import os
import numpy as np
import speech_recognition as sr
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai
from google.genai import types
from gtts import gTTS
from pydub import AudioSegment
from dotenv import load_dotenv
import io
import time
import base64
import streamlit.components.v1 as components
import re
from audio_recorder_component import audio_recorder

# --- PAGE CONFIG ---
st.set_page_config(page_title="Voice AI", page_icon="🎙️", layout="centered")

# --- SIMPLE CLEAN CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main .block-container { padding-top: 2rem; max-width: 700px; }
    #MainMenu, footer, header {visibility: hidden;}
    
    .status-card {
        text-align: center;
        padding: 30px;
        border-radius: 20px;
        margin: 20px auto;
        max-width: 500px;
    }
    .status-idle {
        background: linear-gradient(135deg, #1e293b, #334155);
        border: 2px solid #475569;
    }
    .status-listening {
        background: linear-gradient(135deg, #059669, #10b981);
        border: 2px solid #34d399;
        animation: pulse 1.5s ease-in-out infinite;
    }
    .status-processing {
        background: linear-gradient(135deg, #3b82f6, #6366f1);
        border: 2px solid #818cf8;
    }
    .status-speaking {
        background: linear-gradient(135deg, #8b5cf6, #a855f7);
        border: 2px solid #c084fc;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); box-shadow: 0 0 30px rgba(16, 185, 129, 0.4); }
        50% { transform: scale(1.02); box-shadow: 0 0 50px rgba(16, 185, 129, 0.6); }
    }
    .status-icon { font-size: 60px; margin-bottom: 15px; }
    .status-title { font-size: 24px; font-weight: 700; color: white; margin-bottom: 8px; }
    .status-subtitle { font-size: 14px; color: rgba(255,255,255,0.8); }
    
    .result-box {
        background: rgba(30, 41, 59, 0.9);
        border: 1px solid #475569;
        border-radius: 16px;
        padding: 20px;
        margin: 15px 0;
    }
    .result-label { font-size: 12px; color: #94a3b8; text-transform: uppercase; margin-bottom: 8px; }
    .result-text { font-size: 16px; color: #e2e8f0; line-height: 1.6; }
    
    .stButton > button {
        border-radius: 30px !important;
        padding: 15px 40px !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        border: none !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3) !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #10b981, #059669) !important;
        color: white !important;
    }
    .stButton > button[kind="secondary"] {
        background: linear-gradient(135deg, #ef4444, #dc2626) !important;
        color: white !important;
    }
    
    .stApp { background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%); }
    [data-testid="stSidebar"] { background: #0f172a; }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# --- LOAD ENV ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DB_FAISS_PATH = "vectorstore/db_faiss"

LANGUAGES = {
    "English": {"gtts": "en", "flag": "🇬🇧"},
    "Hindi": {"gtts": "hi", "flag": "🇮🇳"},
    "Hinglish": {"gtts": "hi", "flag": "🔀"},
    "Tamil": {"gtts": "ta", "flag": "🇮🇳"},
    "Telugu": {"gtts": "te", "flag": "🇮🇳"},
}

if not GOOGLE_API_KEY:
    st.error("⚠️ GOOGLE_API_KEY not found")
    st.stop()

# ==================== LOAD MODELS ====================

def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

def load_faiss(embeddings):
    try:
        return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    except:
        return None

def setup_gemini():
    client = genai.Client(api_key=GOOGLE_API_KEY)
    return client

def build_chain(db, model):
    if db is None:
        return None
    
    def qa_chain(question):
        try:
            docs = db.similarity_search(question, k=3)
            # Store the retrieved documents in the session state for reference in the UI
            st.session_state.retrieved_docs = docs
            context = "\n\n".join([doc.page_content for doc in docs])
            
            prompt = f"""Answer the question using the context provided. Keep answer brief (1-2 sentences).

Context:
{context}

Question: {question}

Answer:"""
            
            response = model.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=2048,
                )
            )
            return response.text
        except Exception as e:
            return f"Error: {str(e)[:100]}"
    
    return qa_chain

# Initialize session state for DB loading
if "db_loaded" not in st.session_state:
    st.session_state.db_loaded = False

with st.spinner("🚀 Loading..."):
    embeddings = load_embeddings()
    db = load_faiss(embeddings)
    gemini_model = setup_gemini()
    qa_chain = build_chain(db, gemini_model)
    if db is not None:
        st.session_state.db_loaded = True

# ==================== LANGUAGE DETECTION ====================

def has_hindi_script(text): return any('\u0900' <= c <= '\u097F' for c in text)
def has_tamil_script(text): return any('\u0B80' <= c <= '\u0BFF' for c in text)
def has_telugu_script(text): return any('\u0C00' <= c <= '\u0C7F' for c in text)

# Hinglish words - ONLY Hindi words in Roman script, NO English words
HINGLISH_WORDS = {
    # Hindi verbs in Roman
    'kya', 'hai', 'hain', 'tha', 'thi', 'hoga', 'hogi', 'hoge',
    'karo', 'karna', 'karte', 'karti', 'karenge', 'karega', 'karegi', 'kiya',
    'batao', 'batana', 'bata', 'bataye', 'batayein',
    'bolo', 'bolna', 'bola', 'boli',
    'dekho', 'dekhna', 'dekh', 'dekha', 'dekhi', 'dekhenge',
    'suno', 'sunna', 'suna', 'suni',
    'jao', 'jana', 'gaya', 'gayi', 'gaye', 'jayega', 'jayegi',
    'aao', 'aana', 'aaya', 'aayi', 'aaye', 'aayega', 'aayegi',
    'lena', 'liya', 'liye', 'lenge', 'lega', 'legi',
    'dena', 'diya', 'diye', 'denge', 'dega', 'degi',
    'rakh', 'rakho', 'rakhna', 'rakha', 'rakhi',
    'samjho', 'samajh', 'samjha', 'samjhi', 'samjhao',
    'padho', 'padhna', 'padha', 'padhi',
    'likho', 'likhna', 'likha', 'likhi',
    'khao', 'khana', 'khaya', 'khayi',
    'piyo', 'peena', 'piya',
    'socho', 'sochna', 'socha', 'sochi',
    'ruko', 'rukna', 'ruka', 'ruki',
    'chalo', 'chalna', 'chala', 'chali', 'chalega', 'chalegi',
    'milo', 'milna', 'mila', 'mili', 'milega', 'milegi',
    
    # Hindi pronouns in Roman
    'mujhe', 'mujhko', 'mera', 'meri', 'mere',
    'tum', 'tumhe', 'tumko', 'tumhara', 'tumhari', 'tumhare', 'tera', 'teri', 'tere',
    'aap', 'aapko', 'aapka', 'aapki', 'aapke',
    'hum', 'hume', 'humko', 'humara', 'humari', 'humare', 'hamara', 'hamari', 'hamare',
    'woh', 'usko', 'usse', 'uska', 'uski', 'uske', 'unko', 'unka', 'unki', 'unke',
    'yeh', 'isko', 'isse', 'iska', 'iski', 'iske', 'inko', 'inka', 'inki', 'inke',
    'koi', 'kuch', 'sabhi', 'sabko',
    'apna', 'apni', 'apne', 'khud',
    
    # Hindi question words
    'kaise', 'kaisa', 'kaisi', 'kab', 'kahan', 'kahaan', 'kidhar',
    'kyun', 'kyu', 'kyunki', 'kaun', 'kiska', 'kiski', 'kiske',
    'kitna', 'kitni', 'kitne', 'kaunsa', 'kaunsi', 'kaunse',
    
    # Hindi common words
    'accha', 'achha', 'acha', 'theek', 'thik', 'sahi',
    'bahut', 'bohot', 'bohat', 'zyada', 'jyada', 'thoda', 'thodi', 'thode',
    'abhi', 'jab', 'phir', 'fir', 'baad', 'pehle', 'pahle',
    'nahi', 'nahin', 'nhi', 'mat', 'haan', 'bilkul',
    'aur', 'lekin', 'magar', 'toh', 'bhi', 'sirf',
    'mein', 'wala', 'wali', 'wale',
    'yahan', 'wahan', 'idhar', 'udhar', 'kahin', 'saath', 'sath', 'paas',
    'agar', 'warna', 'isliye', 'islye',
    'pata', 'maloom', 'malum',
    'kaam', 'hogaya', 'hogya', 'kardiya', 'kardia',
    'zaroor', 'zarur', 'pakka', 'shayad',
    'ek', 'teen', 'char', 'paanch', 'panch',
    'pehla', 'pehli', 'doosra', 'doosri', 'doosre',
    'bada', 'badi', 'bade', 'chhota', 'chhoti', 'chhote',
    'naya', 'nayi', 'naye', 'purana', 'purani', 'purane',
    'achi', 'ache', 'bura', 'buri', 'bure',
    'waala', 'waali', 'waale',
}

def get_language_detector_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
        max_output_tokens=10,
    )

def detect_language_with_ai(text):
    """Use AI to detect if text is English or Hinglish"""
    try:
        detector = get_language_detector_llm()
        prompt = f"""Classify this text as either "English" or "Hinglish".
Hinglish = Hindi words written in English/Roman letters (like "kya hai", "kaise ho", "mujhe batao", "file upload kaise kare")
English = Pure English text

Text: "{text}"

Reply with ONLY one word - either "English" or "Hinglish":"""
        response = detector.invoke(prompt)
        result = response.content.strip().lower()
        if 'hinglish' in result or 'hindi' in result:
            return "Hinglish"
        return "English"
    except:
        return None

def detect_language(text):
    if not text: return "English"
    
    # Check for native scripts first (NO API call)
    if has_tamil_script(text): return "Tamil"
    if has_telugu_script(text): return "Telugu"
    if has_hindi_script(text): return "Hindi"
    
    # For Roman script: Check Hinglish words (NO API call)
    text_lower = text.lower()
    words = text_lower.replace(',', ' ').replace('.', ' ').replace('?', ' ').replace('!', ' ').split()
    hinglish_count = sum(1 for w in words if w in HINGLISH_WORDS)
    
    # If 1+ Hindi words found, double-check using the AI detector to confirm Hinglish
    if hinglish_count >= 1:
        ai_detected = detect_language_with_ai(text)
        if ai_detected:
            return ai_detected
        return "Hinglish"
    
    return "English"

# ==================== AUDIO FUNCTIONS ====================

# Domain-specific word corrections (misheard word -> correct word)
WORD_CORRECTIONS = {
    # Verifier variations
    'free fire': 'verifier', 'freefire': 'verifier', 'verify': 'verifier',
    'verified': 'verifier', 'verify her': 'verifier', 'verify year': 'verifier',
    'very fire': 'verifier', 'very fair': 'verifier', 'barrier': 'verifier',
    
    # Monitoring variations
    'monitor': 'monitoring', 'monitory': 'monitoring',
    
    # Baseline variations
    'base line': 'baseline', 'based line': 'baseline',
    'basin': 'baseline', 'base lying': 'baseline',
    
    # C-Trace variations
    'c trace': 'C-Trace', 'ctrace': 'C-Trace', 'sea trace': 'C-Trace',
    'see trace': 'C-Trace', 'c-trace': 'C-Trace',
    
    # Project related
    'carbon credit': 'carbon credits',
    'geo tagged': 'geotagged',
    'methodologies': 'methodology',
    
    # Common terms
    'dash board': 'dashboard',
    'up load': 'upload',
    'down load': 'download',
    'co-ordinator': 'coordinator',
}

def correct_transcription(text):
    """Fix common misheard domain words using word boundary matching to avoid corrupting substrings"""
    if not text:
        return text
    
    corrected = text
    for wrong, correct in WORD_CORRECTIONS.items():
        # Use word boundaries (\b) to match only complete words/phrases
        pattern = re.compile(rf'\b{re.escape(wrong)}\b', re.IGNORECASE)
        corrected = pattern.sub(correct, corrected)
    
    return corrected

def transcribe_audio(selected_lang="Auto Detect"):
    """Transcribe audio file to text based on the preferred language configuration"""
    audio_path = st.session_state.audio_file
    if not os.path.exists(audio_path):
        return "", "English"
        
    recognizer = sr.Recognizer()
    
    try:
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
    except Exception:
        return "", "English"
    
    results = []
    
    def try_lang(lang_code, weight):
        try:
            text = recognizer.recognize_google(audio, language=lang_code)
            if text and text.strip():
                return text.strip()
        except:
            pass
        return None

    if selected_lang == "English":
        # Check en-IN first (good for Indian English accent), fallback to en-US
        text = try_lang("en-IN", 1.0)
        if text:
            results.append(('en', text, 1.0))
        else:
            text = try_lang("en-US", 0.9)
            if text:
                results.append(('en', text, 0.9))
                
    elif selected_lang == "Hindi":
        text = try_lang("hi-IN", 1.0)
        if text:
            results.append(('hi', text, 1.0))
            
    elif selected_lang == "Tamil":
        text = try_lang("ta-IN", 1.0)
        if text:
            results.append(('ta', text, 1.0))
            
    elif selected_lang == "Telugu":
        text = try_lang("te-IN", 1.0)
        if text:
            results.append(('te', text, 1.0))
            
    elif selected_lang == "Hinglish":
        # Run en-IN and hi-IN for hybrid phrases
        text_en = try_lang("en-IN", 0.9)
        text_hi = try_lang("hi-IN", 0.8)
        if text_en:
            results.append(('en', text_en, 0.9))
        if text_hi:
            results.append(('hi', text_hi, 0.8))
            
    else:  # Auto Detect (fallback to checking all sequentially)
        try:
            text = recognizer.recognize_google(audio, language="en-US", show_all=True)
            if text and isinstance(text, dict) and 'alternative' in text:
                for alt in text['alternative'][:3]:
                    if 'transcript' in alt:
                        results.append(('en', alt['transcript'], alt.get('confidence', 0.5)))
            elif text and isinstance(text, str):
                results.append(('en', text, 0.5))
        except:
            pass
            
        text = try_lang("en-IN", 0.4)
        if text:
            results.append(('en', text, 0.4))
            
        text = try_lang("hi-IN", 0.4)
        if text:
            results.append(('hi', text, 0.4))
            
        text = try_lang("ta-IN", 0.3)
        if text:
            results.append(('ta', text, 0.3))
            
        text = try_lang("te-IN", 0.3)
        if text:
            results.append(('te', text, 0.3))

    if not results:
        return "", "English"
    
    # Sort by confidence/weight and pick best
    results.sort(key=lambda x: x[2], reverse=True)
    best_lang, best_text, _ = results[0]
    
    # Apply domain word corrections for English
    if best_lang == 'en':
        best_text = correct_transcription(best_text)
    
    # Detect actual language
    if selected_lang in ["English", "Hindi", "Tamil", "Telugu", "Hinglish"]:
        return best_text, selected_lang
    else:
        if best_lang == 'ta':
            return best_text, "Tamil"
        elif best_lang == 'te':
            return best_text, "Telugu"
        else:
            return best_text, detect_language(best_text)

def get_answer(question, lang):
    if qa_chain is None:
        return "Knowledge base not loaded."
    try:
        answer = qa_chain(question)
        return answer
    except Exception as e:
        return f"Error: {str(e)[:100]}"

def get_gtts_code(lang):
    return {"English": "en", "Hindi": "hi", "Hinglish": "hi", "Tamil": "ta", "Telugu": "te"}.get(lang, "en")

def generate_tts(text, lang):
    """Generate TTS. If pydub/ffmpeg is available, applies speedup and padding, otherwise returns raw gTTS audio."""
    tts = gTTS(text=text, lang=get_gtts_code(lang), slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    raw_audio = buf.getvalue()
    
    try:
        buf.seek(0)
        audio = AudioSegment.from_mp3(buf)
        silence = AudioSegment.silent(duration=500)  # 500ms silence at start
        audio = silence + audio
        
        # Speed up slightly
        faster = audio._spawn(audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * 1.25)}).set_frame_rate(audio.frame_rate)
        
        out = io.BytesIO()
        faster.export(out, format="mp3")
        return out.getvalue()
    except Exception:
        # Graceful fallback to raw gTTS output if ffmpeg or pydub fails
        return raw_audio

def play_audio(audio_bytes):
    """Auto-play audio"""
    b64 = base64.b64encode(audio_bytes).decode()
    components.html(f'<audio autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>', height=0)

# ==================== SESSION STATE ====================

if "audio_file" not in st.session_state:
    import tempfile
    import uuid
    st.session_state.audio_file = os.path.join(
        tempfile.gettempdir(), 
        f"temp_audio_{uuid.uuid4().hex}.wav"
    )

if "step" not in st.session_state:
    st.session_state.step = "idle"  # idle, listening, processing, done
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "response" not in st.session_state:
    st.session_state.response = ""
if "lang" not in st.session_state:
    st.session_state.lang = "English"
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None
if "audio_played" not in st.session_state:
    st.session_state.audio_played = False
if "retrieved_docs" not in st.session_state:
    st.session_state.retrieved_docs = []

# ==================== SIDEBAR ====================

with st.sidebar:
    st.markdown("### 🎙️ Voice AI")
    st.markdown("---")
    st.markdown("**Transcription Settings**")
    selected_language = st.selectbox(
        "Preferred Speaking Language",
        options=["Auto Detect", "English", "Hindi", "Hinglish", "Tamil", "Telugu"],
        index=0,
        help="Specifying your speaking language will significantly speed up responses."
    )
    st.markdown("---")
    st.markdown("**Vector Store**")
    if st.session_state.db_loaded:
        st.success("✅ Loaded successfully")
        if st.button("🔄 Rebuild Store", use_container_width=True):
            with st.spinner("Rebuilding store..."):
                try:
                    from ingest_data import create_vector_db
                    create_vector_db()
                    st.cache_resource.clear()
                    st.cache_data.clear()
                    st.success("✅ Store rebuilt!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.warning("❌ Not found")
        if st.button("📥 Load Vector Store", use_container_width=True, type="primary"):
            with st.spinner("Building store..."):
                try:
                    from ingest_data import create_vector_db
                    create_vector_db()
                    st.cache_resource.clear()
                    st.cache_data.clear()
                    st.success("✅ Store loaded!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    st.markdown("---")
    st.markdown("**How to use**")
    st.caption("1. Click 🎙️ Start")
    st.caption("2. Speak your question")
    st.caption("3. Stop speaking → auto-processes")
    st.caption("4. Listen to AI response")
    st.markdown("---")
    if st.button("🔄 Clear Cache", use_container_width=True):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

# ==================== MAIN UI ====================

st.markdown("<h1 style='text-align:center; color:#e2e8f0;'>🎙️ Voice AI Assistant</h1>", unsafe_allow_html=True)

if not st.session_state.db_loaded:
    st.warning("⚠️ Semantic search database (FAISS) not found or not loaded.")
    st.markdown("""
    To interact with your documents, the system needs to process your PDFs and create vector embeddings.
    
    ### 📥 Setup Instructions
    1. Place your PDF files in the **`Data files`** directory.
    2. Click the **"📥 Load Vector Store"** button in the sidebar (or the button below) to process the documents.
    """)
    
    if st.button("📥 Build Vector Store", use_container_width=True, type="primary"):
        with st.spinner("Ingesting PDFs and building FAISS database..."):
            try:
                from ingest_data import create_vector_db
                create_vector_db()
                st.cache_resource.clear()
                st.cache_data.clear()
                st.success("✅ FAISS database built successfully! Reloading...")
                st.rerun()
            except Exception as e:
                st.error(f"Ingestion failed: {e}")
    st.stop()

# Status display
if st.session_state.step == "idle":
    st.markdown("""
    <div class="status-card status-idle">
        <div class="status-icon">🎙️</div>
        <div class="status-title">Ready</div>
        <div class="status-subtitle">Click Start to speak</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🎙️ Start", use_container_width=True, type="primary"):
            st.session_state.step = "listening"
            st.session_state.transcript = ""
            st.session_state.response = ""
            st.session_state.retrieved_docs = []
            st.rerun()

elif st.session_state.step == "listening":
    st.markdown("""
    <div class="status-card status-listening">
        <div class="status-icon">🎤</div>
        <div class="status-title">Listening...</div>
        <div class="status-subtitle">Speak now! Click the stop button when done.</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Render browser-side recorder
    audio_base64 = audio_recorder(key="voice_input_recorder")
    
    if audio_base64:
        try:
            # Decode base64 audio data from browser
            audio_bytes = base64.b64decode(audio_base64)
            # Write to the unique session WAV file
            with open(st.session_state.audio_file, "wb") as f:
                f.write(audio_bytes)
            
            # Transcribe the saved file
            text, lang = transcribe_audio(selected_language)
            
            if text:
                st.session_state.transcript = text
                st.session_state.lang = lang
                st.session_state.step = "processing"
            else:
                st.warning("⚠️ Couldn't hear you. Try again.")
                st.session_state.step = "idle"
                time.sleep(2)
            st.rerun()
        except Exception as e:
            st.error(f"Error: {str(e)[:80]}")
            st.session_state.step = "idle"
            time.sleep(2)
            st.rerun()

elif st.session_state.step == "processing":
    st.markdown("""
    <div class="status-card status-processing">
        <div class="status-icon">⚡</div>
        <div class="status-title">Processing...</div>
        <div class="status-subtitle">Generating response</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show what was heard
    flag = LANGUAGES.get(st.session_state.lang, {}).get("flag", "🌐")
    st.markdown(f"""
    <div class="result-box">
        <div class="result-label">{flag} You said ({st.session_state.lang})</div>
        <div class="result-text">"{st.session_state.transcript}"</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Get answer
    answer = get_answer(st.session_state.transcript, st.session_state.lang)
    st.session_state.response = answer
    
    # Pre-generate TTS bytes inside processing
    st.session_state.audio_bytes = generate_tts(answer, st.session_state.lang)
    st.session_state.audio_played = False
    
    st.session_state.step = "done"
    st.rerun()

elif st.session_state.step == "done":
    st.markdown("""
    <div class="status-card status-speaking">
        <div class="status-icon">🔊</div>
        <div class="status-title">Done!</div>
        <div class="status-subtitle">Here's your answer</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show transcript
    flag = LANGUAGES.get(st.session_state.lang, {}).get("flag", "🌐")
    st.markdown(f"""
    <div class="result-box">
        <div class="result-label">{flag} You said ({st.session_state.lang})</div>
        <div class="result-text">"{st.session_state.transcript}"</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show response
    st.markdown(f"""
    <div class="result-box">
        <div class="result-label">🤖 AI Response</div>
        <div class="result-text">{st.session_state.response}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show source documents / citations
    if "retrieved_docs" in st.session_state and st.session_state.retrieved_docs:
        st.markdown("<h4 style='color:#e2e8f0; margin-top:20px;'>📄 Source Documents</h4>", unsafe_allow_html=True)
        for idx, doc in enumerate(st.session_state.retrieved_docs):
            page_num = doc.metadata.get("page", 0) + 1
            
            with st.expander(f"🔍 Source {idx+1} (Page {page_num})"):
                st.write(doc.page_content)
    
    # Play audio (once per state transition)
    if st.session_state.audio_bytes and not st.session_state.audio_played:
        play_audio(st.session_state.audio_bytes)
        st.session_state.audio_played = True
    
    # Button to ask again
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🎙️ Ask Again", use_container_width=True, type="primary"):
            st.session_state.step = "listening"
            st.session_state.transcript = ""
            st.session_state.response = ""
            st.session_state.retrieved_docs = []
            st.rerun()

# ==================== TEXT INPUT ====================

st.markdown("---")
st.caption("💬 Or type your question:")
text_input = st.chat_input("Type here...")

if text_input and text_input.strip():
    lang = detect_language(text_input)
    st.session_state.transcript = text_input
    st.session_state.lang = lang
    st.session_state.retrieved_docs = []
    st.session_state.step = "processing"
    st.rerun()