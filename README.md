# 🎤 Saathi AI Voice Assistant (Voice-RAG)

A voice-based Retrieval-Augmented Generation (RAG) system that allows you to interact with your PDF documents using both text and voice input. Powered by **HuggingFace's all-MiniLM-L6-v2** embeddings, **FAISS**, and **Google's Gemini 2.5** language model (`gemini-2.5-flash`).

Saathi uses a fully browser-native custom audio recorder, making it lightweight and deployable anywhere without requiring complex server-side audio libraries (like PyAudio or sounddevice).

---

## 🌟 Features

- **Native Web Audio Recording**: Capture voice directly in the browser using a custom HTML5 / Web Audio API component. Works out-of-the-box on remote servers or local machines.
- **🎨 Dynamic Audio Waveform Visualizer**: A smooth, Siri-like oscilloscope rendering 3 overlapping animated sine waves (Sky Blue, Indigo, and Rose) that dynamically pulse and taper to the volume of your voice.
- **📄 Source Citations & PDF Chunk Viewer**: Verify answers with collapsible accordions (`st.expander`) displaying the exact retrieved context chunks, filenames, page numbers, and text snippets.
- **Smart Language Detection**: Automatically detects English vs. Hinglish inputs using a combination of fast regex and Google Gemini validation to respond in the appropriate synthesized dialect.
- **Efficient TTS Cache & Playback**: Text-to-speech outputs are generated and cached per prompt session. Audio plays exactly once per prompt, resolving loop playback issues during page reruns.
- **Modern SDK**: Fully migrated to the modern, stable `google-genai` SDK.

---

## 📁 Project Structure

```
Saathi/
├── .streamlit/
│   └── config.toml          # Streamlit UI configuration (bypasses file watcher issues)
├── Data files/              # Place your PDF documents here
│   ├── Additional-Info.pdf
│   └── C-Trace-User.pdf
├── audio_recorder_component/# Browser-side custom component files
│   ├── __init__.py          # Component declaration
│   └── index.html           # HTML5 recorder, WAV encoder & Siri-like visualizer
├── vectorstore/             # Auto-generated vector database (created after ingestion)
├── venv1/                   # Python virtual environment folder
├── .env                     # API key configuration
├── .gitignore              # Files excluded from git tracking
├── ingest_data.py          # PDF document loader, splitter, and vector index creator
├── streamlit_app.py        # Main Streamlit application with voice chat UI
├── requirements.txt        # Minimal dependency file
└── README.md              # Project documentation
```

---

## 🚀 Setup Instructions

### 1. Prerequisites

- Python 3.8 or higher
- A modern web browser with microphone access (Chrome, Firefox, Edge, etc.)
- A Google Gemini API key

### 2. Obtain a Gemini API Key

1. Visit [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Click **Create API Key** and copy it.

### 3. Installation

#### Step 1: Clone and Navigate to the Project Folder

Open your PowerShell (or terminal) and navigate to the project directory:

```powershell
cd "c:\Users\Zainab\OneDrive\Documents\Desktop\ZDesktop\Saathi"
```

#### Step 2: Create and Activate Virtual Environment

```powershell
# Create the virtual environment
python -m venv venv1

# Activate the virtual environment
.\venv1\Scripts\Activate.ps1
```

#### Step 3: Install Required Dependencies

```powershell
pip install -r requirements.txt
```

*(No need for complex setups like `pyaudio` or `sounddevice`! All recording is handled browser-side).*

#### Step 4: Configure the API Key

Create or edit the `.env` file in the root folder and paste your Gemini API key:

```env
GOOGLE_API_KEY=your_actual_api_key_here
```

---

## 📖 Usage Guide

### 1. Load your PDFs

Place any PDF files you want to index into the `Data files/` directory.

### 2. Ingest Data (Build Vector Store)

Run the data ingestion script to process your PDFs, chunk the text, and build the local vector database:

```powershell
python ingest_data.py
```

*Alternatively, you can skip this command and use the **"Rebuild Store"** button directly inside the app's sidebar.*

### 3. Run the App

Launch the Streamlit dashboard:

```powershell
.\venv1\Scripts\streamlit.exe run streamlit_app.py
```

Open the local URL displayed (usually `http://localhost:8501`) in your browser.

---

## 💬 Interacting with Saathi

- **Vector Store Status**: The sidebar shows if your database index is loaded. If it says `❌ Not found`, click **Rebuild Store** in the sidebar or run the ingestion script.
- **Voice Tab**:
  1. Click **🎙️** to begin recording.
  2. Grant the browser microphone permission when prompted.
  3. Speak into the microphone. You will see the Siri-style color wave visualize your voice levels.
  4. Click **⏹️** (Stop) when you are done. The audio will automatically upload, transcribe, query the vector store, display the response with references, and speak the answer back to you.
- **Text Tab**: Type your question directly in the text input box and click **Submit**.

---

## 🛠️ Technical Details

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Language Model**: Google Gemini 2.5 (`gemini-2.5-flash`) via the modern `google-genai` SDK.
- **Speech Recognition**: Google Speech Recognition API (handles translation and transcriptions).
- **Text-to-Speech**: Google Text-to-Speech (`gTTS`) to synthesize response audio.

---

## 🔧 Troubleshooting

### Issue: Browser does not start recording
* **Solution**: Ensure you granted microphone access in your browser's address bar. Click the lock icon next to the URL, verify "Microphone" is set to "Allow", and refresh the page.

### Issue: "Vector store not found"
* **Solution**: Place PDFs in the `Data files/` directory and click the **Rebuild Store** button in the sidebar, or run `python ingest_data.py` in your terminal.

### Issue: "GOOGLE_API_KEY not found"
* **Solution**: Ensure you have created a `.env` file containing `GOOGLE_API_KEY=your_key` in the root folder of the project.

---

**Built with ❤️ by Zainab**
