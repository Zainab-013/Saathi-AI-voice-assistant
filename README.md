# 🎤 VOICE-RAG System

A Voice-based Retrieval Augmented Generation (RAG) system that allows you to interact with your PDF documents using both text and voice input. Powered by **HuggingFace's all-MiniLM-L6-v2** embeddings and **Google's Gemini 2.5** language model.

## 🌟 Features

- **Voice Input**: Ask questions using your microphone
- **Text Input**: Type your queries for document retrieval
- **Voice Output**: Get answers read back to you with text-to-speech
- **PDF Processing**: Automatically process and index PDF documents
- **Semantic Search**: Uses HuggingFace embeddings for accurate document retrieval
- **Gemini 2.5 Integration**: Leverages Google's latest AI model for intelligent responses
- **Interactive UI**: Clean Streamlit interface with chat history

## 📁 Project Structure

```
VOICE-RAG/
├── Data files/              # Place your PDF documents here
│   ├── Additional-Info.pdf
│   └── C-Trace-User.pdf
├── vectorstore/             # Auto-generated vector embeddings (created after ingestion)
├── venv1/                   # Virtual environment (create this)
├── .env                     # Environment variables (API keys)
├── .gitignore              # Git ignore patterns
├── ingest_data.py          # PDF processing and vector store creation
├── streamlit_app.py        # Main Streamlit application
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Setup Instructions

### 1. Prerequisites

- Python 3.8 or higher
- Microphone (for voice input)
- Google Gemini API key

### 2. Get Your Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the API key for the next step

### 3. Installation

#### Step 1: Create Virtual Environment

```powershell
# Navigate to project directory
cd "c:\Users\Zainab\OneDrive\Documents\Desktop\Saathi\VOICE-RAG"

# Create virtual environment
python -m venv venv1

# Activate virtual environment
.\venv1\Scripts\Activate.ps1
```

#### Step 2: Install Dependencies

```powershell
pip install -r requirements.txt
```

**Note**: If you encounter issues with `pyaudio`, you may need to install it separately:

```powershell
pip install pipwin
pipwin install pyaudio
```

#### Step 3: Configure Environment Variables

Open the `.env` file and add your Gemini API key:

```env
GOOGLE_API_KEY=your_actual_api_key_here
```

### 4. Add Your PDF Documents

Place your PDF files in the `Data files` folder. The system will automatically process all PDFs in this directory.

### 5. Create Vector Store

Run the data ingestion script to process your PDFs and create embeddings:

```powershell
python ingest_data.py
```

This will:
- Load all PDF files from `Data files/`
- Split documents into chunks
- Create vector embeddings using HuggingFace all-MiniLM-L6-v2
- Save the vector store to `vectorstore/`

### 6. Run the Application

```powershell
streamlit run streamlit_app.py
```

The application will open in your default web browser (usually at `http://localhost:8501`).

## 📖 Usage Guide

### First Time Setup

1. Click **"📥 Load Vector Store"** in the sidebar
2. Wait for the vector store and QA chain to initialize
3. You're ready to ask questions!

### Asking Questions

#### Text Input
1. Click on the **"💬 Text Input"** tab
2. Type your question in the input box
3. Click **"Submit"**
4. View the response in the conversation history

#### Voice Input
1. Click on the **"🎤 Voice Input"** tab
2. Click **"🎤 Start Recording"**
3. Speak your question clearly
4. The system will:
   - Transcribe your speech
   - Process the query
   - Display the answer
   - Play the audio response

### Managing Conversations

- View all previous conversations in the **"💬 Conversation History"** section
- Click **"🗑️ Clear History"** in the sidebar to reset conversations

## 🛠️ Technical Details

### Components

**Embeddings**:
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Dimension: 384
- Optimized for semantic similarity

**Vector Store**:
- FAISS (Facebook AI Similarity Search)
- Fast and efficient similarity search

**Language Model**:
- Google Gemini 2.5 (gemini-2.0-flash-exp)
- Temperature: 0.3 (balanced creativity)

**Speech Recognition**:
- Google Speech Recognition API
- Supports multiple languages

**Text-to-Speech**:
- Google Text-to-Speech (gTTS)
- Natural voice synthesis

### Document Processing

- **Chunk Size**: 500 characters
- **Chunk Overlap**: 50 characters
- **Retrieval**: Top 3 most relevant chunks

## 🔧 Troubleshooting

### Issue: "Vector store not found"
**Solution**: Run `python ingest_data.py` to create the vector store

### Issue: "GOOGLE_API_KEY not found"
**Solution**: Make sure you've added your API key to the `.env` file

### Issue: Microphone not working
**Solution**: 
- Check microphone permissions in Windows Settings
- Ensure pyaudio is properly installed
- Try restarting the application

### Issue: "No module named 'pyaudio'"
**Solution**:
```powershell
pip install pipwin
pipwin install pyaudio
```

### Issue: Import errors with langchain
**Solution**: Make sure all dependencies are installed:
```powershell
pip install -r requirements.txt --upgrade
```

## 📝 Adding New Documents

To add new PDF documents:

1. Place new PDFs in the `Data files/` folder
2. Re-run the ingestion script:
   ```powershell
   python ingest_data.py
   ```
3. Restart the Streamlit app or click "Load Vector Store" again

## 🔐 Security Notes

- Never commit your `.env` file with actual API keys
- Keep your `GOOGLE_API_KEY` private
- The `.gitignore` file is configured to exclude sensitive files

## 📚 Dependencies

Key libraries used:
- `streamlit`: Web interface
- `langchain`: RAG framework
- `sentence-transformers`: Embeddings
- `faiss-cpu`: Vector database
- `google-generativeai`: Gemini API
- `SpeechRecognition`: Voice input
- `gTTS`: Text-to-speech
- `pypdf`: PDF processing

## 🤝 Contributing

Feel free to enhance this project by:
- Adding support for more document types
- Improving the UI/UX
- Adding more language support
- Optimizing performance

## 📄 License

This project is open-source and available for educational purposes.

## 🆘 Support

For issues or questions:
1. Check the Troubleshooting section
2. Verify all setup steps were completed
3. Ensure all dependencies are installed correctly

---

**Built with ❤️ using HuggingFace & Google Gemini**
