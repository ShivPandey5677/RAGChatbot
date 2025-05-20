# 🧠 RAG Q&A Chatbot with Smart Fallback

This project is a Retrieval-Augmented Generation (RAG) based chatbot developed as part of the AI Internship Assignment from BuildFastWithAI. The chatbot is capable of answering user queries by retrieving relevant context from uploaded documents and using a large language model (LLM) for response generation. It also includes smart fallbacks for dictionary lookups and math evaluations.

---

## 🚀 Features

- ✅ **Retrieval-Augmented Generation (RAG)** using LangChain
- ✅ **Document-based context** loading, chunking, and retrieval using FAISS
- ✅ **Meta LLaMA-3.1 8B Instruct LLM** via Hugging Face Hub
- ✅ **Smart fallback logic**:
  - 📖 Dictionary lookup (via dictionaryapi.dev)
  - 🔢 Math expression evaluation (via mathjs API)
  - 🤖 General fallback LLM-based answer (if RAG context is insufficient)
- ✅ **Streamlit Chat UI** with session memory and logs

---

## 📂 Folder Structure

```
.
├── app1.py                # Main chatbot application
├── docs/                  # Folder to place `.txt` documents used as knowledge base
├── requirements.txt       # Dependencies for the app
├── sampleques.txt        # Sample Q&A responses (to be created)
└── README.md              # This file
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/ShivPandey5677/RAGChatbot
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables

Create a `.env` file with your Hugging Face API key:

```env
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
```

---

## ▶️ Running the App

```bash
streamlit run app1.py
```

Upload `./docs` folder before starting the app to build the knowledge base.

---

## 📌 How It Works

1. **Data Loading** (Task 1):

   - Loads `.txt` files from the `/docs` folder using `TextLoader`.
   - Splits text into overlapping chunks with `RecursiveCharacterTextSplitter`.

2. **RAG Setup with LangChain** (Task 2):

   - Uses `HuggingFaceEmbeddings` to vectorize document chunks.
   - Uses FAISS for efficient similarity search.
   - Wraps the LLM with `RetrievalQA` and `LLMChain`.

3. **Chatbot Functionality** (Task 3):
   - Accepts user input and runs retrieval.
   - If no relevant info is found, falls back to:
     - Dictionary lookup
     - Math calculation
     - Fallback LLM prompt

---

## 💬 Sample Questions and Responses

> **Q:** What are you company's services?  
> **A:** (from document)

> **Q:** explain yield curve?  
> **A:** (from fallback prompt)

> **Q:** Define recursion  
> **A:** **recursion**: The repeated application of a recursive procedure.

> **Q:** Calculate 5 _ (3 + 2)  
> **A:** The result of `5 _ (3 + 2)` is 25.

## 🌐 Deployment

```bash
https://rag-q-a-chatbot.streamlit.app/
```

Include the deployed app link in your assignment submission email.

---

## 📚 References

- [LangChain Documentation](https://docs.langchain.com/)
- [Hugging Face Hub](https://huggingface.co/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Dictionary API](https://dictionaryapi.dev/)
- [Math.js API](https://api.mathjs.org/)

---
