import os
import streamlit as st
import warnings
import requests
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langchain.schema import Document

warnings.filterwarnings("ignore")
load_dotenv()

# os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
else:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# --- Load documents ---
def load_documents(folder="./docs"):
    documents = []
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(folder, file), encoding="utf-8")
            documents.extend(loader.load())
    return documents

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(documents)

def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)

def define_term(query):
    term = query.lower().replace("define", "").replace("what is", "").replace("what do you mean by", "").strip()
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{term}"
    try:
        response = requests.get(url)
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            definition = data[0]['meanings'][0]['definitions'][0]['definition']
            return f"**{term}**: {definition}"
        else:
            return f"No definition found for '{term}'."
    except Exception as e:
        return f"Error fetching definition: {e}"

def calculate_expression(query):
    expression = query.lower().replace("calculate", "").strip()
    url = f"https://api.mathjs.org/v4/?expr={requests.utils.quote(expression)}"
    try:
        response = requests.get(url)
        return f"The result of `{expression}` is {response.text}" if response.status_code == 200 else "Could not evaluate the expression."
    except Exception as e:
        return f"Error: {e}"

rag_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant. Use the following context to answer the question.
Only use the information in the context. If the answer is unclear, make your best guess based on the context.

Context:
{context}

Question:
{question}

Answer:
"""
)

fallback_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are a knowledgeable assistant. The user asked:
"{question}"

You don't have access to external documents now.
Provide the best possible answer based on your general knowledge.
If you do not know the answer, say "I'm sorry, I don't know the answer."
"""
)
# meta-llama/Llama-3.1-8B-Instruct
def get_llm():
    return HuggingFaceHub(
        repo_id="deepseek-ai/DeepSeek-R1",
        huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        model_kwargs={"temperature": 0.7, "max_new_tokens": 512}
    )

def extract_related_questions(docs, max_suggestions=3):
    questions = []
    for doc in docs:
        lines = doc.page_content.split('\n')
        for line in lines:
            line = line.strip()
            if line.endswith('?'):
                questions.append(line)
            if len(questions) >= max_suggestions:
                break
        if len(questions) >= max_suggestions:
            break
    return questions

def main():
    st.set_page_config(page_title="RAG Q&A Chatbot", layout="wide")
    st.markdown("""
        <h2 style='text-align: center;'>🧠 RAG Q&A Chatbot with Smart Fallback</h2>
        <p style='text-align: center;'>Ask questions and get answers from your documents or general knowledge.</p>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.title("📚 About")
        st.markdown("""
        This chatbot uses:
        - ✅ FAISS vector search
        - ✅ Meta LLaMA-3 LLM
        - ✅ Smart fallback answering
        - ✅ Dictionary & Calculator fallback
        """)
        st.markdown("---")
        st.markdown("📂 Put `.txt` files in the `./docs` folder")

    if "vectorstore" not in st.session_state:
        documents = load_documents()
        if not documents:
            st.error("No `.txt` documents found in ./docs.")
            return
        chunks = chunk_documents(documents)
        st.session_state.vectorstore = create_vector_store(chunks)
        st.session_state.logs = []

    user_query = st.chat_input("Ask me anything...")

    if user_query:
        query_lower = user_query.lower()
        decision = "RAG"

        if "define" in query_lower:
            answer = define_term(user_query)
            decision = "Dictionary"
        elif "calculate" in query_lower:
            answer = calculate_expression(user_query)
            decision = "Calculator"
        else:
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
            rag_chain = RetrievalQA.from_chain_type(
                llm=get_llm(),
                retriever=retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": rag_prompt},
                return_source_documents=True,
            )
            result = rag_chain(user_query)
            rag_answer = result["result"].strip()
            context = "\n\n".join([doc.page_content for doc in result["source_documents"]])
            source_docs = result["source_documents"]

            fallback_phrases = ["i couldn't find the answer", "i could not find the answer"]
            if rag_answer.lower() in fallback_phrases:
                fallback_chain = LLMChain(llm=get_llm(), prompt=fallback_prompt)
                fallback_answer = fallback_chain.run(question=user_query).strip()

                if fallback_answer.lower().startswith("i'm sorry") or len(fallback_answer) < 5:
                    if query_lower.startswith(("what is", "define")):
                        answer = define_term(user_query)
                        decision = "Dictionary (fallback)"
                    else:
                        answer = "Sorry, I couldn't find an answer to your question."
                        decision = "No answer found"
                else:
                    answer = fallback_answer
                    decision = "LLM fallback (no context)"
                related_questions = extract_related_questions(source_docs)
            else:
                answer = rag_answer
                decision = "RAG"
                related_questions = extract_related_questions(source_docs)

        st.session_state.logs.append({"query": user_query, "decision": decision})

        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            st.markdown(answer)
            if related_questions:
                st.markdown("### 💡 Related questions you might try:")
                for q in related_questions:
                    st.markdown(f"- {q}")

        with st.expander("🔍 Retrieved Context"):
            st.write(context if decision == "RAG" else "No documents used.")

        with st.expander("📝 Chat Log"):
            for log in st.session_state.logs:
                st.markdown(f"- **Query:** {log['query']} | **Decision:** {log['decision']}")

if __name__ == "__main__":
    main()