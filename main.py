import os
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 1. Load and index legal documents
def load_and_index_documents(data_dir):
    docs = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_dir, filename))
            docs.extend(loader.load())
    # Split into chunks for better retrieval
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    # Create vector store
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = FAISS.from_documents(split_docs, embeddings)
    return vectordb

# 2. Query Agent: Fetch relevant legal sections
def query_agent(user_query, vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    return retriever.get_relevant_documents(user_query)

# 3. Summarization Agent: Simplify and summarize
def summarization_agent(docs, user_query):
    llm = OpenAI(temperature=0.2, openai_api_key=OPENAI_API_KEY)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = (
        f"You are a legal assistant. Summarize the following legal text in simple, clear language for a layperson. "
        f"Focus on answering: '{user_query}'.\n\n"
        f"Legal Text:\n{context}\n\n"
        f"Summary:"
    )
    return llm(prompt)

# 4. Main chatbot loop
def chatbot():
    vectordb = load_and_index_documents(r"E:\qest\lega_chatbot\data")
    print("Legal Chatbot Ready. Ask your legal question!")
    while True:
        user_query = input("\nUser: ")
        if user_query.lower() in ["exit", "quit"]:
            break
        # Query Agent
        relevant_docs = query_agent(user_query, vectordb)
        # Summarization Agent
        answer = summarization_agent(relevant_docs, user_query)
        print(f"\nBot: {answer}")

if __name__ == "__main__":
    chatbot()
