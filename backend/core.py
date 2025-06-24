import os
from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

INDEX_NAME = "langchain-doc-index"

def run_llm(query: str, chat_history: list = None):
    if chat_history is None:
        chat_history = []

    google_api_key = os.environ["GOOGLE_API_KEY"]

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_api_key
    )

    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    chat = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash",
        temperature=0.2,
        google_api_key=google_api_key,
        verbose=True
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    qa = create_retrieval_chain(
        retriever=docsearch.as_retriever(), combine_docs_chain=stuff_documents_chain
    )

    result = qa.invoke(input={"input": query, "chat_history": chat_history})

    new_result = {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"]
    }

    return new_result


if __name__ == "__main__":
    res = run_llm(query="What is a LangChain Chain?")
    print(res["result"])
