import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from .prompt_templates import rag_prompt
from langchain_community.tools.tavily_search import TavilySearchResults

# Get env variables
index_name = os.environ["index_name"]
model = os.environ["model"]

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})
tavily_search = TavilySearchResults(max_results=3)

llm = ChatGoogleGenerativeAI(model=model, temperature=0.3, max_tokens=1024)

rag_chain = rag_prompt | llm

__all__ = ["llm", "retriever", "rag_chain", "tavily_search"]
