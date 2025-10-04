from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from modules.llm import get_llm_chain
from modules.query_handlers import query_chain
from langchain_core.documents import Document
from langchain.schema import BaseRetriever
# --- CHANGED: Import the Google Generative AI Embeddings model ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from pydantic import Field
from typing import List, Optional
from logger import logger
import os

router=APIRouter()

@router.post("/ask/")
async def ask_question(question: str = Form(...)):
    try:
        logger.info(f"user query: {question}")

        # --- CHANGED: Get Google API Key instead of Hugging Face Token ---
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
        
        # Check if environment variables are loaded
        if not all([GOOGLE_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME]):
            raise ValueError("API keys or index name not found in environment variables.")

        # Embed model + Pinecone setup
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # --- CHANGED: Use the Google Generative AI model ---
        embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        embedded_query = embed_model.embed_query(question)
        res = index.query(vector=embedded_query, top_k=3, include_metadata=True)

        docs = [
            Document(
                page_content=match["metadata"].get("text", ""),
                metadata=match["metadata"]
            ) for match in res["matches"]
        ]

        # ... rest of your retriever and chain logic remains the same ...
        class SimpleRetriever(BaseRetriever):
            tags: Optional[List[str]] = Field(default_factory=list)
            metadata: Optional[dict] = Field(default_factory=dict)

            def __init__(self, documents: List[Document]):
                super().__init__()
                self._docs = documents

            def _get_relevant_documents(self, query: str) -> List[Document]:
                return self._docs

        retriever = SimpleRetriever(docs)
        chain = get_llm_chain(retriever)
        result = query_chain(chain, question)

        logger.info("query successful")
        return result

    except Exception as e:
        logger.exception("Error processing question")
        return JSONResponse(status_code=500, content={"error": str(e)})
