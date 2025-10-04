from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from modules.llm import get_llm_chain
from modules.query_handlers import query_chain
from langchain_core.documents import Document
from langchain.schema import BaseRetriever
from pinecone import Pinecone
from pydantic import Field
from typing import List, Optional
from logger import logger
import os
import voyageai
from langchain.embeddings.base import Embeddings

router = APIRouter()

# ------------------- Corrected VoyageAI Embeddings Class -------------------
class VoyageAIEmbeddings(Embeddings):
    def __init__(self, model_name="voyage-3.5-lite", device="cpu"):
        self.client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
        self.model_name = model_name
        self.device = device

    def embed_documents(self, texts):
        response = self.client.embed(
            texts,                 # positional argument
            model=self.model_name,
            input_type="document"
        )
        # Access the embeddings via the .embeddings attribute
        return response.embeddings

    def embed_query(self, query):
        response = self.client.embed(
            [query],
            model=self.model_name,
            input_type="query"
        )
        return response.embeddings[0]


# ------------------- FastAPI Route -------------------
@router.post("/ask/")
async def ask_question(question: str = Form(...)):
    try:
        logger.info(f"user query: {question}")

        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

        if not all([PINECONE_API_KEY, PINECONE_INDEX_NAME]):
            raise ValueError("API keys or index name not found in environment variables.")

        # Pinecone index
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)

        # Instantiate VoyageAI embeddings
        embed_model = VoyageAIEmbeddings(model_name="voyage-3.5-lite", device="cpu")
        embedded_query = embed_model.embed_query(question)
        res = index.query(vector=embedded_query, top_k=3, include_metadata=True)

        docs = [
            Document(
                page_content=match["metadata"].get("text", ""),
                metadata=match["metadata"]
            )
            for match in res["matches"]
        ]

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
