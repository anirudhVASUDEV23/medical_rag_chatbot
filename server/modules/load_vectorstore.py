import os
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
import voyageai

load_dotenv()

# Pinecone config
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAME = "medicalindex"

UPLOAD_DIR = "./uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ------------------- VoyageAI Embeddings Class -------------------
class VoyageAIEmbeddings(Embeddings):
    def __init__(self, model_name="voyage-3.5-lite", device="cpu"):
        self.client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
        self.model_name = model_name
        self.device = device

    def embed_documents(self, texts):
        # Pass texts as positional argument, not with "inputs="
        response = self.client.embed(
            texts,
            model=self.model_name,
            input_type="document"  # optional
        )
        return [res["embedding"] for res in response["results"]]

    def embed_query(self, query):
        response = self.client.embed(
            [query],
            model=self.model_name,
            input_type="query"  # optional
        )
        return response["results"][0]["embedding"]

# ------------------- Pinecone Setup -------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)
existing_indexes = [i["name"] for i in pc.list_indexes()]

if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=512,  # Voyage AI voyage-3.5-lite embeddings dimension
        metric="dotproduct",
        spec=spec
    )
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)

index = pc.Index(PINECONE_INDEX_NAME)

# ------------------- Load PDFs and Embed -------------------
def load_vectorstore(uploaded_files):
    embed_model = VoyageAIEmbeddings(model_name="voyage-3.5-lite", device="cpu")

    file_paths = []
    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR) / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())
        file_paths.append(str(save_path))

    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        batch_size = 50
        for i in tqdm(range(0, len(chunks), batch_size), desc=f"Processing {Path(file_path).name}"):
            batch_chunks = chunks[i:i + batch_size]
            texts = [chunk.page_content for chunk in batch_chunks]
            metadatas = [chunk.metadata for chunk in batch_chunks]
            ids = [f"{Path(file_path).stem}-{i+j}" for j in range(len(batch_chunks))]

            print(f"🔍 Embedding batch of {len(texts)} chunks...")
            embeddings = embed_model.embed_documents(texts)

            print("📤 Uploading batch to Pinecone...")
            index.upsert(vectors=zip(ids, embeddings, metadatas))

        print(f"✅ Upload complete for {file_path}")
