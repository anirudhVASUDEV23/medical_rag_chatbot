import os
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
PINECONE_ENV="us-east-1"
PINECONE_INDEX_NAME="medicalindex"

UPLOAD_DIR="./uploaded_docs"
os.makedirs(UPLOAD_DIR,exist_ok=True)

# initialize pinecone instance
pc=Pinecone(api_key=PINECONE_API_KEY)
spec=ServerlessSpec(cloud="aws",region=PINECONE_ENV)
existing_indexes=[i["name"] for i in pc.list_indexes()]

if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=768,
        metric="dotproduct",
        spec=spec
    )
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)

index=pc.Index(PINECONE_INDEX_NAME)

# load,split,embed and upsert pdf docs content
def load_vectorstore(uploaded_files):
    embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
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

        # --- BATCHING LOGIC STARTS HERE ---
        batch_size = 50  # Process 50 chunks per batch to stay under the limit
        for i in tqdm(range(0, len(chunks), batch_size), desc=f"Processing {Path(file_path).name}"):
            # Get the next batch of chunks
            batch_chunks = chunks[i:i + batch_size]
            
            # Prepare data for this batch
            texts = [chunk.page_content for chunk in batch_chunks]
            metadatas = [chunk.metadata for chunk in batch_chunks]
            # Make sure IDs are unique for each chunk in the batch
            ids = [f"{Path(file_path).stem}-{i+j}" for j in range(len(batch_chunks))]

            print(f"🔍 Embedding batch of {len(texts)} chunks...")
            embeddings = embed_model.embed_documents(texts)

            print("📤 Uploading batch to Pinecone...")
            index.upsert(vectors=zip(ids, embeddings, metadatas))

            # --- THE CRITICAL PAUSE ---
            # If there are more chunks to process, wait for a second
            if i + batch_size < len(chunks):
                print("⏳ Waiting 1 second to respect API rate limit...")
                time.sleep(1) 

        print(f"✅ Upload complete for {file_path}")
