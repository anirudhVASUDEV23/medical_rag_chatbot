from fastapi import APIRouter,UploadFile,File
from typing import List
from modules.load_vectorstore import load_vectorstore
from fastapi.responses import JSONResponse
from logger import logger

router=APIRouter()

@router.post("/upload_pdfs/")
async def upload_pdfs(files:List[UploadFile]=File(...)):
   
    try:
        logger.info("Received uploaded files")
        load_vectorstore(files)
        logger.info("Document added to vector store")
        return {"message":"Files processed and vector store is updated"}

    
    except Exception as e:
        logger.exception("Error on uploading pdfs")
        return JSONResponse(status_code=500, content={"error": str(e)})