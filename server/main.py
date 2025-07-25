from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware##backend and frontend can communicate with each other as they are on different ports one on render other on streamlit
from middlewares.exception_handlers import catch_exception_middleware
from routes.upload_pdfs import router as upload_router
from routes.ask_question import router as ask_router

app=FastAPI(title="Medical Assistant API",description="API for Medical Assistant Chatbot")

##CORS  Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/",tags=["healthcheck"])
async def root():
    return {"message": "Hello World"}


##middleware exception handlers
app.middleware("http")(catch_exception_middleware)


#routers

#1.Upload pdf document
app.include_router(upload_router)
#2.Asking Query
app.include_router(ask_router)

