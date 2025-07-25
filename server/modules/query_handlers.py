from logger import logger

def query_chain(chain,user_input:str):
    try:
        logger.debug(f"Querying chain with input: {user_input} ")
        result=chain({"query": user_input})
        response={
            "response":result["result"],
            "sources":[doc.metadata.get("source") for doc in result["source_documents"]]
        }
        logger.debug(f"Chain Response: {response}")
        return response
    except Exception as e:
        logger.exception("Error on query chain")
        raise
