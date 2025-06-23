from prompts import doc_summary
from langchain_openai import ChatOpenAI
import logging
from functools import lru_cache

logger=logging.getLogger(__name__)

def summary_document(final_docs, llm=None):
    """
    Summarises all the chunks and merge them into a single document to answer document level questions
    parameters:
        llm: a model to summarise chunks
        final_docs: list of all the chunks
    """
    if llm is None:
        llm = ChatOpenAI()
    summary_prompt= doc_summary()
    logger.debug("summary prompt loaded.")
    summary_chain=summary_prompt | llm
    try:

        summarised_docs=[summary_chain.invoke({'text': doc.page_content}) for doc in final_docs]
        single_doc="\n".join([doc.content.replace("Summary:", "").replace("\n", "") for doc in summarised_docs]) 
        logger.debug(f"Summary document created. Sample: {single_doc[:150]}")
        return single_doc
    except Exception as e:
        logger.critical(f"Couldn't merge documents to answer document level question: {str(e)}", exc_info=True)
        raise e
    
    

#caching the document
@lru_cache(maxsize=1)
def get_document_summary(docs, llm=None):
    """
    Summarises all the chunks and merge them into a single document to answer document level questions
    parameters:
        llm: a model to summarise chunks
        final_docs: list of all the chunks
    """
    if llm is None:
        llm = ChatOpenAI()
    return summary_document(docs,llm)