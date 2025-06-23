import re
from langchain_core.documents import Document
import logging
from prompts import doc_summary
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

def split_docs(complete_pdf):
    """
    Splits PDF into smaller chunks.
    Parameters:
        complete pdf file
    output:
        smaller chunks of the PDF
    """
    summary=""

    chunks=re.split(r"(?=\n?[A-Z][a-zA-Z\s\.]*?:)", complete_pdf.page_content)
    #splitting at <name>:
    import os
    doc_summary_path = 'doc_summary.txt'
    final_docs=[]
    for doc in chunks:
        if not doc or ":" not in doc:
            continue
        doc=re.sub(r"\s+", " ", doc)
        index_colon=doc.find(":") if doc.find(":")!=-1 else -1
        final_docs.append(Document(page_content=doc[index_colon+1:], metadata={'speaker_name':doc[:index_colon]}))
        
        if os.path.isfile(doc_summary_path) and os.path.getsize(doc_summary_path) > 0:
            logger.info("Doc summary exists, not generating again.")
        else:
            #summarizing docs and adding them into a single document to answer document level questions
            llm=ChatOpenAI()
            summary_prompt= doc_summary()
            parser=StrOutputParser()
            summary_chain=summary_prompt | llm | parser
            try:
                summary=summary + " " + summary_chain.invoke({'text': doc[index_colon+1:]})
            except Exception as e:
                logger.critical(f"Couldn't summarize individual chunks: {str(e)}")
            
            try:
                with open(doc_summary_path, 'w', encoding='utf-8') as f:
                    f.write(summary)
                logger.info("Summary document saved to doc_summary.txt. sample: ")
            except Exception as e:
                logger.error(f"Couldn't save summar document: {str(e)}", exc_info=True)

    logger.info(f"Documents split done. Total chunks created: {len(final_docs)}")
    
    return final_docs