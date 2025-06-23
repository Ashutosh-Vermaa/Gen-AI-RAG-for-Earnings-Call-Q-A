import re
from langchain_core.documents import Document
import logging
from prompts import doc_summary
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

logger = logging.getLogger(__name__)

def split_docs_semantic(complete_pdf):
    """
    Splits PDF into smaller chunks.
    Parameters:
        complete pdf file
    output:
        smaller chunks of the PDF
    """
    summary=""

    text_splitter = SemanticChunker(OpenAIEmbeddings(model='text-embedding-3-small')
                                    )
    chunks = text_splitter.create_documents([complete_pdf.page_content])

    print("Right after the chunks:", len(chunks))
    # chunks=re.split(r"(?=\n?[A-Z][a-zA-Z\s\.]*?:)", complete_pdf.page_content)
    #splitting at <name>:
    import os
    doc_summary_path = 'doc_summary.txt'
    for doc in chunks:
        if os.path.isfile(doc_summary_path) and os.path.getsize(doc_summary_path) > 0:
            logger.info("Doc summary exists, not generating again.")
            break
        else:
            #summarizing docs and adding them into a single document to answer document level questions
            llm=ChatOpenAI()
            summary_prompt= doc_summary()
            parser=StrOutputParser()
            summary_chain=summary_prompt | llm | parser
            try:
                summary=summary + " " + summary_chain.invoke({'text': doc})
            except Exception as e:
                logger.critical(f"Couldn't summarize individual chunks: {str(e)}")
            
            try:
                with open(doc_summary_path, 'w', encoding='utf-8') as f:
                    f.write(summary)
                logger.info("Summary document saved to doc_summary.txt. sample: ")
            except Exception as e:
                logger.error(f"Couldn't save summar document: {str(e)}", exc_info=True)

    logger.info(f"Documents split done. Total chunks created: {len(chunks)}")
    
    return chunks