from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import re
import logging

logger = logging.getLogger(__name__)

def load_and_clean_pdf(file_path):
    """
    Loads a PDF file using LangChain's PyPDFLoader and returns a list of documents.
    Parameters:
        file_path (str): Path to the PDF file.
    Returns:
        list: List of Document objects if successful, None otherwise.
    """
    try:
        loader=PyPDFLoader(file_path)
        input_file=loader.load()
        logger.info(f"PDF loaded with total docs/pages: {len(input_file)}")
    except Exception as e:
        logger.critical(f"Failed to load PDF file '{file_path}': {str(e)}", exc_info=True)
        print("Could not load PDF. Please check the path or format")
        raise e

    #pre-processing
    pattern = r"Amara Raja Energy & Mobility Limited\s+\(Formerly known as Amara Raja Batteries Limited\)\s+.*, \d+\s+Page \d+ of \d+\s*"
    complete_pdf=Document(page_content="")
    for doc in input_file:
        
        doc.page_content= re.sub(pattern, "", doc.page_content)
        # doc.page_content=text.replace("\n", "")
        complete_pdf.page_content+= " " + doc.page_content
    logger.info("Merged all the pages to create a complete_pdf")

    return complete_pdf