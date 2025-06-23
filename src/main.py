import logging
from ingestion import load_and_clean_pdf, split_docs, split_docs_semantic
from Indexing import create_index, create_retriever, contextual_compression_retriever
from retriever import get_answer

with open("logs/debug.log", "w"):
    pass

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s",
    handlers=[
        logging.FileHandler("logs/debug.log"),
        logging.StreamHandler()  # Still prints to console
    ]

)
logger = logging.getLogger(__name__)

def main():
    file_path = r"D:\Documents\LangChain\4. End to End Project\Transcript of 39th AGM FY24.pdf"

    logger.info(" Loading and cleaning the PDF...")
    complete_pdf = load_and_clean_pdf(file_path)

    logger.info(" Splitting the PDF into chunks...")
    chunks = split_docs(complete_pdf)
    print(len(chunks) )
    for chunk in chunks:
        print(chunk)
        break

    logger.info(" Creating index...")
    index = create_index(chunks, "earning-call-index-semanticsplit")

    logger.info(" Creating retriever from index...")
    retriever = create_retriever(index)
    contextual_retriever=contextual_compression_retriever(index, retriever)

    # question = input("Enter your question (or type 'exit' to quit)").strip()
    # response = get_answer(question, retriever, chunks)
    # print(response)
    
    while True:
        question = input("Enter your question (or type 'exit' to quit): \n").strip()
        with open("logs/debug.log", "w"):
            pass
        if question.lower() in ['exit', "quit", 'q']:
            print("Exiting. Thank you!")
            break
        
        logger.info(f" Answering question: {question}")
        try:            
            response = get_answer(question, contextual_retriever, chunks)
            print(response, "\n\n")
        except Exception as e:
            logger.error(f"Error while processing question: {str(e)}", exc_info=True)
            print("An error occured, please try again")

if __name__ == "__main__":
    main()
