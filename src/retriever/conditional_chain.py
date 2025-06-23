from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough, RunnableBranch
import logging
import json
from langchain_core.output_parsers import StrOutputParser
from prompts import response_prompt, question_classifier_prompt
# from retriever.summarize_docs_into_one import summary_document
from question_classifier import classify_question_level
from Indexing import create_retriever

logger=logging.getLogger(__name__)

# case 1: topic level question
def preprocessing(docs):
    return "\n".join([doc.page_content for doc in docs])

def topic_chain(retriever):
    topic_chain= retriever  \
                            |RunnableParallel( 
                                {'context': RunnableLambda(preprocessing),
                                'question': RunnablePassthrough()
                                }) 
                            
    return topic_chain
# case 2: document level question

######## COmbining two CASES

def integrate_chains(retriever,llm=ChatOpenAI()):
    """
    Puts together topic and document level chain.
    Parameters:
        topic_chain: to answer topic level questions
        docs: to generate summary of docs to answer document level questions
    """

    topic_chain_fn=topic_chain(retriever)
    parser=StrOutputParser()

    common_chain= response_prompt() | llm | parser
    logger.info("Common chain created")
    
    #loading summary document to answer document level questions
    try:
        with open("doc_summary.txt", 'r', encoding='utf-8') as f:
            summary=f.read()
        logger.debug(f"Loaded document summary. Sample: {summary[:150]}")
    except:
        logger.error(f"Couldn't load document summary")

    final_chain=RunnableBranch(
        (classify_question_level, topic_chain_fn | common_chain),
        RunnableLambda(lambda q : {'question': q, 'context': summary}) |common_chain #default chain
    )
    logger.info("Final chain created")
    return final_chain

