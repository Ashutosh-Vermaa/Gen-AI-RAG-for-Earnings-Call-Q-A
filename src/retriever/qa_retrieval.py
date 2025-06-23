from retriever.conditional_chain import integrate_chains
import logging

logger=logging.getLogger(__name__)

def get_answer(question, retriever, docs):
    """
    generates the final output
    Parameters:
        question: of the user
        retriever: to retriever docs
        docs: to generate summary if needed
    Returns:
        final response to the question
    """
    final_chain=integrate_chains(retriever)
    try:
        response=final_chain.invoke(question)
        logger.debug(f"Response generated. Sample: {response[:150]}")
        return response
    except Exception as e:
        logger.critical(f"Couldn't generate response: {str(e)}", exc_info=True)
        raise e
    
