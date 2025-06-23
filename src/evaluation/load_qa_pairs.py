import logging
import sys

logger=logging.getLogger(__name__)

def load_pairs(file_path=r"D:\Documents\LangChain\5. Q&A of earinings call\src\evaluation\q_and_a_pairs.txt"):
    """
    loads question and answer pairs
    Parameters:
        file path
    Returns:
        pair of question and answers
    """
    try:
        with open(file_path, 'r') as f:
            qa_pairs=f.read()
        logger.debug(f"Loaded the q&a pairs. Count of it: {len(qa_pairs)}")
        return eval(qa_pairs)
    except Exception as e:
        logger.critical(f"Couldn't load evaluation QA pairs: {str(e)}")
        raise RuntimeError(f"Failed to load QA pairs: {str(e)}")