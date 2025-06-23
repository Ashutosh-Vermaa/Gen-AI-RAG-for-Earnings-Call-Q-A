from langchain_core.prompts import PromptTemplate

def doc_summary():
    """
    Helps LLM summarise a document. 
    Returns:
        summarizer prompt
    """

    summary_prompt=PromptTemplate(
        template="""
        You are a knowledgeable financial assistant specialized in interpreting company earnings calls.
        Summarise the below text to give important details.

        text: {text}
        """, input_variables=['text']
    )

    return summary_prompt