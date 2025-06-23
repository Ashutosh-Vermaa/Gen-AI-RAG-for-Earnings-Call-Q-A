from langchain_core.prompts import PromptTemplate

def question_classifier_prompt(schema):

    """
    Prompts that helps classify a question either 'topic' or 'document' level
    Parameters:
        schema (class): defines the structure of the output 
    Returns:
        prompt to classify a question
    """

    template=PromptTemplate(
        template="""
        You are a knowledgeable financial assistant with expertise in analyzing company earnings calls. 
        Your task is to classify the following user question into one of two categories: "document-level" or "topic-level".

        - A **document-level** question requires understanding or summarizing the entire earnings call transcript. 
        Examples include questions about the overall summary, key takeaways, or general themes discussed in the call.

        - A **topic-level** question focuses on a specific section, detail, or aspect of the call and can usually be answered by referencing only a portion of the transcript.

        Use the following format when providing your classification:
        {format_instructions}

        Question: {question}
        """,
        input_variables=["question"],
        partial_variables={'format_instructions': schema.get_format_instructions()}
    )

    return template