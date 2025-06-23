from pydantic import BaseModel, Field
from typing import Literal
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import logging
from prompts import  question_classifier_prompt
import json

logger=logging.getLogger(__name__)

class TopicOrDocumentLevel(BaseModel):
    question_level: Literal["document", "topic"] = Field(
        description=(
            "Classifies the user's question as either 'document' or 'topic' level. "
            "'Document' level questions require understanding the entire document to answer — for example, questions about the overall summary, key takeaways, or main themes. "
            "'Topic' level questions focus on a specific detail or section and can typically be answered using only part of the document."
        )
    )

def classify_question_level(question: str, llm=ChatOpenAI())-> str:
    """
    Classifies whether the question requires looking at the complete PDF (document level)
    or just some parts of the pdf (topic level)
    Parameters:
        question: question of the user
        llm: (optional)
    Returns:
        document or topic (category of the question)
    """

    schema=PydanticOutputParser(pydantic_object=TopicOrDocumentLevel)

    # creating prompt
    classify_question_prompt = question_classifier_prompt(schema)
    logger.debug(f"Prompt created.")

    parser=StrOutputParser()

    question_classifier_chain= classify_question_prompt | llm | parser
    try:
        logger.debug(f"Classifying question: {question}")
        result=question_classifier_chain.invoke({"question": question})
        logger.debug(f"Classification result: {result}")
    except Exception as e:
        logger.error(f"Couldn't classify the question as topic or document level: {str(e)}", exc_info=True)
        return 'document'

    parsed = json.loads(result)
    return parsed.get("question_level") == "topic"