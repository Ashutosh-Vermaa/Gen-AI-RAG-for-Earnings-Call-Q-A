import logging
from ingestion import load_and_clean_pdf, split_docs
from Indexing import create_index, create_retriever,contextual_compression_retriever
from retriever import get_answer
from ragas.evaluation import evaluate
from ragas import EvaluationDataset

from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, NoiseSensitivity,AnswerAccuracy
from langchain_openai import ChatOpenAI
from evaluation import load_pairs # self-defined



# Loading Q and A pairs
qa_pairs=load_pairs()

sample_queries, expected_responses=qa_pairs.keys(), qa_pairs.values()

# loading retriever and other components of RAG

file_path = r"D:\Documents\LangChain\4. End to End Project\Transcript of 39th AGM FY24.pdf"

complete_pdf = load_and_clean_pdf(file_path)

chunks = split_docs(complete_pdf)

index = create_index(chunks)

retriever = create_retriever(index)

context_retriever=contextual_compression_retriever(index,retriever)
# Creating evaluation dataset

dataset = []
for query, reference in zip(sample_queries, expected_responses):
    relevant_docs = context_retriever.invoke(query)
    response = get_answer(query, context_retriever, chunks)
    dataset.append(
        {
            "user_input": query,
            "retrieved_contexts": [rdoc.page_content for rdoc in relevant_docs],
            "response": response,
            "reference": reference,
        }
    )

llm=ChatOpenAI()
"""evaluation_dataset = EvaluationDataset.from_list(dataset)

# Evaluation

evaluator_llm = LangchainLLMWrapper(llm)

result = evaluate(
    dataset=evaluation_dataset,
    metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness(), NoiseSensitivity(), AnswerAccuracy()],
    llm=evaluator_llm,
)

print(result)"""

from datasets import Dataset
data=Dataset.from_list(dataset)
# alternative approach
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

result = evaluate(
    dataset = data, 
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
)

df = result.to_pandas()
print(df)
df.to_csv("evaluation_results_latest(1).csv", index=False)
