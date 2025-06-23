from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer
from ingestion import load_and_clean_pdf, split_docs, split_docs_semantic


file_path = r"D:\Documents\LangChain\4. End to End Project\Transcript of 39th AGM FY24.pdf"
complete_pdf = load_and_clean_pdf(file_path)
chunks = split_docs(complete_pdf)

if __name__=='__main__':
    with Run().context(RunConfig(nranks=1, experiment="earnings_call")):

        config = ColBERTConfig(
            nbits=2,
            root=r"D:\Documents\LangChain\5. Q&A of earinings call\src\Indexing",
        )
        indexer = Indexer(checkpoint=r"D:\Documents\LangChain\5. Q&A of earinings call\src\Indexing\colbert", config=config)
        indexer.index(name="colb_ind", collection=chunks)


"""
if __name__=='__main__':
    with Run().context(RunConfig(nranks=1, experiment="earnings_call")):

        config = ColBERTConfig(
            root=r"D:\Documents\LangChain\5. Q&A of earinings call\src\Indexing",
        )
        searcher = Searcher(index="colbert_earning_call", config=config)
        queries = Queries("/path/to/MSMARCO/queries.dev.small.tsv")
        ranking = searcher.search_all(queries, k=100)
        ranking.save("msmarco.nbits=2.ranking.tsv")
"""