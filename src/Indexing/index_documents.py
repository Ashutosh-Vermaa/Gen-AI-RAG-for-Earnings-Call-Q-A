from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pinecone import ServerlessSpec
import logging
from langchain_openai import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.chain_extract import LLMChainExtractor

logger = logging.getLogger(__name__)

def create_index(documents, index_name="earning-call-index", dimension=700):
    """
    Creates a Pinecone index and adds embedded documents.
    
    Parameters:
        documents (List[Document]): List of LangChain Document objects.
        index_name (str): Name of the Pinecone index.
        dimension (int): Embedding dimension (must match model output).
    
    Returns:
        vector_store: LangChain Pinecone vector store instance.
    """
    try:
        embedding = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=dimension)
        pc = Pinecone()  # Uses env vars for api_key and env
        
        if not pc.has_index(index_name):
            pc.create_index(
                name=index_name,
                dimension=dimension,
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            index = pc.Index(index_name)
            vector_store = PineconeVectorStore(index=index, embedding=embedding)
            vector_store.add_documents(documents)
            logger.info(f"Created new index & added {len(documents)} documents to index: {index_name}")
        else:
            logger.info(f"Index already exists: {index_name}. Skipping document ingestion")
            index = pc.Index(index_name)
            vector_store = PineconeVectorStore(index=index, embedding=embedding)
       
        # logger.info(f"Added {len(documents)} documents to index: {index_name}")
        return vector_store

    except Exception as e:
        logger.critical(f"Couldn't create index: {str(e)}", exc_info=True)
        print("Error creating index of the documents")
        raise e

#----------------------Creating a Retriver ----------------------------
def create_retriever(vector_store, k=3):
    """
    Creates a retriever from a Pinecone vector store using MMR search.

    Parameters:
        vector_store: PineconeVectorStore object.
        k: No. of docs to retrieve

    Returns:
        retriever: Configured retriever object.
    """
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={'k': k, 'lambda_mult': 0.5}
    )
    logger.info("Retriever created.")
    return retriever

def contextual_compression_retriever(vector_store,best_retriever):
    #set up compressor (LLM) using LLM
    llm=ChatOpenAI()
    compressor=LLMChainExtractor.from_llm(llm)

    #retriever
    contextual_retriever=ContextualCompressionRetriever(
        base_retriever=best_retriever,
        base_compressor=compressor
    )
    return contextual_retriever
