from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Chroma
from langchain.schema import BaseRetriever

#Custom Retriever
class RedundantFilterRetriever(BaseRetriever):

    embeddings: Embeddings
    chroma: Chroma

    def get_relevant_documents(self, query):

        #calculate embedding for query and feed that and document embeddings into the max_marginal_relevance_search_by_vector method(Chroma)
        emb = self.embeddings.embed_query(query)
       
        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding=emb,
            lambda_mult=0.9
        )
    
    async def aget_relevant_documents(self):
        return []