from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
chat = ChatOpenAI()
embeddings = OpenAIEmbeddings()


db = Chroma(
    persist_directory="emb",
    embedding_function=embeddings
)

'''this line basically makes sure that db object has a get_relevant_document method which is used by the RetrievalQA'''
''' the as_retriever method of the chroma instance add the get_relev_doc method to itself'''
'''This get_rele_doc method inturn calls the similarity_search method of the chroma instance'''
'''All this is drama is to make sure that RetrievalQA is vector store agnostic'''
retriever = db.as_retriever()

'''chain type can be stuff, map_reduce, map_rerank, refine'''
chain = RetrievalQA.from_chain_type(
    llm = chat,
    retriever = retriever,
    chain_type="stuff",
)

result = chain.invoke("What is an interesting fact about food?")

print(result)