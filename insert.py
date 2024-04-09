from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import faiss
from langchain_community.vectorstores import qdrant
from langchain_community.vectorstores import Chroma

data_path = ("data/")
db_chroma_path = "vectorstores/db_chroma"

#create vector database
def create_vector_db():
    loader = DirectoryLoader(data_path,glob='*.pdf',loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500,chunk_overlap= 50 )
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",  
    model_kwargs={"device": "cpu"}  
    )

    db = Chroma.from_documents(texts, embeddings, persist_directory= db_chroma_path)
    

if __name__ == '__main__':
    create_vector_db()