from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import  HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document

def load_pdf_files(data):
  loader=DirectoryLoader(
    data,
    glob='**/*.pdf',
    loader_cls=PyPDFLoader
  )

  documents = loader.load()
  return documents



def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
  """ 
  Given a list of Document objects, return a new list of Document objects
  containing only 'source' in metedata and the original page_content
  """
  minimal_docs: List[Document] = []
  for doc in docs:
      src = doc.metadata.get("source")
      minimal_docs.append(
         Document(
            page_content=doc.page_content,
            metadata={"source": src}
         )
      )
  return minimal_docs


def text_split(minimal_docs):
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20,
  )
  texts_cunck = text_splitter.split_documents(minimal_docs)
  return texts_cunck

def download_embeddings():
  """
  Download and return the HuggingFace embeddings model. 
  """
  model_name = "sentence-transformers/all-MiniLM-L6-V2"
  embeddings = HuggingFaceEmbeddings(
    model_name=model_name
  )
  return embeddings

embedding = download_embeddings()