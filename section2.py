import pandas as pd
data=pd.read_csv('filtered_complaints.csv')
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
document=[]
for _,row in data.iterrows():
  doc=Document(
      page_content=str(row['Consumer complaint narrative']),
      metadata={'Product':row['Product'],'Complaint ID':row['Complaint ID']}
  )
  document.append(doc)
split=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=300)
chunk=split.split_documents(document)
model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings=model.encode([doc.page_content for doc in chunk])
vectore_store=Chroma(
    persist_directory='vector_store',
    collection_name='Intelligent',
    embedding_function=embeddings,
)