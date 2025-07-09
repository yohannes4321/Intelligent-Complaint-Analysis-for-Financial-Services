import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os 
from typing_extensions import List,TypedDict
from langgraph.graph import START,StateGraph
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()
google_api_key=os.getenv('GOOGLE_API_KEY')

data=pd.read_csv('filtered_complaints.csv')

document=[]
for _,row in data.iterrows():
  doc=Document(
      page_content=str(row['Consumer complaint narrative']),
      metadata={'Product':row['Product'],'Complaint ID':row['Complaint ID']}
  )
  document.append(doc)

split=RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=100)
chunk=split.split_documents(document)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a financial analyst assistant for CrediTrust."
              " Your task is to answer questions about customer complaints. Use the following "
              "retrieved complaint excerpts to formulate your answer. If the context doesn't contain "
              "the answer, state that you don't have enough information"),
    ("human", "Context {context}\n\nQuestion: {question}\n\n Answer:")
])


embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Initialize Chroma with the HuggingFaceEmbeddings object
vectore_store=Chroma(
    persist_directory='vector_store',
    collection_name='Intelligent',
    embedding_function=embeddings,
)
# from langchain.schema import Document

def batch_process(documents, batch_size, process_function):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        process_function(batch)

def process_function(batch):
    # Each item in batch is a Document object with metadata
    vectore_store.add_documents(documents=batch)


llm=init_chat_model("gemini-2.0-flash",model_provider="google_genai",google_api_key=google_api_key)

class State(TypedDict):
  question:str
  context:List[Document]
  answer:str
def retrive(state: State):
  retrived_docs=vectore_store.similarity_search(state['question'])
  return {'context':retrived_docs}
def generate(state :State):
  docs_content="\n\n".join(doc.page_content for doc in state['context'])
  messages=prompt.invoke({'question':state['question'],"context": docs_content})

graph_builder=StateGraph(State).add_sequence([retrive,generate])
graph_builder.add_edge(START,'retrive')
graph=graph_builder.compile()
result=graph.invoke({'question':"What is the topic of discussion"})
print(result['answer'])
