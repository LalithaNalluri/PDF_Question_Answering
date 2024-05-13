import openai
import langchain
import pinecone
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import streamlit as st
from langchain.vectorstores import Pinecone

from dotenv import load_dotenv
load_dotenv()

#Read the document
def read_doc(directory):
    file_loader=PyPDFDirectoryLoader(directory)
    document=file_loader.load()
    return document
doc=read_doc('C:\/Users\/lalit\/Documents\/')
len(doc)

#divide the doc into chunks
def chunk(docs,chunk_size=800, chunk_overlap=50):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    docs=text_splitter.split_documents(doc)
    return docs
documents=chunk(docs=doc)
#documents

#OpenAI Embedding Technique
import os
embeddings=OpenAIEmbeddings(api_key=os.environ["OpenAI_API_Key"])
#embeddings

pinecone.Pinecone(
    api_key="4d00d134-36d2-4156-af8b-b073aac6bcc4"
)
index_name="langchainvector"
from pinecone import Pinecone,ServerlessSpec
os.environ["PINECONE_API_KEY"]="4d00d134-36d2-4156-af8b-b073aac6bcc4"
pc = Pinecone(api_key="4d00d134-36d2-4156-af8b-b073aac6bcc4")
index_name = "langchainvector" 
index = pc.Index(index_name)

from langchain_pinecone import PineconeVectorStore
docsearch = PineconeVectorStore.from_documents(documents= doc, embedding=embeddings, index_name=index_name)
from langchain.vectorstores import Pinecone
def retrieve_query(query,k=2):
    matching_results=docsearch.similarity_search(query,k=k)
    return matching_results

from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI
llm=OpenAI(model_name="gpt-3.5-turbo-instruct",temperature=0.5)
chain=load_qa_chain(llm,chain_type="stuff")

def retrieve_answ(query):
    doc_search=retrieve_query(query)
    print(doc_search)
    input_data = {
    'input_documents': doc_search,
    'question': query
    }

# Now, pass the 'input_data' dictionary to the 'invoke' method
    response=chain.invoke(input=input_data)
  #  response=chain.invoke(input_documents=doc_search,question=query)
    return response

_ , col2,_ = st.columns([1,7,1])
with col2:
    col2 = st.header("Simplchat: Chat with your data")
    url = False
    query = False
    pdf = False
    data = False
    data= True
    query = st.text_input("Enter your query")
    response=retrieve_answ(query)
    button = st.button("Submit")



if button:
    st.subheader("Response is")
    st.write(response)