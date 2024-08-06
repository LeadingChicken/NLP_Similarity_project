from fastapi import FastAPI
import getpass
import os
api_key = "sk-0NAZiHm3_xTZMBXUzpNJj-9ZMqp1ErwXENztS5vOgcT3BlbkFJuVdLBjrmtBgjxfPsJ24WootDeM7agi9oJbOwmkSxgA"
hgf_key = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["OPENAI_API_KEY"] = api_key
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


app = FastAPI()

fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]

def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    return docs
def document_preprocess(docs):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    return all_splits
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
    
@app.get("/message/")
async def read_item(pdf_path: str, language: str, question: str, k:int = 4):
    docs = load_pdf(pdf_path)
    all_splits = document_preprocess(docs)
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = ChatPromptTemplate.from_template("""Use the following pieces of context to answer the question about the paper that user is asking.
    If the context doesn't provide enough information, just say that you don't know, don't try to make up an answer.
    The question from user can be in Vietnamese or English.
    Pay attention to the context of the question rather than just looking for similar keywords in the corpus.
    Use three sentences maximum and keep the answer as concise as possible.
    Output the answer in the language that user want to see.
    Context: 
    {context}
    Question: {question}
    Answer the question in {language}
    Helpful Answer:
    """)
    rag_chain = (
    {"context": retriever | format_docs,"language":lambda x: language, "question": RunnablePassthrough(),}
    | prompt
    | llm
    | StrOutputParser()
    )
    result = ""
    for chunk in rag_chain.stream("How many criterias in MFC benchmark?"):
        result = result + chunk
    return {"model_output":result}
    