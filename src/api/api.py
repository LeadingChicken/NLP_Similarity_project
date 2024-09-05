from fastapi import FastAPI
from pydantic import BaseModel
import getpass
import os
api_key = "sk-gE4n4twSu7MFVrc2GnRyDkfp6g_zKlAiOjgTfA15_8T3BlbkFJtk4b4FtnldE8P7HLzm_gTqvRRu984jgGMOwAMt0sMA"
hgf_key = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["OPENAI_API_KEY"] = api_key
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import LLMChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer, util
import torch
from sklearn.decomposition import PCA
app = FastAPI()


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
    prompt_template = """
    Tóm tắt văn bản sau đây một cách ngắn gọn:
    {text}
    """

    # Đặt prompt với văn bản cần tóm tắt
    prompt_summarize = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = LLMChain(llm=llm, prompt=prompt_summarize)

    # Gửi văn bản qua API GPT để tóm tắt
    summary = chain.run(docs)
    prompt = ChatPromptTemplate.from_template("""Use the following pieces of context to answer the question about the paper that user is asking.
    You will be given about the summary of the paper.
    If the context doesn't provide enough information, just say that you don't know, don't try to make up an answer.
    The question from user can be in Vietnamese or English.
    Pay attention to the context of the question rather than just looking for similar keywords in the corpus.
    Use three sentences maximum and keep the answer as concise as possible.
    Output the answer in the language that user want to see.
    Summary: {summary}
    Context: 
    {context}
    Question: {question}
    Answer the question in {language}
    Helpful Answer:
    """)
    rag_chain = (
    {"summary": lambda x: summary,"context": retriever | format_docs,"language":lambda x: language, "question": RunnablePassthrough(),}
    | prompt
    | llm
    | StrOutputParser()
    )
    result = ""
    for chunk in rag_chain.stream(question):
        result = result + chunk
    return {"model_output":result}



class Item(BaseModel):
    sentence1: str
    sentence2: str
    model_name:str

@app.post('/similarity')
async def similarity_score(item: Item):
    sentence1 = item.sentence1
    sentence2 = item.sentence2
    model_name = item.model_name

    if model_name == 'mBERT':
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        model = BertModel.from_pretrained("bert-base-multilingual-cased")
        encoded_sentence1 = tokenizer(sentence1, return_tensors='pt')
        encoded_sentence2 = tokenizer(sentence2, return_tensors='pt')
        output1 = model(**encoded_sentence1)['last_hidden_state'][0,0,:]
        output2 = model(**encoded_sentence2)['last_hidden_state'][0,0,:]
        print(output1.size())
        print(output2.size())
        score = float(torch.cosine_similarity(output1,output2,dim=0))
        return {"similarity_score": score}
    elif model_name == "XML-Roberta":
        model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
        output1 = model.encode(sentence1)
        output2 = model.encode(sentence2)
        score = util.pytorch_cos_sim(output1, output2).item()
        return {"similarity_score": score}
    else:
        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
        model = SentenceTransformer('sentence-transformers/LaBSE')
        output1 = model.encode(sentence1)
        output2 = model.encode(sentence2)
        score = util.pytorch_cos_sim(output1, output2).item()
        return {"similarity_score": score}


    