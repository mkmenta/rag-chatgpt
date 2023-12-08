from operator import itemgetter

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter


import os

# load dotenv
load_dotenv()


def is_valid_document(file_path):
    extensions = ('.pdf', '.txt')
    return file_path.lower().endswith(extensions)


all_files = [fn for fn in os.listdir('data') if is_valid_document(fn)]
docs = []
for file in all_files:
    if file.endswith('.pdf'):
        loader = PyPDFLoader
    elif file.endswith('.txt'):
        loader = TextLoader
    else:
        raise ValueError(f'Unknown file type: {file}')
    loader = loader(os.path.join("data", file))
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs.extend(text_splitter.split_documents(documents))

# vectorstore = FAISS.from_texts(
#     ["harrison worked at woowoo"], embedding=OpenAIEmbeddings()
# )
vectorstore = FAISS.from_documents(docs, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI()

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

x = chain.invoke("where did robertson work?")
print(x)
