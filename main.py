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


if os.path.exists("data/faiss_index"):
    vectorstore = FAISS.load_local("data/faiss_index", embeddings=OpenAIEmbeddings())
tracked_files = set()
if os.path.exists('data/tracked_files'):
    with open('data/tracked_files', 'r') as f:
        for line in f:
            tracked_files.add(line.strip())
all_files = [fn for fn in os.listdir('data') if is_valid_document(fn)]
missing_files = [fn for fn in all_files if fn not in tracked_files]
missing_docs = []
for file in missing_files:
    if file.endswith('.pdf'):
        loader = PyPDFLoader
    elif file.endswith('.txt'):
        loader = TextLoader
    else:
        raise ValueError(f'Unknown file type: {file}')
    loader = loader(os.path.join("data", file))
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    missing_docs.extend(text_splitter.split_documents(documents))

# vectorstore = FAISS.from_texts(
#     ["harrison worked at woowoo"], embedding=OpenAIEmbeddings()
# )
if missing_docs:
    new_vectorstore = FAISS.from_documents(missing_docs, embedding=OpenAIEmbeddings())
    vectorstore.merge_from(new_vectorstore)
    vectorstore.save_local("data/faiss_index")

    with open('data/tracked_files', 'w') as f:
        for fn in all_files:
            f.write(fn + '\n')

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

x = chain.invoke("where did melbourne work?")
print(x)
