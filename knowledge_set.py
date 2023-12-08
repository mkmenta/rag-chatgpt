import os

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter


def compute_knowledge_vectorstore(name: str, embeddings, base_path: str = "data"):
    full_path = os.path.join(base_path, name)
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    # Load if exists
    tracked_files = set()
    vectorstore = None
    if os.path.exists(os.path.join(full_path, "faiss_index")):
        vectorstore = FAISS.load_local(os.path.join(full_path, "faiss_index"), embeddings=embeddings)
        with open(os.path.join(full_path, "tracked_files"), 'r') as f:
            for line in f:
                tracked_files.add(line.strip())

    # Check missing
    all_files = [fn for fn in os.listdir(full_path)
                 if fn not in ('faiss_index', 'tracked_files')]
    missing_files = [fn for fn in all_files if fn not in tracked_files]

    # Compute missing
    missing_docs = []
    for file in missing_files:
        if file.lower().endswith('.pdf'):
            loader = PyPDFLoader
        elif file.lower().endswith('.txt'):
            loader = TextLoader
        else:
            print(f'Unsupported file type: {file}')
        loader = loader(os.path.join(full_path, file))
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        missing_docs.extend(text_splitter.split_documents(documents))

    if missing_docs:
        new_vectorstore = FAISS.from_documents(missing_docs, embedding=OpenAIEmbeddings())
        if vectorstore is None:
            vectorstore = new_vectorstore
        else:
            vectorstore.merge_from(new_vectorstore)
        vectorstore.save_local(os.path.join(full_path, "faiss_index"))

        with open(os.path.join(full_path, "tracked_files"), 'w') as f:
            for fn in all_files:
                f.write(fn + '\n')
    return vectorstore
