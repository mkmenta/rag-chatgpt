import os
import utils
import streamlit as st
from streaming import StreamHandler

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import random
import time
import os
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from dotenv import load_dotenv
from knowledge_set import compute_knowledge_vectorstore

st.set_page_config(page_title="ChatPDF", page_icon="📄")
# st.header('Chat with your documents')
# st.write('Has access to custom documents and can respond to user queries by referring to the content within those documents')
# st.write('[![view source code ](https://img.shields.io/badge/view_source_code-gray?logo=github)](https://github.com/shashankdeshpande/langchain-chatbot/blob/master/pages/4_%F0%9F%93%84_chat_with_your_documents.py)')

KNOWLEDGE_FOLDER = "data"
# Spinner and cache


@st.cache_resource(show_spinner=True)
# @st.spinner('Loading knowledge..')
def load_knowledge(knowledge):
    return compute_knowledge_vectorstore(knowledge, OpenAIEmbeddings())


class CustomDataChatbot:

    def __init__(self):
        load_dotenv()
        # self.openai_model = "gpt-3.5-turbo"

    # def save_file(self, file):
    #     folder = 'tmp'
    #     if not os.path.exists(folder):
    #         os.makedirs(folder)

    #     file_path = f'./{folder}/{file.name}'
    #     with open(file_path, 'wb') as f:
    #         f.write(file.getvalue())
    #     return file_path

    # @st.spinner('Analyzing documents..')
    # def setup_qa_chain(self, uploaded_files):
    #     # Load documents
    #     docs = []
    #     for file in uploaded_files:
    #         file_path = self.save_file(file)
    #         loader = PyPDFLoader(file_path)
    #         docs.extend(loader.load())

    #     # Split documents
    #     text_splitter = RecursiveCharacterTextSplitter(
    #         chunk_size=1500,
    #         chunk_overlap=200
    #     )
    #     splits = text_splitter.split_documents(docs)

    #     # Create embeddings and store in vectordb
    #     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    #     vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)

    #     # Define retriever
    #     retriever = vectordb.as_retriever(
    #         search_type='mmr',
    #         search_kwargs={'k': 2, 'fetch_k': 4}
    #     )

    #     # Setup memory for contextual conversation
    #     memory = ConversationBufferMemory(
    #         memory_key='chat_history',
    #         return_messages=True
    #     )

    #     # Setup LLM and QA chain
    #     llm = ChatOpenAI(model_name=self.openai_model, temperature=0, streaming=True)
    #     qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, verbose=True)
    #     return qa_chain

    def setup_chain(self, llm, inject_knowledge, retriever):
        # Setup memory for contextual conversation
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )
        if not inject_knowledge:
            return ConversationChain(llm=llm, memory=memory, verbose=True)
        else:
            return ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, verbose=True)

    @utils.enable_chat_history
    def main(self):
        # st.title("RAG ChatGPT")

        # Knowledge configuration
        with st.sidebar:
            inject_knowledge = st.checkbox("Inject knowledge", value=True)
            knowledge_names = [fn for fn in os.listdir(KNOWLEDGE_FOLDER)
                               if os.path.isdir(os.path.join(KNOWLEDGE_FOLDER, fn))]
            knowledge = st.selectbox("Select a knowledge folder:", knowledge_names)

        retriever = None
        if inject_knowledge:
            vectorstore = load_knowledge(knowledge)
            retriever = vectorstore.as_retriever()

        # User Inputs
        # uploaded_files = st.sidebar.file_uploader(label='Upload PDF files', type=['pdf'], accept_multiple_files=True)
        # if not uploaded_files:
        #     st.error("Please upload PDF documents to continue!")
        #     st.stop()

        user_query = st.chat_input(placeholder="Ask me anything!")

        if user_query:
            llm = ChatOpenAI()

            qa_chain = self.setup_chain(llm=llm, inject_knowledge=inject_knowledge, retriever=retriever)

            utils.display_msg(user_query, 'user')

            with st.chat_message("assistant"):
                # st_cb = StreamHandler(st.empty())
                response = qa_chain.run(user_query)  # , callbacks=[st_cb])
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    obj = CustomDataChatbot()
    obj.main()
