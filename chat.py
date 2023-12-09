import os

import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import (ConversationBufferMemory,
                              StreamlitChatMessageHistory)

from chains.conversational_chain import ConversationalChain
from chains.conversational_retrieval_chain import ConversationalRetrievalChain
from knowledge_set import compute_knowledge_vectorstore
from streaming import StreamHandler


@st.cache_resource(show_spinner=True)
# @st.spinner('Loading knowledge..')
def load_knowledge(knowledge):
    return compute_knowledge_vectorstore(knowledge, OpenAIEmbeddings())


class StreamlitChatView:
    def __init__(self, knowledge_folder: str) -> None:
        st.set_page_config(page_title="RAG ChatGPT", page_icon="📚", layout="wide")

        with st.sidebar:
            self.inject_knowledge = st.checkbox("Inject knowledge", value=True)
            knowledge_names = [fn for fn in os.listdir(knowledge_folder)
                               if os.path.isdir(os.path.join(knowledge_folder, fn))]
            self.knowledge = st.selectbox("Select a knowledge folder:", knowledge_names)
        self.user_query = st.chat_input(placeholder="Ask me anything!")

    def add_message(self, message: str, author: str):
        assert author in ["user", "assistant"]
        with st.chat_message(author):
            st.markdown(message)

    def add_message_stream(self, author: str):
        assert author in ["user", "assistant"]
        return StreamHandler(st.chat_message(author).empty())


def setup_memory():
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    return ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)


def setup_chain(llm, memory, inject_knowledge, retriever):
    if not inject_knowledge:
        # Custom conversational chain
        return ConversationalChain(
            llm=llm,
            memory=memory,
            system_message="You are a nice chatbot having a conversation with a human.",
            verbose=True)
    else:
        return ConversationalRetrievalChain(
            llm=llm,
            retriever=retriever,
            memory=memory,
            system_message="You are a nice chatbot having a conversation with a human.",
            verbose=True)


STREAM = True

# Setup
load_dotenv()
view = StreamlitChatView("data")
memory = setup_memory()
retriever = None
if view.inject_knowledge:
    retriever = load_knowledge(view.knowledge).as_retriever()
chain = setup_chain(ChatOpenAI(streaming=STREAM), memory, view.inject_knowledge, retriever)

# Display previous messages
for message in memory.chat_memory.messages:
    view.add_message(message.content, 'assistant' if message.type == 'ai' else 'user')

# Send message
if view.user_query:
    view.add_message(view.user_query, "user")
    if STREAM:
        st_callback = view.add_message_stream("assistant")
        chain.run({"question": view.user_query}, callbacks=[st_callback])
    else:
        response = chain.run({"question": view.user_query})
        view.add_message(response, "assistant")
