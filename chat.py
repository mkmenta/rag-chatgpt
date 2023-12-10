import datetime
import os
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import (ConversationBufferMemory,
                              StreamlitChatMessageHistory)

from chains.conversational_chain import ConversationalChain
from chains.conversational_retrieval_chain import (
    TEMPLATE, ConversationalRetrievalChain)
from knowledge_set import compute_knowledge_vectorstore
from streaming import StreamHandler
from utils import get_available_openai_models


@st.cache_resource(show_spinner=True)
def load_knowledge(knowledge, model_name):
    return compute_knowledge_vectorstore(knowledge, OpenAIEmbeddings(model=model_name))


@st.cache_data
def chat_model_list():
    return get_available_openai_models(put_first='gpt-3.5-turbo', filter_by='gpt')


@st.cache_data
def embedding_model_list():
    return get_available_openai_models(filter_by='embedding')


class StreamlitChatView:
    def __init__(self, knowledge_folder: str) -> None:
        st.set_page_config(page_title="RAG ChatGPT", page_icon="📚", layout="wide")
        with st.sidebar:
            st.title("RAG ChatGPT")
            with st.expander("Model parameters"):
                self.model_name = st.selectbox("Model:", options=chat_model_list())
                self.temperature = st.slider("Temperature", min_value=0., max_value=2., value=0.7, step=0.01)
                self.top_p = st.slider("Top p", min_value=0., max_value=1., value=1., step=0.01)
                self.frequency_penalty = st.slider("Frequency penalty", min_value=0., max_value=2., value=0., step=0.01)
                self.presence_penalty = st.slider("Presence penalty", min_value=0., max_value=2., value=0., step=0.01)
            with st.expander("Prompts"):
                curdate = datetime.datetime.now().strftime("%Y-%m-%d")
                model_name = self.model_name.replace('-turbo', '').upper()
                system_message = (f"You are ChatGPT, a large language model trained by OpenAI, "
                                  f"based on the {model_name} architecture.\n"
                                  f"Knowledge cutoff: 2021-09\n"
                                  f"Current date: {curdate}\n")
                self.system_message = st.text_area("System message", value=system_message)
                self.context_prompt = st.text_area("Context prompt", value=TEMPLATE)
            with st.expander("Embeddings parameters"):
                self.embeddings_model_name = st.selectbox("Embeddings model:", options=embedding_model_list())
            self.inject_knowledge = st.checkbox("Inject knowledge", value=True)
            knowledge_names = [fn for fn in os.listdir(knowledge_folder)
                               if os.path.isdir(os.path.join(knowledge_folder, fn))]
            self.knowledge = st.selectbox("Select a knowledge folder:", knowledge_names)
        self.user_query = st.chat_input(placeholder="Ask me anything!")

    def add_message(self, message: str, author: str, context: Optional[List] = None):
        assert author in ["user", "assistant"]
        with st.chat_message(author):
            st.markdown(message)
            if context is not None:
                with st.expander("Context", expanded=False):
                    for doc in context:
                        st.markdown(f"**{doc['metadata']['source']}**")
                        st.text(doc['page_content'])

    def add_message_stream(self, author: str):
        assert author in ["user", "assistant"]
        return StreamHandler(st.chat_message(author).empty())


def setup_memory():
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    return ConversationBufferMemory(memory_key="chat_history", output_key="response", chat_memory=msgs,
                                    return_messages=True)


def setup_chain(llm, memory, inject_knowledge, system_message, context_prompt, retriever):
    if not inject_knowledge:
        # Custom conversational chain
        return ConversationalChain(
            llm=llm,
            memory=memory,
            system_message=system_message,
            verbose=True)
    else:
        return ConversationalRetrievalChain(
            llm=llm,
            retriever=retriever,
            memory=memory,
            system_message=system_message,
            context_prompt=context_prompt,
            verbose=True)


STREAM = False

# Setup
load_dotenv()
view = StreamlitChatView("data")
memory = setup_memory()
retriever = None
if view.inject_knowledge:
    retriever = load_knowledge(view.knowledge, model_name=view.embeddings_model_name).as_retriever()
llm = ChatOpenAI(
    streaming=STREAM,
    model_name=view.model_name,
    temperature=view.temperature,
    top_p=view.top_p,
    frequency_penalty=view.frequency_penalty,
    presence_penalty=view.presence_penalty)
chain = setup_chain(llm=llm, memory=memory, inject_knowledge=view.inject_knowledge,
                    retriever=retriever, system_message=view.system_message,
                    context_prompt=view.context_prompt)

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
        response = chain({"question": view.user_query})
        view.add_message(response['response'], "assistant", context=response['context'])
