from langchain import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)


class ConversationalChain(LLMChain):
    """Basic conversational chain.

    The original chains from `langchain` do not use the `MessagesPlaceholder` template.
    They summarize all the conversation in a single prompt that they send to ChatGPT API.
    This chain allows to use the natural messaging structure from ChatGPT API.
    """

    def __init__(self, llm, memory, system_message: str, verbose: bool = False):
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(system_message),
                # The `variable_name` here is what must align with memory
                MessagesPlaceholder(variable_name=memory.memory_key),
                HumanMessagePromptTemplate.from_template("{question}"),
            ]
        )
        super().__init__(llm=llm, prompt=prompt, verbose=verbose, memory=memory)
