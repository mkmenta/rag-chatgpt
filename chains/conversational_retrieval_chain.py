from __future__ import annotations
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
    Callbacks,
)
from pydantic import Extra
from langchain.schema.language_model import BaseLanguageModel
from langchain.prompts.base import BasePromptTemplate
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from typing import Any, Dict, List, Optional
from typing import Any, Dict, Optional
from langchain import LLMChain
from langchain.chains.base import Chain

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.schema import BaseRetriever

TEMPLATE = (
    "Question: {question}\n\n"
    "Use the following pieces of context to answer the question.\n"
    "If you don't know the answer, just say that you don't know, don't try to make up an answer.\n"
    "----------------\n"
    "{context}"
)


class ConversationalRetrievalChain(Chain):
    """
    An example of a custom chain.
    """
    system_message: str
    context_prompt: str = TEMPLATE
    """Prompt object to use."""
    llm: BaseLanguageModel
    output_key: str = "text"  #: :meta private:
    retriever: BaseRetriever
    # docs_combiner: BaseCombineDocumentsChain = StuffDocumentsChain()

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return ["question"]

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(self.system_message),
                # The `variable_name` here is what must align with memory
                MessagesPlaceholder(variable_name=self.memory.memory_key),
                HumanMessagePromptTemplate.from_template(self.context_prompt),
            ]
        )

        # TODO maybe it makes sense to use the vectorstore directly with a k
        docs = self.retriever.get_relevant_documents(
            inputs['question'], callbacks=run_manager.get_child()
        )
        inputs = inputs.copy()
        inputs['context'] = "\n\n".join([doc.page_content for doc in docs])

        # Your custom chain logic goes here
        # This is just an example that mimics LLMChain
        prompt_value = prompt.format_prompt(**inputs)

        # Whenever you call a language model, or another chain, you should pass
        # a callback manager to it. This allows the inner run to be tracked by
        # any callbacks that are registered on the outer run.
        # You can always obtain a callback manager for this by calling
        # `run_manager.get_child()` as shown below.
        response = self.llm.generate_prompt(
            [prompt_value], callbacks=run_manager.get_child() if run_manager else None
        )

        # If you want to log something about this run, you can do so by calling
        # methods on the `run_manager`, as shown below. This will trigger any
        # callbacks that are registered for that event.
        # if run_manager:
        #     run_manager.on_text("Log something about this run")

        return {self.output_key: response.generations[0][0].text}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # Your custom chain logic goes here
        # This is just an example that mimics LLMChain
        prompt_value = self.prompt.format_prompt(**inputs)

        # Whenever you call a language model, or another chain, you should pass
        # a callback manager to it. This allows the inner run to be tracked by
        # any callbacks that are registered on the outer run.
        # You can always obtain a callback manager for this by calling
        # `run_manager.get_child()` as shown below.
        response = await self.llm.agenerate_prompt(
            [prompt_value], callbacks=run_manager.get_child() if run_manager else None
        )

        # If you want to log something about this run, you can do so by calling
        # methods on the `run_manager`, as shown below. This will trigger any
        # callbacks that are registered for that event.
        if run_manager:
            await run_manager.on_text("Log something about this run")

        return {self.output_key: response.generations[0][0].text}

    @property
    def _chain_type(self) -> str:
        return "my_custom_chain"
