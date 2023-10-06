import os
# import openai
import streamlit as st

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from typing import Any, Dict, List, Optional
from langchain.chat_models import AzureChatOpenAI
from langchain.agents.agent import AgentExecutor
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.chains.llm import LLMChain
from langchain.tools.python.tool import PythonAstREPLTool
from langchain.memory import ConversationBufferMemory
from langchain.agents.agent_toolkits.pandas.prompt import (
    PREFIX,
)
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
# import prompt_templates

openai_key = "8cdd021ec1d2452e8145423389daae0d"

# Langchain Openai config files
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = "https://gen-ai-openai-resource.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = openai_key
os.environ["OPENAI_API_VERSION"] = "2023-05-15"

# Openai Config variables
# openai.api_type = "azure"
# openai.api_key = openai_key
# openai.api_base = "https://gen-ai-openai-resource.openai.azure.com/"
# openai.api_version = "2023-05-15"
openai_embedding_model = "embedding_model"
deployment_name="generative_model"

prefix = PREFIX
# suffix = """Please read through the documents and provide a detailed, comprehensive, and accurate interpretation of the content.\n Remember to:\n
# - Extract and analyze any data or statistics presented, providing a clear explanation of what they represent.\n- Maintain a neutral and objective perspective, focusing solely on the information provided in the documents.\n- Identify and ensure to highlight any key details or findings.\n- Provide clear and concise response based on the information extracted from the documents.\n- Use table format to present data and statistical information.\n"
# {chat_history}
# Question: {input}
# {agent_scratchpad}"""

suffix = """You are a financial analyst. Understand the context properly and answer clearly"
{chat_history}
Question: {input}
{agent_scratchpad}"""

include_df_in_prompt = None
input_variables = None
callback_manager = None
verbose = True
return_intermediate_steps = True
max_iterations = 15
max_execution_time = None
early_stopping_method = "force"
agent_executor_kwargs = None

llm = AzureChatOpenAI(
    deployment_name = deployment_name,
#     model_name = "gpt-4",
    temperature=0, streaming =True,callbacks=[StreamingStdOutCallbackHandler()])

st.title("HF GPT Demo App")

with st.container():
    uploaded_file = st.file_uploader('upload')
    if uploaded_file != None:
        try:
            df = pd.read_csv(uploaded_file)
        except:
            df = pd.read_excel(uploaded_file)

try:
    suffix_to_use = suffix
    input_variables = ["input", "chat_history", "agent_scratchpad"]
    tools = [PythonAstREPLTool(locals={"df": df})]

    prompt = ZeroShotAgent.create_prompt(
        tools, prefix=prefix, suffix=suffix_to_use, input_variables=input_variables
    )

    memory = ConversationBufferMemory(memory_key="chat_history")

    if "df" in input_variables:
        partial_prompt = prompt.partial(df=str(df.head().to_markdown()))
    else:
        partial_prompt = prompt

    llm_chain = LLMChain(
        llm=llm,
        prompt=partial_prompt,
        callback_manager=callback_manager,
    )

    tool_names = [tool.name for tool in tools]

    agent = ZeroShotAgent(
        llm_chain=llm_chain,
        allowed_tools=tool_names,
        callback_manager=callback_manager,
    #     **kwargs,
    )

    # agent = initialize_agent(
    #     tool_names,
    #     llm_chain,
    #     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    #     verbose=True,
    #     return_intermediate_steps=True,
    # )
    
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        callback_manager=callback_manager,
        verbose=verbose,
        # return_intermediate_steps=return_intermediate_steps,
        max_iterations=max_iterations,
        max_execution_time=max_execution_time,
        early_stopping_method=early_stopping_method,
        memory=memory
    #         **(agent_executor_kwargs or {}),
    )
except Exception as e:
    print(e)
    pass

with st.container():
    st.write('---')
    # query = st.text_input("Ask your question here:")
    # resp = agent_chain({"input":query})
    # st.write(resp)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Type your question here"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            resp = agent_chain({"input":prompt})
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(resp['output'])
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": resp['output']})
        except Exception as e:
            with st.chat_message("assistant"):
                st.markdown(e)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": e})
