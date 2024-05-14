import os
import tiktoken
import streamlit as st
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    load_index_from_storage,
)
from llama_index.llms.openai import OpenAI
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from dotenv import load_dotenv

load_dotenv()
index_name = "./cache"
documents_folder = "./data"
Settings.llm = OpenAI(temperature=0.0, model="gpt-4")
token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-4").encode,
    verbose=False,  # set to true to see usage printed to the console
)

Settings.callback_manager = CallbackManager([token_counter])
@st.cache_resource
def initialize_index(index_name, documents_folder):
    if os.path.exists(index_name):
        storage_context = StorageContext.from_defaults(persist_dir=index_name)
        index = load_index_from_storage(storage_context)
    else:
        documents = SimpleDirectoryReader(documents_folder).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=index_name)

    return index


@st.cache_data(max_entries=200, persist=True)
def query_index(_index, query_text):
    if _index is None:
        return "Please initialize the index!"
    response = _index.as_query_engine().query(query_text)
    return str(response)


st.title("Self-ERP Guide Demo")
st.header("Odoo module for accounting in Ukraine")

index = initialize_index(index_name, documents_folder)

text = st.text_input("Запитання:", value="Як налаштувати облік надходження основних засобів?")

if st.button("Отримати відповідь") and text is not None:
    response = query_index(index, text)
    st.markdown(response)

    llm_col, embed_col = st.columns(2)
    with llm_col:
        st.markdown(
            f"LLM Prompt Tokens Used: {token_counter.prompt_llm_token_count}"
        )
        st.markdown(
            f"LLM Completion Tokens Used: {token_counter.completion_llm_token_count}"
        )

    with embed_col:
        st.markdown(
            f"Embedding Tokens Used: {token_counter.total_embedding_token_count}"
        )
        st.markdown(
            f"Total LLM Tokens Used: {token_counter.total_llm_token_count}"
        )
