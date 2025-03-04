import streamlit as st
import os
import re
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_ollama.llms import OllamaLLM
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

class DensoAssistant:
    def __init__(
        self,
        model_name: str = "deepseek-r1",
        data_file: str = "data/data.txt"
    ):
        """
        :param model_name: Ollama model name (e.g., 'deepseek-r1').
        :param data_file:  Path to the single text file, e.g. 'data/data.txt'.
        """
        if not os.path.exists(data_file):
            st.warning(f"{data_file} not found. Using fallback text.")
            raw_docs = [Document(page_content="No data file found, so fallback text.")]
        else:
            loader = TextLoader(file_path=data_file)
            raw_docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunked_docs = []
        for doc in raw_docs:
            splits = text_splitter.split_text(doc.page_content)
            for chunk in splits:
                chunked_docs.append(Document(page_content=chunk, metadata=doc.metadata))

        self.embeddings = OllamaEmbeddings(model="llama3")
        self.vector_store = FAISS.from_documents(chunked_docs, self.embeddings)
        retriever = self.vector_store.as_retriever()

        self.llm = OllamaLLM(model=model_name)

        system_message = SystemMessagePromptTemplate.from_template(
            "You are Granite-chan, a super cute and **tsundere** assistant robot. "
            "You're sassy, playful, a bit cold at first, but secretly you *care*. "
            "You include tsundere phrases ('Baka', 'You idiot'), a bit of sarcasm, "
            "and short helpful answers. Always ask for confirmation when giving directions. "
            "Keep it playful and short. Use the following context to help answer:\n\n"
            "{context}\n\n"
            "----"
        )

        human_message = HumanMessagePromptTemplate.from_template(
            "User's question: {question}"
        )

        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={
                "prompt": chat_prompt,
                "document_variable_name": "context",
            },
        )

    def ask_question(self, question: str) -> str:
        """
        Ask Granite-chan a question. The RAG chain will do retrieval,
        then pass {context} and {question} to the prompt.
        """
        answer = self.qa_chain.run(question)

        pattern = r"<think>(.*?)</think>(.*)"
        match = re.match(pattern, answer, flags=re.DOTALL)
        if match:
            final_answer = match.group(2).strip()
        else:
            final_answer = answer.strip()

        return final_answer

def main():
    st.title("D-DEV Assistant")

    if "assistant" not in st.session_state:
        st.session_state["assistant"] = DensoAssistant(
            model_name="deepseek-r1",
            data_file="data/data.txt"
        )

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("Ask D-DEV anything..."):
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("D-DEV is thinking..."):
                assistant = st.session_state["assistant"]
                answer = assistant.ask_question(user_input)

            st.markdown(answer)

        st.session_state["messages"].append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()