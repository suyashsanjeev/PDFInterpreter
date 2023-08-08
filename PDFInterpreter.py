import streamlit as st

import os
from apikey import apikey
from PyPDF2 import PdfReader

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document

CHUNK_SIZE = 200
SIMILARITY_SEARCH_K = 3

def clear_history():
    if 'history' in st.session_state:
        st.session_state['history'] = []

def extract_text_from_pdf(reader):
    """Extracts text from a PDF and returns it as a list of Document objects."""
    documents = []

    if len(reader.pages) < 5:
        text = "".join([page.extract_text() for page in reader.pages])

        # split text from document into chunks
        text = text.split()
        chunks = [' '.join(text[i:i + CHUNK_SIZE]) for i in range(0, len(text), CHUNK_SIZE)]

        # generate a list of Document objects from chunks
        for i, chunk in enumerate(chunks):
            page_number = i + 1
            metadata = {"page": page_number}
            document = Document(page_content=chunk, metadata=metadata)
            documents.append(document)
    else:
        # generate a list of Document objects from pages
        for page_num, page in enumerate(reader.pages):
            metadata = {"page": page_num + 1}
            document = Document(page_content=page.extract_text(), metadata=metadata)
            documents.append(document)

    return documents


def getEmbeddedText(documents: list[Document]) -> str:
    # perform similarity search on documents to find relevant information
    faiss_index = FAISS.from_documents(documents, OpenAIEmbeddings())
    docs = faiss_index.similarity_search(st.session_state['question'], k=SIMILARITY_SEARCH_K)

    # return combined relevant PDF content
    return "".join([f'{doc.metadata["page"]}:{doc.page_content[:500]}' for doc in docs])


def displaySupplementalInfo(embedded_text: str) -> None:
    with st.expander('Q&A History'):
        for q, a in reversed(st.session_state['history']):
            st.info(f'Question: {q}\n\nAnswer: {a}')

    with st.expander('Relevant PDF Content'):
        st.info(embedded_text)


def main():
    os.environ['OPENAI_API_KEY'] = apikey

    st.title('PDF Interpreter')
    uploaded_file = st.file_uploader("Upload PDF", type="pdf", on_change=clear_history)

    interpreter_template = PromptTemplate(
        input_variables=['question', 'pdf_content'],
        template='Given the text from the PDF: {pdf_content}, answer the following question: {question}'
    )

    llm = OpenAI(temperature=0.9)
    interpreter_chain = LLMChain(llm=llm, prompt=interpreter_template, verbose=True, output_key='answer')

    if uploaded_file is not None:
        reader = PdfReader(uploaded_file)

        question = st.text_input('Ask a question about the PDF content')

        if question:
            # extract text from PDF
            documents = extract_text_from_pdf(reader)

            # get relevant info from PDF documents
            embedded_text = getEmbeddedText(documents)

            # run the interpreter chain and display answer
            answer = interpreter_chain.run(question=question, pdf_content=embedded_text)
            st.write('Answer:', answer)
            
            if 'history' not in st.session_state:
                st.session_state['history'] = []
            st.session_state['history'].append((question, answer))

            displaySupplementalInfo(embedded_text)


if __name__ == '__main__':
    main()