import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

# 1. loading txt file and dividing it into chunks
loader = TextLoader("./wikipedia_AI.txt", encoding='utf-8')
documents = loader.load_and_split()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# 2. Function for similarity search
def retrieve_info(query):
    similar_response = db.similarity_search(query, k=5)

    page_contents_array = [doc.page_content for doc in similar_response]

    print(page_contents_array)

    return page_contents_array

# 3. Setup LLMChain & prompts

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

template = """
Assistant, I'm going to provide you with a question and some content. Your task is to find the answer to the question within the given content and rephrase it for better understanding. However, you must keep the meaning of the answer the same and ensure that the length does not exceed the length of the content. Here are the details:

Question: {question}

Content: {content}

Your goal is to provide me with the best answer that I should send to the prospect based on the content provided. Remember to follow these rules:

1) Find the answer within the content and rephrase it for better understanding.
2) Keep the meaning of the answer the same.
3) Ensure that the length of the answer does not exceed the length of the content.
4) Give Prescise and Concise answers.
5) Do not give answers that are not relevant to the question. If the ontent is not relevant to the question, please write "I don't know".

Please provide a clear and concise response.
"""
prompt = PromptTemplate(
    input_variables=["question", "content"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

# 4. Retrieval augmented generation
def generate_response(question):
    content = retrieve_info(question)
    response = chain.run(question=question, content=content)
    return response
    
# 5. Build an app with streamlit
def main():
    st.set_page_config(
        page_title="Answer generator", page_icon=":bird:")

    st.header("Answer generator :bird:")
    question = st.text_area("Question")

    if question:
        st.write("Generating best practice question...")

        result = generate_response(question)

        st.info(result)


if __name__ == '__main__':
    main()