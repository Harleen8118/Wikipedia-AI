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
query = "What is AI?"
similar_response = db.similarity_search(query, k=1)

page_contents_array = [doc.page_content for doc in similar_response]

print(page_contents_array)