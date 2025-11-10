from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

loader = TextLoader("data.txt", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
chunks = text_splitter.split_documents(documents)

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever()

prompt = ChatPromptTemplate.from_messages([
    ("system", "ใช้ข้อมูลจากเอกสารในการตอบคำถามให้เข้าใจได้ง่าย กระชับ เเละชัดเจน"),
    ("human", "ข้อมูลที่เกี่ยวข้อง: {context}\nคำถาม: {question}")
])

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke("เครื่อง AED คืออะไร")
print(response)




