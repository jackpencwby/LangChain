from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("data.txt", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
chunks = text_splitter.split_documents(documents)
print(chunks)

