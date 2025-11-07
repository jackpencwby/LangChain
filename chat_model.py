from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser
from dotenv import load_dotenv

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", "คุณเป็นผู้เชี่ยวชาญด้าน {expertise} ที่ตอบคำถามเข้าใจได้ง่าย กระชับ เเละชัดเจน"),
    ("human", "อธิบายเกี่ยวกับ {topic}")
])

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0
)

chain = prompt | llm | StrOutputParser()

response = chain.invoke(
    {
        "expertise": "ฟิสิกส์",
        "topic": "กฎการเคลื่อนที่ของนิวตัน"
    }
)

print(response)