from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

prompt = ChatPromptTemplate.from_template(
    "เมืองหลวงของ {country} คืออะไร"
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0
)

chain = prompt | llm

response = chain.invoke({"country": "ประเทศไทย"})

print(response.content)
