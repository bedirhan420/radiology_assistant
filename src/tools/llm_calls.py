import os
from typing import Type
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from dotenv import load_dotenv

from config import GEMINI_MODEL_NAME
from schemas.form_schemas import FormMRI,FormBloodTest
from config import FORM_TYPE

load_dotenv()

def get_llm():
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("Please add your Gemini API key to an .env file or environment variable with the name 'GOOGLE_API_KEY'.")
    
    return ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME)

class RouterSchema(BaseModel):
    form_type:FORM_TYPE
    
def get_routing_chain(llm):
    return llm.with_structured_output(RouterSchema)

def get_extraction_chain(llm,schema:Type[BaseModel]):
    return llm.with_structured_output(schema)