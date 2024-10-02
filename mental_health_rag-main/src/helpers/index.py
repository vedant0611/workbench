import os

import src.helpers.langchain_helper as langchain_helper

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

def initializaion():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = "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"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_PROJECT"] = "RAG_With_Memory"

    os.environ["GOOGLE_API_KEY"] = 'AIzaSyCBGtPby__8TQnnKXh5V7I9B8J1Wwdgrlo'

def models():
    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    model = ChatGoogleGenerativeAI(model="gemini-1.0-pro",convert_system_message_to_human=True)

    return gemini_embeddings, model