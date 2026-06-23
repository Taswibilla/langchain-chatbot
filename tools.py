from langchain.tools import Tool
from dotenv import load_dotenv
import os

load_dotenv()

def explain_like_2(text: str) -> str:
    return f"Let me explain '{text}' simply — it will be answered by the main LLM agent."

tools = [
    Tool(
        name="SimplifyExplainer",
        func=explain_like_2,
        description="Use this tool to explain any concept in extremely simple words, like for a 2 year old child"
    )
]