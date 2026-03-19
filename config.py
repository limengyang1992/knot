# Global API configuration
from openai import OpenAI

DEEPSEEK_API_KEY = "sk-bf7741dc268c4cbaac9e723231e04eaf"
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
MODEL = "deepseek-chat"
