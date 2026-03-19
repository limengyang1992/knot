# Global API configuration
import os
from openai import OpenAI

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY environment variable is not set. "
                     "Please run: export DEEPSEEK_API_KEY=your_key_here")

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
MODEL = "deepseek-chat"
