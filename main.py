import litellm
import os

response = litellm.completion(
    model="ollama_chat/gpt-oss:20b",
    messages=[{"content": "Hello, how are you?", "role": "user"}],
    api_base="http://localhost:11434",
)
print(response)
