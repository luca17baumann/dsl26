import openai
import os
from dotenv import load_dotenv

load_dotenv()

your_prompt = "What is ETH Zurich known for?"

client = openai.Client(
    api_key=os.environ.get("CSCS_SERVING_API"),
    base_url="https://api.swissai.cscs.ch/v1"
)
res = client.chat.completions.create(
    model="zai-org/GLM-4.7-Flash",
    messages=[
        {
            "content": your_prompt,
            "role": "user",
        }
    ],
    stream=True,
)

for chunk in res:
    if len(chunk.choices) > 0 and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)