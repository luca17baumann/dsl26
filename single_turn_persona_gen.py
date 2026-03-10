from openai import AsyncOpenAI
import asyncio
import os
import pandas as pd
from dsl26.tulu_prompts import prompt1, response1
from dotenv import load_dotenv

load_dotenv()

# load the dataset
df = pd.read_parquet("data/tulu-3-sft-instruction-following.parquet")
df = df.drop("id", axis=1)

MODEL = "zai-org/GLM-4.7-Flash"

client = AsyncOpenAI(
    api_key=os.environ.get("CSCS_SERVING_API"),
    base_url="https://api.swissai.cscs.ch/v1"
)


async def main() -> None:
    df_new = {}
    for i in range(5): # loop through the first 5 rows of the dataset
        tmp = {} # to store the row data
        # get the prompt from the dataset
        example = df.iloc[i]["prompt"]
        tmp["prompt"] = example
        # get the constraints from the dataset
        constraints = df.iloc[i]["constraints"]
        tmp["constraints"] = constraints
        
        # persona generation
        persona_response = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": f"Generate a detailed persona for the following example: {example} with the following constraints: {constraints}. Describe country of origin, place of birth, gender, age, languages and marital status."}]
        )
        persona = persona_response.choices[0].message.content
        tmp["persona"] = persona
        print(f"Example {i+1}: Generated Persona:\n{persona}\n{'-'*50}\n")
        
        # to store the conversation history
        result = []
        
        # user prompt
        user_response = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt1}]
        )
        generated_instructions = user_response.choices[0].message.content
        result.append({"role": "user", "content": generated_instructions})
        
        # assistant response
        assistant_response = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "assistant", "content": response1}]
        )
        result.append({"role": "assistant", "content": assistant_response.choices[0].message.content})
        tmp["conversation"] = result
        df_new[i] = tmp

    df_new = pd.DataFrame.from_dict(df_new, orient="index")
    df_new.to_parquet("data/tulu-3-sft-instruction-following-with-persona.parquet", index=False)

if __name__ == "__main__":
    asyncio.run(main())