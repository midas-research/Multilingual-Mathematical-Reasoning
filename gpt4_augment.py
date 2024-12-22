from openai import OpenAI
from tqdm import tqdm 
import os

FILE_PATH = "dataset/file/path"
OUTPUT = "output/file/path"

os.environ["OPENAI_API_KEY"] = "openai/key"
client = OpenAI()

import json 
with open(FILE_PATH, "r") as f:
    data = json.load(f)

def prompt_generator(sample):
    prompt = f"""Your task is to create similar conceptual and diverse difficulty level(either similar simple or same or complex) question and answer using provided problem.\n\n# Problem:\n## Question: {sample["Example"]}\n## Answer: {sample["refined_solution"]}\n\n# New Problem:\n"""

    return prompt

with open(OUTPUT, "a") as f:
    for i in tqdm(range(len(data))):
        for _ in range(10):
            prompt = prompt_generator(data[i])

            completion = client.chat.completions.create(
                model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )
            
            hindi_q_a = completion.choices[0].message.content
            new_data = {
                "new_data": hindi_q_a
            }

            json.dump(new_data, f, ensure_ascii=False)
            f.write("\n")