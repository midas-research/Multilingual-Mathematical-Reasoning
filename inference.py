import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import transformers
from tqdm import tqdm
import os

MODEL = "model/path"
DATASET = "dataset/file/path"
OUTPUT = "output/file/path"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
pipeline = transformers.pipeline(
    "text-generation",
    model=MODEL,
    # model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

def sequence_inference(prompt_sample, pipeline):
    sequences = pipeline(
        prompt_sample,
        max_length=2048,
    )

    predict = sequences[0]['generated_text']
        
    return predict

with open(DATASET, "r") as f:
    test = json.load(f)

def prompt_generator(sample):
    prompt = f"""Below is an ENGLISH/HINDI instruction that describes a task. Write a response that appropriately completes the request. Beaware of wrong calculation and do not repeat it.\n\n### Instruction:\n{sample['problem']}\n\n### Response: """
    return prompt 

with open(OUTPUT, "a") as f:
    for test_sample in tqdm(test):
        prompt = prompt_generator(test_sample)
        predict = sequence_inference(prompt, pipeline)

        predict_json = {
            "question": test_sample["problem"],
            "target": test_sample["solution"],
            "predict": predict, 
        }

        json.dump(predict_json, f, ensure_ascii=False)
        f.write("\n")
