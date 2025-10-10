import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# --- Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# --- Load your sample JSON (replace with your path)
payload = {
  "question_id": 0,
  "config_idx": 0,
  "domain": "01_nq_closed_book",
  "config": "Low-N-Beg",
  "prompt":
    "Question: who sang i dreamed a dream in les miserables? \n\n"
    "Answer the question using the documents below: \n\n\n\n"
    "Document [1]\n ... (truncated for brevity) ...",
  "question": "who sang i dreamed a dream in les miserables",
  "gold_text": "...",
  "distractors": ["..."],
  "row_index": 1
}

# Closed-book means: DO NOT feed docs. Use just the question.
question = payload["question"]

# Mistral-Instruct chat template
messages = [
    {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
    {"role": "user", "content": question}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        temperature=0.2,
        top_p=0.9
    )

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# If the template echoes the prompt, keep only the assistant’s last turn:
print("\n=== MODEL OUTPUT ===\n")
print(text.split(messages[-1]["content"])[-1].strip())