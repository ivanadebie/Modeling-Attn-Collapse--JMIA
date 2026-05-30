from datasets import load_dataset
import json

dataset = load_dataset("hotpot_qa", "distractor")
train = dataset["train"]

formatted = []
for ex in train:
    q = ex["question"]
    ans = ex["answer"]
    
    context = {title: sents for title, sents in ex["context"]}
    
    supporting_titles = set([sf[0] for sf in ex["supporting_facts"]])
    distractors = [title for title in context.keys() if title not in supporting_titles]
    
    formatted.append({
        "question": q,
        "answer": ans,
        "context": context,
        "supporting_facts": ex["supporting_facts"],
        "distractors": distractors,
        "type": ex["type"]
    })

with open("hotpotqa_structured.jsonl", "w", encoding="utf-8") as f:
    for ex in formatted:
        f.write(json.dumps(ex) + "\n")

print("saved", len(formatted), "examples")
