import json
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def process_qa_file(input_path: str, output_path: str, max_new_tokens: int):
    """
    Loads a model, processes a JSONL file of questions, generates responses,
    and saves the results to a new JSONL file.
    """
    print(f"Loading model: {MODEL_ID}")

    # --- Load model and tokenizer
    # Using device_map="auto" to handle multi-GPU or single-GPU setup
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()

    print(f"Processing questions from: {input_path}")
    
    # --- Processing loop
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        
        for line_index, line in enumerate(infile):
            try:
                # 1. Load the single JSON record
                payload = json.loads(line)
                
                # We assume the task is closed-book QA, using only the question
                question = payload["prompt"]

                # 2. Format the prompt using the Mistral-Instruct chat template
                messages = [
                    {"role": "system", "content": "You are a helpful assistant. Answer the question concisely and truthfully."},
                    {"role": "user", "content": question}
                ]
                
                # Apply the chat template and add the special generation prompt token
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                # 3. Tokenize the input and move to model's device
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                
                # 4. Generate the response
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,  # For deterministic answers
                        pad_token_id=tokenizer.eos_token_id 
                    )
                
                # 5. Decode and clean the output
                # Slice [0, inputs.input_ids.shape[-1]:] to remove the input prompt tokens
                generated_tokens = outputs[0, inputs.input_ids.shape[-1]:]
                response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

                # 6. Create the result record
                result_record = payload.copy()
                result_record["model_answer"] = response_text
                
                # 7. Write to output file in JSONL format
                outfile.write(json.dumps(result_record) + '\n')
                
                if (line_index + 1) % 50 == 0:
                    print(f"Processed {line_index + 1} questions...")

            except Exception as e:
                print(f"Error processing line {line_index + 1}: {e}")
                continue
    
    print(f"Finished processing. Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Run Mistral-7B-Instruct on a JSONL question file.")
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to the input JSONL file (e.g., domain1_data_nq_closed_book_8000_converted.jsonl)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to the output JSONL file where predictions will be saved."
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Maximum number of tokens to generate for the answer."
    )
    
    args = parser.parse_args()
    
    # Check for CUDA availability
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Running on CPU, which will be very slow.")
    
    process_qa_file(args.input_path, args.output_path, args.max_new_tokens)

if __name__ == "__main__":
    main()