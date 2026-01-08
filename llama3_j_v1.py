with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
    for line_index, line in enumerate(infile):
        try:
            line = line.strip()
            if not line:
                continue

            payload = json.loads(line)
            question = payload["prompt"] 

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ]

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            generated_tokens = outputs[0, inputs.input_ids.shape[-1]:]
            response_text = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            ).strip()

            result_record = payload.copy()
            result_record["model_answer"] = response_text

            outfile.write(json.dumps(result_record) + "\n")

            if (line_index + 1) % 50 == 0:
                print(f"Processed {line_index + 1} questions...")

        except Exception as e:
            print(f"Error processing line {line_index + 1}: {e}")
