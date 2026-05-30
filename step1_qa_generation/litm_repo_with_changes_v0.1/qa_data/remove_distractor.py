import json
import re
import argparse

def remove_distractors(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            for line in f_in:
                if not line.strip():
                    continue
                    
                # Parse the JSON line
                data = json.loads(line)
                
                prompt = data.get('prompt', '')
                distractors = data.get('distractors', [])
                
                # Remove each distractor from the prompt
                for distractor in distractors:
                    # Escape the distractor text so regex doesn't misinterpret special characters
                    escaped_distractor = re.escape(distractor)
                    
                    # Regex pattern to match "Document [X]\n" + the distractor text + trailing newlines
                    pattern = r'Document \[\d+\]\n' + escaped_distractor + r'\n*'
                    
                    # Substitute the matched pattern with an empty string
                    prompt = re.sub(pattern, '', prompt)
                
                # Update the prompt with the cleaned text
                data['prompt'] = prompt
                
                # Remove the 'distractors' list from the JSON object
                if 'distractors' in data:
                    del data['distractors']
                    
                # Write the updated JSON object back to the new file
                f_out.write(json.dumps(data) + '\n')

        print(f"Finished processing! Cleaned data saved to '{output_file}'.")
        
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Remove distractors from a JSONL dataset.")
    parser.add_argument(
        "-i", "--input", 
        required=True, 
        help="Path to the input JSONL file"
    )
    parser.add_argument(
        "-o", "--output", 
        required=True, 
        help="Path to the output JSONL file"
    )
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Run the function with the provided arguments
    remove_distractors(args.input, args.output)