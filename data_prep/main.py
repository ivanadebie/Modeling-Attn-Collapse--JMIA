import pandas as pd
from make_prompt import make_prompt, make_documents
from openai import OpenAI
import json
import os
from dotenv import load_dotenv
import time

load_dotenv("../.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Sets up basic table to export
def makeTable():
    export = {'Setup': {}, 'Prompt': {}, 'Question': {}, 'Gold_text': {}, 'Row_index': {}}

    for i in range(1, 11):
        export['Distractor_' + str(i) + '_text'] = {}

    return export

# Makes all 27 configurations for a given question
def makeConfig(ques):
    configs = []
    for density in range(3):  # 3 distractor densityities
        for inference_type in range(3):  # 3 types
            for position in range(3):  # 3 positions
                configs.append((density, inference_type, position))
    return configs

# Fills the table with prompts and distractors
def fillTable(tokens, question_data_path, domain, output_path):
    export = makeTable()
    row_index = 1

    question_data = []
    with open(question_data_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            question_data.append(json.loads(line))

    # Load existing JSON results
    json_filename = f"{output_path.split('.')[0]}.jsonl"
    try:
        with open(json_filename, 'r') as json_file:
            all_results = json.load(json_file)
    except FileNotFoundError:
        all_results = []

    num_questions = len(question_data)
    for i in range(num_questions):
        start_time_question = time.time()
        configs = makeConfig(i)  # Get all 27 configurations for this question
        
        for config_idx, (density, inference_type, position) in enumerate(configs):

            start_time = time.time()    
            # Inserts name of the config into table
            setup = ['Low', 'Medium', 'High'][density] + '-' + ['N', 'T', 'P'][inference_type] + '-' + ['Beg', 'Mid', 'End'][position]
            print(f"Question {i+1}, Config {config_idx+1}/27: {setup}")
            # Generate documents and prompt
            docs = make_documents(density, inference_type, position, question_data[i]['question'], client)
            prompt = make_prompt(density, inference_type, position, question_data[i]['question'], question_data[i]['gold_text'], docs, tokens, client)
            # Inserts prompt, question, and correct answer into table
            # print("Prompt: ", prompt)
            
            export['Setup'][row_index] = setup
            export['Prompt'][row_index] = prompt
            export['Question'][row_index] = question_data[i]['question']
            export['Gold_text'][row_index] = question_data[i]['gold_text']
            export['Row_index'][row_index] = row_index

            print(len(docs), time.time() - start_time, "seconds for config", config_idx+1)
            print("================================================")
            # Inserts distractors into table
            for j in range(len(docs)):
                export['Distractor_' + str((j + 1)) + '_text'][row_index] = docs[j]
            # print(question_data[i]['question'], question_data[i]['gold_text'])
            
            
            # Add current result to the list
            json_result = {
                'question_id': i,
                'config_idx': config_idx,
                'domain': domain,
                'config': setup,
                'prompt': prompt,
                'question': question_data[i]['question'],
                'gold_text': question_data[i]['gold_text'],
                'distractors': docs,
                'row_index': row_index
            }
            all_results.append(json_result)
            
            # Save updated results back to JSON file
            with open(json_filename, 'w') as json_file:
                json.dump(all_results, json_file, indent=2)
 
            row_index += 1
        print("Completed question", i+1, "in", time.time() - start_time_question, "seconds")


    data = pd.DataFrame(export)
    data.to_csv(output_path, sep="\t", index=False)

# Calls the function
if __name__ == "__main__":
    tokens=8000
    fillTable(tokens=tokens, question_data_path='qbank_by_domain/01_nq_closed_book.jsonl', domain='01_nq_closed_book', output_path=f'domain1_data_{tokens}.tsv')
