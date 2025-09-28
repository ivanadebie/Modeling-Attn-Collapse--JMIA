import pandas as pd
from make_prompt import make_prompt, make_documents
from openai import OpenAI
import json

client = OpenAI(api_key="Open AI Key")


# Sets up basic table to export
def makeTable():
    export = {'Setup': {}, 'Prompt': {}, 'Question': {}, 'Gold Text': {}}

    for i in range(1, 10):
        export['Distractor ' + str(i) + ' Text'] = {}

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
def fillTable(tokens, question_data_path, output_path):
    export = makeTable()
    row_index = 1
    question_data = []
    with open(question_data_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            question_data.append(json.loads(line))

    num_questions = len(question_data)
    for i in range(num_questions):
        configs = makeConfig(i)  # Get all 27 configurations for this question
        
        for config_idx, (density, inference_type, position) in enumerate(configs):
            # Inserts name of the config into table
            setup = ['Low', 'Medium', 'High'][density] + '-' + ['N', 'T', 'P'][inference_type] + '-' + ['Beg', 'Mid', 'End'][position]
            print(f"Question {i+1}, Config {config_idx+1}/27: {setup}")
            
            # Generate documents and prompt
            docs = make_documents(density, inference_type, position, i, client)
            prompt = make_prompt(density, inference_type, position, i, docs, tokens, client)  # Using 40000 tokens as default

            # Inserts prompt, question, and correct answer into table
            export['Setup'][row_index] = setup
            export['Prompt'][row_index] = prompt
            export['Question'][row_index] = question_data[i]['question']
            export['Gold Text'][row_index] = question_data[i]['gold_text']

            # Inserts distractors into table
            # for j in range(len(docs)):
            #     export['Distractor ' + str((j + 1)) + ' Text'][row_index] = docs[j]
            # print(question_data[i]['question'], question_data[i]['gold_text'])
            # print("================================================")
            row_index += 1
        
        if i == 0:
            break

    data = pd.DataFrame(export)
    data.to_excel(output_path)

# Calls the function
if __name__ == "__main__":
    fillTable(tokens=8000, question_data_path='qbank_by_domain/01_nq_closed_book.jsonl', output_path='domain1_data_final.xlsx')
