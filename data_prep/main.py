import pandas as pd
from make_prompt import make_prompt, make_documents
from openai import OpenAI
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
    for dens in range(3):  # 3 distractor densities
        for type in range(3):  # 3 types
            for pos in range(3):  # 3 positions
                configs.append((dens, type, pos))
    return configs

# Fills the table with prompts and distractors
def fillTable(numQues, tokens):
    export = makeTable()
    row_index = 1

    for i in range(numQues):
        configs = makeConfig(i)  # Get all 27 configurations for this question
        
        for config_idx, (dens, type, pos) in enumerate(configs):
            # Inserts name of the config into table
            setup = ['Low', 'Medium', 'High'][dens] + '-' + ['N', 'T', 'P'][type] + '-' + ['Beg', 'Mid', 'End'][pos]
            print(f"Question {i+1}, Config {config_idx+1}/27: {setup}")
            
            if i < 4:
                continue
            if i == 4:
                return
                
            export['Setup'][row_index] = setup

            # Generate documents and prompt
            docs = make_documents(dens, type, pos, i, client)
            prompt = make_prompt(dens, type, pos, i, docs, tokens, client)  # Using 40000 tokens as default

            # Inserts prompt, question, and correct answer into table
            export['Prompt'][row_index] = prompt
            export['Question'][row_index] = questions[i]
            export['Gold Text'][row_index] = answers[i]

            # Inserts distractors into table
            for j in range(len(docs)):
                export['Distractor ' + str((j + 1)) + ' Text'][row_index] = docs[j]
            
            row_index += 1

    data = pd.DataFrame(export)
    data.to_excel("Distractors.xlsx")

# Calls the function
if __name__ == "__main__":
    fillTable(81, 8000)
