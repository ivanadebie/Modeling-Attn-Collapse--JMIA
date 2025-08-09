# Sets up basic table to export
def makeTable():
    export = {'Setup': {}, 'Prompt': {}, 'Question': {}, 'Gold Text': {}}

    for i in range(1, 10):
        export['Distractor ' + str(i) + ' Text'] = {}

    return export

# Makes the configuration for the 3x3x3 matrix for the questions
def makeConfig(ques):
    dens = ques // 9 % 3
    type = ques // 27 % 3
    pos = ques // 3 % 3
    return dens, type, pos

# Fills the table with prompts and distractors
def fillTable(numQues):
    export = makeTable()

    for i in range(numQues):
        dens, type, pos = makeConfig(i)

        # Inserts name of the config into table
        setup = ['Low', 'Medium', 'High'][dens] + '-' + ['N', 'T', 'P'][type] + '-' + ['Beg', 'Mid', 'End'][pos]
        export['Setup'][i + 1] = setup

        # Generate documents and prompt
        docs = make_documents(dens, type, pos, i)
        prompt = make_prompt(dens, type, pos, i, docs)

        # Inserts prompt, question, and correct answer into table
        export['Prompt'][i + 1] = prompt
        export['Question'][i + 1] = questions[i]
        export['Gold Text'][i + 1] = answers[i]

        # Inserts distractors into table
        for j in range(len(docs)):
            export['Distractor ' + str((j + 1)) + ' Text'][i + 1] = docs[j]

    data = pd.DataFrame(export)
    data.to_excel("Distractors.xlsx")
