export = {'Setup': {}, 'Prompt': {}, 'Question': {}, 'Gold Text': {}}

for i in range(1, 12):
    export['Distractor ' + str(i) + ' Text'] = {}

for i in range(10):
    dens = i // 9 % 3
    type = i // 27 % 3
    pos = i // 3 % 3
    setup = ['Low', 'Medium', 'High'][dens] + '-' + ['N', 'T', 'P'][type] + '-' + ['Beg', 'Mid', 'End'][pos]
    export['Setup'][i + 1] = setup

    docs = make_documents(dens, type, pos, i)
    prompt = make_prompt(dens, type, pos, i, docs)
    export['Prompt'][i + 1] = prompt

    export['Question'][i + 1] = questions[i]
    export['Gold Text'][i + 1] = answers[i]

    for j in range(len(docs)):
        export['Distractor ' + str((j + 1)) + ' Text'][i + 1] = docs[j]

# Uncomment these lines to make the export file
# data = pd.DataFrame(export)
# data.to_excel("Distractors.xlsx")
