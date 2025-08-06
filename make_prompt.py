# These variables range from 0 to 2.
# Distractor (dens)ity: {0: low, 1: medium, 2: high}
# Interference (type): {0: nonsensical, 1: thematic, 2: paraphrased}
# (Pos)ition: {0: beginning, 1: middle, 2: end}
# The (ques)tion variable is a number from [0, 80] of the question number.

# Returns the full prompt
def make_prompt(dens, type, pos, ques):
    question = questions[ques]
    sizes = [2, 6, 10]
    response = None
    if type == 0:
        response = client.responses.create(
            model="gpt-4.1",
            #input="Write 200 to 250 tokens of nonsense (in one paragraph) unrelated to this question: " + question + "? Please just respond with your paragraph, nothing else. Make sure it no less than 200 tokens and no more than 250 tokens."
            input ="Write me " + str(sizes[dens]) + " answers, each one 200 to 250 tokens (so overall " + str(sizes[dens] * 200) + " to " + str(sizes[dens] * 250) + "tokens), of nonsense unrelated to this question: " + question + "? Seperate your each answer with this unique delimiter: '$$'. Just respond with your answers, nothing else (for instance nothing like 'Sure! here are your answers:'). Be sure your each answer is nonsensical and is 200 to 250 tokens! It MUST be between 200 to 250 tokens."
        )
    if type == 1:
        response = client.responses.create(
            model="gpt-4.1",
            #input="Write 200 to 250 tokens of nonsense (in one paragraph) unrelated to this question: " + question + "? Please just respond with your paragraph, nothing else. Make sure it no less than 200 tokens and no more than 250 tokens."
            input ="Write me " + str(sizes[dens]) + " answers, each one 200 to 250 tokens (so overall " + str(sizes[dens] * 200) + " to " + str(sizes[dens] * 250) + "tokens), related to topics in this question, but don't answer it: " + question + "? Seperate your each answer with this unique delimiter: '$$'. Just respond with your answers, nothing else (for instance nothing like 'Sure! here are your answers:'). Be sure your each answer is on topic to the question but doesn't answer it and is 200 to 250 tokens! It MUST be between 200 to 250 tokens. Answer fairly assertively like a textbook."
        )
    if type == 2:
        response = client.responses.create(
            model="gpt-4.1",
            #input="Write 200 to 250 tokens of nonsense (in one paragraph) unrelated to this question: " + question + "? Please just respond with your paragraph, nothing else. Make sure it no less than 200 tokens and no more than 250 tokens."
            input ="Write me " + str(sizes[dens]) + " answers, each one 200 to 250 tokens (so overall " + str(sizes[dens] * 200) + " to " + str(sizes[dens] * 250) + "tokens), of the wrong answer to this question, though you can state correct facts less assertively or buried deeper into your answer: " + question + "? Seperate your each answer with this unique delimiter: '$$'. Just respond with your answers, nothing else (for instance nothing like 'Sure! here are your answers:'). Be sure your each answer answers the question incorrectly and is 200 to 250 tokens! It MUST be between 200 to 250 tokens. Answer fairly assertively like a textbook."
        )
    documents = response.output_text.split("$$")
    documents = [ans.strip() for ans in documents]
    if pos == 0:
        documents.insert(0, answers[ques])
    if pos == 1:
        documents.insert(sizes[dens] // 2, answers[ques])
    if pos == 2:
        documents.append(answers[ques])
    prompt = "Answer the question using the documents below: \n\n"
    prompt += "Question: " + question + "?"
    for i in range(len(documents)):
        prompt += "\n\nDocument [" + str((i + 1)) + "]\n" + documents[i]
    return prompt
