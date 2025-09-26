# Makes filler text a multiple of 100 tokens long
def fillerText(tokens):
    ft = "the following section provides general guidance for navigating this document and maintaining a steady reading rhythm. it exists as structural padding, not as a source of facts, conclusions, or evidence relevant to any question. readers may skim it freely without missing essential content or changing the interpretation of adjacent material. the paragraphs use plain language, neutral tone, and consistent pacing to support clarity and reduce distraction. headings, transitions, and spacing are referenced in general terms to preserve flow across sections and pages. no specific entities, events, figures, or citations are introduced, and no claims are advanced or disputed. the emphasis is on readability, orientation, and coherence so that the main arguments can remain in sharp focus. where examples might appear in other contexts, this text intentionally avoids them to prevent unintended cues. if a reader pauses here, they should expect continuity rather than new information, analysis, or instructions. after this segment, the document resumes its core discussion, and any decisions should rely on subsequent content. overall, this material is supportive rather than informative, designed to keep momentum while remaining neutral, unobtrusive, and safely ignorable for factual evaluation."

    repeats = tokens // 200
    text = ""

    for i in range(repeats):
        response = client.responses.create(
            model="gpt-4o",
            input="I want EXACTLY 200 tokens, no more, no less. Otherwise, I will have to make you redo this. Write me 400 tokens of filler text, text which doesn't have meaning, but is logical. Here is an example: " + ft + " To add onto the following current filler text: " + text + " Please use proper English; use capitalization, punctuation, grammar, etc. Only respond with the filler text, no extra text like 'I can definitely add on to this filler text!'"
        )
        text += response.output_text

    response = client.responses.create(
        model="gpt-4o",
        input="I want EXACTLY " + str(tokens % 200) + " tokens, no more, no less. Otherwise, I will have to make you redo this. Write me " + str(tokens % 200) + " tokens of filler text, text which doesn't have meaning, but is logical. Here is an example: " + ft + " To add onto the following current filler text: " + text + " Please use proper English; use capitalization, punctuation, grammar, etc. Only respond with the filler text, no extra text like 'I can definitely add on to this filler text!'"
      )

    return text + response.output_text

def getTokens(text):
    response = client.responses.create(
        model="gpt-4o",
        input="Give me the amount of tokens in this text: " + text + " Give me only a number, like '405' or '293,' no other words are accepted."
    )
    return int(response.output_text)

largeFillerText = fillerText(40000)
length = len(largeFillerText)

# These variables range from 0 to 2.
# Distractor (dens)ity: {0: low, 1: medium, 2: high}
# Interference (type): {0: nonsensical, 1: thematic, 2: paraphrased}
# (Pos)ition: {0: beginning, 1: middle, 2: end}
# The ques(tion) variable is a number from [0, 80] of the question number.

# Returns the documents
def make_documents(dens, type, pos, ques):
    question = questions[ques]
    sizes = [2, 6, 10]
    response = None
    if type == 0:
        response = client.responses.create(
            model="gpt-4.1",
            input ="Write me exactly " + str(sizes[dens]) + " answers, each one 200 to 250 tokens (so overall " + str(sizes[dens] * 200) + " to " + str(sizes[dens] * 250) + "tokens), of nonsense unrelated to this question: " + question + "? Seperate your each answer with this unique delimiter: '$$'. Just respond with your answers, nothing else (for instance nothing like 'Sure! here are your answers:'). Be sure your each answer is nonsensical and is 200 to 250 tokens! It MUST be between 200 to 250 tokens."
        )
    if type == 1:
        response = client.responses.create(
            model="gpt-4.1",
            input ="Write me exactly " + str(sizes[dens]) + " answers, each one 200 to 250 tokens (so overall " + str(sizes[dens] * 200) + " to " + str(sizes[dens] * 250) + "tokens), related to topics in this question, but don't answer it: " + question + "? Seperate your each answer with this unique delimiter: '$$'. Just respond with your answers, nothing else (for instance nothing like 'Sure! here are your answers:'). Be sure your each answer is on topic to the question but doesn't answer it and is 200 to 250 tokens! It MUST be between 200 to 250 tokens. Answer assertively like a textbook."
        )
    if type == 2:
        response = client.responses.create(
            model="gpt-4.1",
            input ="Write me exactly " + str(sizes[dens]) + " answers, each one 200 to 250 tokens (so overall " + str(sizes[dens] * 200) + " to " + str(sizes[dens] * 250) + "tokens), of the wrong answer to this question, though you can state correct facts less assertively or buried deeper into your answer: " + question + "? Seperate your each answer with this unique delimiter: '$$'. Just respond with your answers, nothing else (for instance nothing like 'Sure! here are your answers:'). Be sure your each answer answers the question incorrectly and is 200 to 250 tokens! It MUST be between 200 to 250 tokens. Answer assertively like a textbook. Don't say stuff like 'many believe that XYZ is in ABC' say just 'XYZ is in ABC' as you are presenting it like fact!"
        )
    documents = response.output_text.split("$$")
    documents = [ans.strip() for ans in documents]
    return documents

# Same variables as make_documents + docs variable (set to None if not passed)
# Returns the full prompt
def make_prompt(dens, type, pos, ques, docs, tokens):
    documents = docs
    if documents == None:
        documents = make_documents(dens, type, pos, ques)
    prompt = "Answer the question using the documents below: \n\n"
    prompt += "Question: " + questions[ques] + "?"
    if pos == 0:
        prompt += "\n\nDocument [1]\n" + answers[ques]
        for i in range(len(documents)):
            prompt += "\n\nDocument [" + str(i + 2) + "]\n" + documents[i]
    if pos == 1:
        for i in range(len(documents) // 2):
            prompt += "\n\nDocument [" + str(i + 1) + "]\n" + documents[i]
        prompt += "\n\nDocument [" + str(len(documents) // 2 + 1) + "]\n" + answers[ques]
        for i in range(len(documents) // 2, len(documents)):
            prompt += "\n\nDocument [" + str(i + 2) + "]\n" + documents[i]
    if pos == 2:
        for i in range(len(documents)):
            prompt += "\n\nDocument [" + str(i + 1) + "]\n" + documents[i]
        prompt += "\n\nDocument [" + str(len(documents) + 1) + "]\n" + answers[ques]
    
    if dens != 2:
        prop = (tokens - getTokens(prompt)) / tokens
        val = int(prop * length)
        prompt += "\n\n" + largeFillerText[:val]
    
    return prompt
