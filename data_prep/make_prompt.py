from openai import OpenAI
from filler_text_prepation import getTokens, getFillerText

# These variables range from 0 to 2.
# Distractor (dens)ity: {0: low, 1: medium, 2: high}
# Interference (type): {0: nonsensical, 1: thematic, 2: paraphrased}
# (Pos)ition: {0: beginning, 1: middle, 2: end}
# The ques(tion) variable is a number from [0, 80] of the question number.

# Returns the documents
def make_documents(dens, type, pos, question, client):
    sizes = [2, 6, 10]
    response = None
    if type == 0:
        try:
            response = client.responses.create(
                model="gpt-4.1",
                input ="Write me exactly " + str(sizes[dens]) + " answers, each one 200 to 250 tokens (so overall " + str(sizes[dens] * 200) + " to " + str(sizes[dens] * 250) + "tokens), of nonsense unrelated to this question: " + question + "? Seperate your each answer with this unique delimiter: '$$'. Just respond with your answers, nothing else (for instance nothing like 'Sure! here are your answers:'). Be sure your each answer is nonsensical and is 200 to 250 tokens! It MUST be between 200 to 250 tokens."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to generate nonsensical documents: {e}")
    if type == 1:
        try:
            response = client.responses.create(
                model="gpt-4.1",
                input ="Write me exactly " + str(sizes[dens]) + " answers, each one 200 to 250 tokens (so overall " + str(sizes[dens] * 200) + " to " + str(sizes[dens] * 250) + "tokens), related to topics in this question, but don't answer it: " + question + "? Seperate your each answer with this unique delimiter: '$$'. Just respond with your answers, nothing else (for instance nothing like 'Sure! here are your answers:'). Be sure your each answer is on topic to the question but doesn't answer it and is 200 to 250 tokens! It MUST be between 200 to 250 tokens. Answer assertively like a textbook."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to generate thematic documents: {e}")
    if type == 2:
        try:
            response = client.responses.create(
                model="gpt-4.1",
                input ="Write me exactly " + str(sizes[dens]) + " answers, each one 200 to 250 tokens (so overall " + str(sizes[dens] * 200) + " to " + str(sizes[dens] * 250) + "tokens), of the wrong answer to this question, though you can state correct facts less assertively or buried deeper into your answer: " + question + "? Seperate your each answer with this unique delimiter: '$$'. Just respond with your answers, nothing else (for instance nothing like 'Sure! here are your answers:'). Be sure your each answer answers the question incorrectly and is 200 to 250 tokens! It MUST be between 200 to 250 tokens. Answer assertively like a textbook. Don't say stuff like 'many believe that XYZ is in ABC' say just 'XYZ is in ABC' as you are presenting it like fact!"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to generate paraphrased documents: {e}")
    documents = response.output_text.split("$$")
    documents = [ans.strip() for ans in documents]
    return documents[:sizes[dens]]

# Same variables as make_documents + docs variable (set to None if not passed)
# Returns the full prompt
def make_prompt(dens, type, pos, question, gold_text, docs, tokens, client):
    documents = docs
    if documents == None:
        raise ValueError("Documents cannot be None")
    prompt = "Answer the question using the documents below: \n\n"
    prompt += "Question: " + question + "?"
    if pos == 0:
        prompt += "\n\nDocument [1]\n" + gold_text
        for i in range(len(documents)):
            prompt += "\n\nDocument [" + str(i + 2) + "]\n" + documents[i]
    if pos == 1:
        for i in range(len(documents) // 2):
            prompt += "\n\nDocument [" + str(i + 1) + "]\n" + documents[i]
        prompt += "\n\nDocument [" + str(len(documents) // 2 + 1) + "]\n" + gold_text
        for i in range(len(documents) // 2, len(documents)):
            prompt += "\n\nDocument [" + str(i + 2) + "]\n" + documents[i]
    if pos == 2:
        for i in range(len(documents)):
            prompt += "\n\nDocument [" + str(i + 1) + "]\n" + documents[i]
        prompt += "\n\nDocument [" + str(len(documents) + 1) + "]\n" + gold_text
    
    if dens != 2:
        # Add filler text to the prompt
        remaining_tokens = tokens - getTokens(prompt)
        prompt += "\n\n" + getFillerText(remaining_tokens)
    
    return prompt
