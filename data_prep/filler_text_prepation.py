import tiktoken

enc = tiktoken.encoding_for_model("gpt-4.1")  

def getTokens(text):
    return len(enc.encode(text))

def fillerText(tokens):
    enc = tiktoken.encoding_for_model("gpt-4.1")  
    
    ft = "the following section provides general guidance for navigating this document and maintaining a steady reading rhythm. it exists as structural padding, not as a source of facts, conclusions, or evidence relevant to any question. readers may skim it freely without missing essential content or changing the interpretation of adjacent material. the paragraphs use plain language, neutral tone, and consistent pacing to support clarity and reduce distraction. headings, transitions, and spacing are referenced in general terms to preserve flow across sections and pages. no specific entities, events, figures, or citations are introduced, and no claims are advanced or disputed. the emphasis is on readability, orientation, and coherence so that the main arguments can remain in sharp focus. where examples might appear in other contexts, this text intentionally avoids them to prevent unintended cues. if a reader pauses here, they should expect continuity rather than new information, analysis, or instructions. after this segment, the document resumes its core discussion, and any decisions should rely on subsequent content. overall, this material is supportive rather than informative, designed to keep momentum while remaining neutral, unobtrusive, and safely ignorable for factual evaluation."
    counter = 0
    text = ""

    while(counter < tokens):
        response = client.responses.create(
            model="gpt-4.1",
            input="Give me thousands of tokens of filler text, text which doesn't have meaning, but is logical. Here is an example: " + ft + " Please use proper English; use capitalization, punctuation, grammar, etc. Only respond with the filler text, no extra text like 'I can definitely add on to this filler text! NO EXTRA TEXT OR YOU WILL HAVE TO REDO IT.'",
            temperature=0.4,
        )
        txt = response.output_text
        text += txt
        length = getTokens(txt)
        counter += length

    return [text, counter]

filler = fillerText(40000)
txt = filler[0]
count = filler[1]
length = len(txt)

fill = open("filler.txt", "w")
fill.write(txt)