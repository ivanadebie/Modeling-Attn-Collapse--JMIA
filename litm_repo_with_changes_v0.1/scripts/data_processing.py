from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")

def count_tokens(text):
    """Counts the number of tokens in a given text string."""
    return len(tok.encode(text))

def classify_evidence_position(gold_idx, n_docs):
    """Classifies the position of a gold document into 'beginning', 'middle', or 'end'."""
    pct = (gold_idx + 1) / n_docs
    if pct <= 0.33:
        return "beginning"
    elif pct <= 0.66:
        return "middle"
    else:
        return "end"

def distractor_ratio(ctxs, gold_idx):
    """Calculates the ratio of distractor tokens to total tokens in a list of contexts."""
    total = sum(count_tokens(d["title"] + "\n" + d["text"]) for d in ctxs)
    distractor = sum(count_tokens(d["title"] + "\n" + d["text"])
                      for i, d in enumerate(ctxs) if i != gold_idx)
    return distractor / max(total, 1)

def bucket_density(ratio):
    """Buckets the distractor ratio into 'low', 'medium', or 'high'."""
    if ratio < 0.25:
        return "low"
    elif ratio < 0.60:
        return "medium"
    else:
        return "high"

def find_distractor_tokens_in_response(response, ctxs, gold_idx):
    """
    Finds tokens from distractor documents that appear in the LLM's response.
    Returns the count of such tokens.
    """
    distractor_text = "".join([d["title"] + " " + d["text"] for i, d in enumerate(ctxs) if i != gold_idx])
    
    # Tokenize the response and the distractor text
    response_tokens = tok.encode(response)
    distractor_tokens = tok.encode(distractor_text)
    
    # Count how many of the response tokens are also in the distractor text
    distractor_set = set(distractor_tokens)
    tokens_found = sum(1 for token in response_tokens if token in distractor_set)
    
    return tokens_found