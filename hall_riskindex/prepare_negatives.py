import pandas as pd

# Load the results_nli_labeled.csv file
results_nli = pd.read_csv('results_nli_labeled.csv')

# Apply the threshold rule to create the label
threshold = 0.2
results_nli['label_not_hallu'] = results_nli['hallucination_score'].apply(
    lambda x: 'not hallu' if x <= threshold else 'hallu'
)

# Calculate the percentage of negative examples ("not hallu")
num_not_hallu = (results_nli['label_not_hallu'] == 'not hallu').sum()
total_examples = len(results_nli)
percentage_not_hallu = (num_not_hallu / total_examples) * 100

print(f"Percentage of negative examples (not hallu): {percentage_not_hallu:.2f}%")

# Optionally, save the labeled DataFrame to a new CSV
results_nli.to_csv('results_nli_labeled_with_negatives.csv', index=False)
