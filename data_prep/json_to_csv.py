#For using the json file in the data_prep folder to convert it to a csv file - DATASET w/ CONFIG prep
import json
import pandas as pd

def json_to_csv(input_filename, output_filename):
    """
    Convert a JSON file to CSV format.
    
    Args:
        input_filename (str): Path to the input JSON file
        output_filename (str): Path to the output CSV file
    """
    # Read the JSON file
    with open(input_filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Convert distractors list to string representation for CSV compatibility
    df['distractors'] = df['distractors'].astype(str)

    # Save to CSV file
    df.to_csv(output_filename, index=False, encoding='utf-8')

    print(f"CSV file '{output_filename}' has been created successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

if __name__ == "__main__":
    json_to_csv('domain1_data_3questions_8000_final.json', 'domain1_data_3questions_8000_final.csv')
