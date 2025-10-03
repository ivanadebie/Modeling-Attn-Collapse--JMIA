from dataclasses import asdict
import pandas as pd

def save_results_to_csv(records, output_path):
    """Helper function to convert records and save to CSV"""
    results_data = [asdict(record) for record in records]
    
    # Add aggregated numeric columns and convert dictionaries to strings
    for item in results_data:

        # Convert dictionaries to strings for CSV compatibility
        # item['confidence'] = str(item['confidence'])

        df_output = pd.DataFrame(results_data)
        df_output.to_csv(output_path, index=False)

