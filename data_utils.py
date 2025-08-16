def save_results_to_csv(records, output_path):
    """Helper function to convert records and save to CSV"""
    results_data = [asdict(record) for record in records]
    
    # Add aggregated numeric columns and convert dictionaries to strings
    for item in results_data:
        # Remove fields that are not populated by this script
        item.pop('per_sample_scores', None)
        item.pop('risk_score_mean', None)
        item.pop('risk_score_p95', None)
        item.pop('self_consistency_vote', None)

        # Convert dictionaries to strings for CSV compatibility
        item['confidence'] = str(item['confidence'])
        item['binary_confidence'] = str(item['binary_confidence'])
        item['nli_probabilities'] = str(item['nli_probabilities']) # Also convert this new field

    df_output = pd.DataFrame(results_data)
    df_output.to_csv(output_path, index=False)

def save_diagnostics_to_csv(records, output_path):
    """Saves per-item diagnostics to a separate CSV file."""
    diagnostics_data = []
    for record in records:
        conf_dict = record.confidence if record.confidence else {}
        bin_dict = record.binary_confidence if record.binary_confidence else {}
        
        diag_item = {'qa_id': record.qa_id}
        if conf_dict:
            scores = list(conf_dict.values())
            diag_item['n_sentences'] = len(scores)
            diag_item['std_conf'] = np.std(scores) if len(scores) > 1 else 0.0
            diag_item['pos_rate'] = np.mean(list(bin_dict.values())) if bin_dict else 0.0
        else:
            diag_item['n_sentences'] = 0
            diag_item['std_conf'] = 0.0
            diag_item['pos_rate'] = 0.0
        diagnostics_data.append(diag_item)
        
    df_diagnostics = pd.DataFrame(diagnostics_data)
    df_diagnostics.to_csv(output_path, index=False)
