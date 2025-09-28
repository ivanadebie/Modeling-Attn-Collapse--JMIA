dataset = []

"""
Prepare time series dataset for change point detection pipeline.
Assumes sentence/chunk-level scoring and hallucination labels are already computed.
This script structures the per-step features and saves them for downstream analysis.
"""
import pandas as pd
import numpy as np
from hall_riskindex.data_structuring import structure_data

# Load already-scored chunk/sentence data (replace with your actual source)
features = pd.read_csv("prepared_dataset_raw.csv").to_dict(orient="records")

# Structure as time series matrix
data_structured = structure_data(features)

# Save outputs
np.savez("prepared_dataset_structured.npz", **data_structured)
print("Time series structuring complete.")
