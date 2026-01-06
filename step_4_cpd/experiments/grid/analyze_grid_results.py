import pandas as pd

results = pd.read_csv('output/grid/grid_search_results.csv')

print('GRID SEARCH RESULTS SUMMARY')
print('='*70)
print(f'\nTotal configurations tested: {len(results)}')
print(f'Successful (with data): {(results["status"] == "success").sum()}')

# Filter successful ones
success = results[results['status'] == 'success']

print(f'\nDetection Statistics:')
print(f'  Avg detections: {success["avg_detections"].mean():.2f}')
print(f'  % with detections: {success["pct_with_detections"].mean():.1f}%')
print(f'  Range: {success["pct_with_detections"].min():.1f}% to {success["pct_with_detections"].max():.1f}%')

print(f'\nTop 10 Configurations:')
top = success.nlargest(10, 'pct_with_detections')[['evidence_pos', 'density', 'interference', 'pct_with_detections', 'avg_detections']]
for idx, row in top.iterrows():
    print(f'  {row["evidence_pos"]:5s} | {row["density"]:7s} | {row["interference"]:3s} | {row["pct_with_detections"]:6.1f}% | {row["avg_detections"]:5.2f} avg')

print(f'\nVariability by Evidence Position:')
by_pos = success.groupby('evidence_pos')['pct_with_detections'].agg(['mean', 'min', 'max'])
print(by_pos)

print(f'\nVariability by Density:')
by_density = success.groupby('density')['pct_with_detections'].agg(['mean', 'min', 'max'])
print(by_density)

print(f'\nVariability by Interference:')
by_interference = success.groupby('interference')['pct_with_detections'].agg(['mean', 'min', 'max'])
print(by_interference)
