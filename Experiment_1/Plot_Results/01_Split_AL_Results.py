import pandas as pd
import os

file_path = './02_Active Leanring/active_learning_metrics_detailed_PascalVoc.csv'
output_folder = './02_Active Leanring/AL_Results_Split_Reports_PascalVoc'

os.makedirs(output_folder, exist_ok=True)
data = pd.read_csv(file_path)

current_label_size = 500
batch_sizes = []
max_iteration = data['iteration'].max()
data['Labeled Data'] = data['labeled_sample_size'] / 100

output_file_path = os.path.join(output_folder, 'AL_Results_PascalVoc.csv')
data.to_csv(output_file_path, index=False)

unique_avg_types = data['average_type'].unique()
for avg_type in unique_avg_types:
    split_data = data[data['average_type'] == avg_type]
    file_name = os.path.join(output_folder, f"AL_Results_PascalVoc_{avg_type.replace(' ', '_')}.csv")
    split_data.to_csv(file_name, index=False)

print("Files have been processed and saved successfully:")
print(f"Full file: {output_file_path}")
for avg_type in unique_avg_types:
    print(f"Split file for '{avg_type}': {os.path.join(output_folder, f'AL_Results_PascalVoc_{avg_type.replace(' ', '_')}.csv')}")
