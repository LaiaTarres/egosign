'''
The idea with this script is to create the tsv with all the information. 
This we should have it for TD and SLT, it should be the same file with all the info.
What we decided is that the poses are at video level, so for the TD is perfect, but for the SLT we will need to cut them into the sentences
'''

import pandas as pd
from tqdm import tqdm
import os

# File paths
partition = "train"
input_file = f"path_to/How2Sign/TopicDetection/mediapipe_keypoints/how2sign_{partition}_proves.tsv"
output_file = f"path_to/How2Sign/TopicDetection/mediapipe_keypoints/how2sign_{partition}_proves_filtered.tsv"
video_path = f"path_to//How2Sign/video_level/{partition}/rgb_front/raw_videos/"

# Load the .tsv file into a DataFrame
data = pd.read_csv(input_file, sep='\t')

# Identify rows where the corresponding .mp4 file does not exist
missing_files = data[~data['id_vid'].apply(
    lambda vid: os.path.isfile(os.path.join(video_path, f"{vid}-rgb_front.mp4"))
)]

# Print the IDs of removed rows
if not missing_files.empty:
    print("IDs of removed rows:")
    print(missing_files['id_vid'].tolist())
    print(f"Removed {len(missing_files)} rows.")
else:
    print("No rows were removed. All files were found.")
    
# Filter rows where the corresponding .mp4 file exists
filtered_data = data[data['id_vid'].apply(
    lambda vid: os.path.isfile(os.path.join(video_path, f"{vid}-rgb_front.mp4"))
)]

# Save the filtered DataFrame to a new .tsv file
filtered_data.to_csv(output_file, sep='\t', index=False)

print(f"Filtered data has been saved to {output_file}")

'''
salloc --time=1:00:00 --mem 10G --nodes=1

module load Anaconda3/2023.09-0
eval "$(conda shell.bash hook)"
conda activate SLTopicDetection_2

cd slt_how2sign_wicv2023/examples/SL_topic_detection/
'''