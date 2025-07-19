'''
This file is to add the extra field of vid_length into the .tsv
'''

from pose_format import Pose
import pandas as pd
import csv
from tqdm import tqdm

def add_video_length_column(tsv_path, output_path):
    # Load TSV file efficiently
    df = pd.read_csv(tsv_path, sep='\t', quoting=csv.QUOTE_NONE)
    
    # Dictionary to store cached frame counts
    frame_cache = {}
    video_lengths = []
    
    for signs_file in tqdm(df['signs_file'], desc="Processing files"):
        if signs_file in frame_cache:
            video_lengths.append(frame_cache[signs_file])
        else:
            try:
                with open(signs_file, "rb") as f:
                    pose = Pose.read(f.read())
                    num_frames = pose.body.data.shape[0]
                    frame_cache[signs_file] = num_frames  # Cache it
                    video_lengths.append(num_frames)
            except Exception as e:
                print(f"Error processing {signs_file}: {e}")
                video_lengths.append(None)  # Handle errors gracefully
    
    # Add new column
    df['video_length'] = video_lengths
    
    # Save the updated file
    df.to_csv(output_path, sep='\t', index=False, quoting=csv.QUOTE_NONE)
    print(f"Processed file saved to {output_path}")


add_video_length_column("path_to_folder/how2sign_val_proves_filtered.tsv", "val_proves.tsv")
