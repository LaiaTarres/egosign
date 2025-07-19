'''
The idea with this script is to create the tsv with all the information. 
This we should have it for TD and SLT, it should be the same file with all the info.
What we decided is that the poses are at video level, so for the TD is perfect, but for the SLT we will need to cut them into the sentences


# We want to generate a tsv file that has the following columns:
id	id_vid	signs_file	signs_offset	signs_length	signs_type	signs_lang	translation	translation_lang	glosses	topic	signer_id

We need to get the information from the How2Sign tsv, as well as the sentence_alignment files that we have.
path_to/Egosign/egosign/alignment_files/sentence_alignment_val.tsv
path_to/Egosign/egosign/alignment_files/sentence_alignment_test.tsv

This is what we have in this frames
VIDEO_ID	TEXT	START_FRAME	END_FRAME
But the videos are cut, so we should have that the start frame should be of the first one, should be 0
And the last frame we should check how many frames are there.

Also, for the sentences, we should take them from the original .tsv file: 
path_to_egosign/mediapipe_keypoints/how2sign_val_proves_filtered.tsv
Be careful because for some of them, the text has some weird characters... let's put the ones from How2Sign

id	id_vid	signs_file	signs_offset	signs_length	signs_type	signs_lang	translation	translation_lang	glosses	topic	signer_id
-d5dN54tH2E_0-1-rgb_front	-d5dN54tH2E-1	path_to/How2Sign/video_level/val/rgb_front/features/mediapipe/-d5dN54tH2E-1-rgb_front.pose	324	165	mediapipe_keypoints	asl	We're going to work on a arm drill that will help you have graceful hand movements in front of you.	en		8	1

'''

import pandas as pd
import difflib
import os
from pose_format import Pose
from tqdm import tqdm

def normalize_text(text):
    """Normalize text to handle encoding issues and make matching more robust."""
    if not isinstance(text, str):
        print(f"Warning: text is not a string: {text}")
        return ""
    return text.strip().lower()

def main():
    partition = "test"
    # File paths
    egosign_file = f"path_to/Egosign/egosign/alignment_files/sentence_alignment_{partition}.tsv"
    how2sign_file = f"path_to/TopicDetection/mediapipe_keypoints/how2sign_{partition}_proves.tsv"
    output_file = f"path_to/How2Sign/TopicDetection/mediapipe_keypoints/egosign_{partition}_proves_filtered.tsv"

    # Load data
    egosign_df = pd.read_csv(egosign_file, sep="\t")
    how2sign_df = pd.read_csv(how2sign_file, sep="\t")


    # Normalize text columns for robust matching
    egosign_df['normalized_TEXT'] = egosign_df['TEXT'].apply(normalize_text)
    how2sign_df['normalized_translation'] = how2sign_df['translation'].apply(normalize_text)
    
    # Prepare for merging
    merged_data = []

    # Let's have the new START and END frames, because we cut the videos:
    video_stats = egosign_df.groupby("VIDEO_ID").agg({"START_FRAME": "min", "END_FRAME": "max"}).reset_index()
    
    # Group by video_id
    grouped_egosign = egosign_df.groupby("VIDEO_ID")

    for video_id, ego_group in tqdm(grouped_egosign):
        ego_video_id = video_id[:11]  # First 11 characters, this is the video id without the signer id.
            
        ## Things to do for the full video:
        # Check that the video exists
        video_path_id = f"path_to/Egosign/egosign/video_level/rgb/{partition}/{video_id}/{video_id}-rgb_front.mp4"
        # Check the length of the "cut video", in number of frames.
        if not os.path.exists(video_path_id):
            print(f"Video {video_path_id} does not exist")
            continue
        
        # Check that the pose file exists
        pose_path_id = f"path_to/Egosign/egosign/video_level/features/mediapipe/front/{partition}/{video_id}.pose"
        if not os.path.exists(video_path_id):
            print(f"Pose {pose_path_id} does not exist")
            continue
        
        #Load the pose to check how many frames are there
        with open(pose_path_id, "rb") as f:
            pose = Pose.read(f.read())
        len_pose = pose.body.data.shape[0]
        
        ## For each of the sentences        
        for _, sentence in ego_group.iterrows():
            if ego_video_id == "g1uA0f9I0Sg":
                new_START_FRAME = sentence['START_FRAME']
                new_END_FRAME = sentence['END_FRAME']
                #IN this case, we don't cut it like this because the pose starts at the 0 and it has not been cut. There was a nan that we have not solved...
            else:
                new_START_FRAME = sentence['START_FRAME'] - int(video_stats[video_stats['VIDEO_ID']==video_id]['START_FRAME'])
                new_END_FRAME = sentence['END_FRAME'] - int(video_stats[video_stats['VIDEO_ID']==video_id]['START_FRAME'])
            
            length_sentence = new_END_FRAME - new_START_FRAME
            assert length_sentence == sentence['END_FRAME'] - sentence['START_FRAME']
            
            if new_END_FRAME > len_pose:
                if new_START_FRAME > len_pose:
                    print(f"Warning: The end of the sentence is longer than the video {video_id}. len_pose: {len_pose} and new_START_FRAME: {new_START_FRAME}")
                    continue
                else:
                    new_end_FRAME = len_pose
                    length_sentence = new_end_FRAME - new_START_FRAME
            
            #We need the pose file too: pose_path_id
            
            ego_text = sentence['normalized_TEXT']

            # Find matching rows in how2sign
            matches = how2sign_df[
                (how2sign_df['id_vid'].str[:11] == video_id[:11]) &
                (how2sign_df['normalized_translation'].apply(lambda x: difflib.SequenceMatcher(None, x, ego_text).ratio() > 0.8))
            ]
            
            if matches.empty:
                print(f"No matches found for video_id: {video_id} and text: {ego_text}")
                #Here we should add them to the file, but with the original text, and figure out the rest of the values from another sentence
                matches_new = how2sign_df[(how2sign_df['id_vid'].str[:11] == video_id[:11])]
                matches_new = matches_new.head(1) #This is still a dataframe, so we need to do the iloc to extract the value.
                
                signer_id = int(video_id[12:])
                new_id = how2_row['id'].replace(f"-{how2_row['signer_id']}-", f"-{signer_id}-")
                
                merged_row = {
                    'id': new_id, 
                    'id_vid': video_id,
                    'signs_file': pose_path_id,
                    'signs_offset': new_START_FRAME,
                    'signs_length': new_END_FRAME - new_START_FRAME,
                    'signs_type': 'mediapipe_keypoints',
                    'signs_lang': matches_new['signs_lang'].iloc[0],
                    'translation': sentence['TEXT'],
                    'translation_lang': matches_new['translation_lang'].iloc[0],
                    'glosses': matches_new['glosses'].iloc[0],
                    'topic': matches_new['topic'].iloc[0],
                    'signer_id': signer_id
                }
                merged_data.append(merged_row)
                continue
            
            if len(matches) > 1:
                #print(f"video_id: {video_id} has multiple matches: \n{matches}, we keep the first one")
                #This is the case when there are 2 signers that signed the same sentence. What we do is keep with the first one
                #matches = matches.head(1)
                # for example, G0PNAsonBGk_18, this should be doing weird things.
                
                signers_id = matches['signer_id'].unique()
                if len(signers_id) > 1:
                    matches = matches.head(1) # We keep the first one
                else:
                    #Check if there is a match that is better than the other
                    ratios = [difflib.SequenceMatcher(None, match['normalized_translation'], ego_text).ratio() for i, match in matches.iterrows()]
                    #If there si only 1 ratio == 1.0, then we keep that one, otherwise continue with the code
                    if ratios.count(1.0) == 1:
                        matches = matches.iloc[[ratios.index(1.0)]]
                    else:
                        #We need to figure out which match to return
                        for i, how2_row in matches.iterrows():
                            sentence_id = how2_row['id'].replace("-"+str(how2_row['signer_id'])+"-rgb_front", "")
                            #merged data is a list of dicts
                            if not any(sentence_id in d['id'] for d in merged_data):
                                matches = matches.iloc[matches.index==i]
                                break
                            else:
                                #if it's already in merged_data, continue checking next match
                                continue
                
            
            for _, how2_row in matches.iterrows():
                #Here we should do the merging
                #print(f"how2_row.to_dict():\n{how2_row.to_dict()}")
                
                signer_id = int(video_id[12:])
                new_id = how2_row['id'].replace(f"-{how2_row['signer_id']}-", f"-{signer_id}-")
                
                #if how2_row['signer_id'] != int(video_id[12:]):
                #    signer_id = int(video_id[12:])
                #    #print(f"Warning: Signer ID mismatch: {how2_row['signer_id']} != {signer_id}")
                #    new_id = how2_row['id'].replace(f"-{how2_row['signer_id']}-", f"-{signer_id}-")
                #    #print(f"New id: {new_id}")
                #else:
                #    signer_id = how2_row['signer_id']
                #    new_id = how2_row['id']
                    
                #print(f"to double check, here are the two sentences: \n{ego_text}\n{how2_row['translation']}")
                #Check how alike they are
                semblance_text = difflib.SequenceMatcher(None, how2_row['normalized_translation'], ego_text).ratio() #1 means totally equal.
                if semblance_text < 0.95:
                    print(f"Semblance between both texts: {semblance_text}")
                
                merged_row = {
                    'id': new_id, 
                    'id_vid': video_id,
                    'signs_file': pose_path_id,
                    'signs_offset': new_START_FRAME,
                    'signs_length': new_END_FRAME - new_START_FRAME,
                    'signs_type': 'mediapipe_keypoints',
                    'signs_lang': how2_row['signs_lang'],
                    'translation': how2_row['translation'], 
                    'translation_lang': how2_row['translation_lang'],
                    'glosses': how2_row['glosses'],
                    'topic': how2_row['topic'],
                    'signer_id': signer_id
                }
                merged_data.append(merged_row)
        
    # Create a DataFrame from merged data
    merged_df = pd.DataFrame(merged_data)

    # Save to a new TSV file
    merged_df.to_csv(output_file, sep="\t", index=False)
    print(f"Merged data saved to {output_file}")

if __name__ == "__main__":
    main()


'''
salloc --time=2:00:00 --mem 10G --nodes=1

module load Anaconda3/2023.09-0
eval "$(conda shell.bash hook)"
conda activate SLTopicDetection_2

cd slt_how2sign_wicv2023/examples/SL_topic_detection/

python prep_egosign_pose.py

'''




'''
For the validation partition:

  2%|██▊                                                                                                                                                                           | 2/122 [00:39<42:15, 21.13s/it]
Warning: The end of the sentence VIDEO_ID                           00dWJ4YRRSI-12
TEXT           you simply get him up to the boat.
START_FRAME                                  3363
END_FRAME                                    3489
Name: 788, dtype: object is longer than the video 00dWJ4YRRSI-12. Check what might be causing this.
  4%|███████▏                                                                                                                                                                      | 5/122 [01:04<21:41, 11.13s/it]
  Warning: The end of the sentence VIDEO_ID                                          0zvsqf23tmw-8
TEXT           and i'm not going to do way solid onto the wall.
START_FRAME                                                3664
END_FRAME                                                  3807
Name: 591, dtype: object is longer than the video 0zvsqf23tmw-8. Check what might be causing this.
Warning: The end of the sentence VIDEO_ID                                      0zvsqf23tmw-8
TEXT           you don't need a lot of paint on your brush.
START_FRAME                                            3681
END_FRAME                                              3769
Name: 600, dtype: object is longer than the video 0zvsqf23tmw-8. Check what might be causing this.
  5%|████████▌                                                                                                                                                                     | 6/122 [01:17<22:42, 11.75s/it]
  Warning: The end of the sentence VIDEO_ID                                          1-xK5UtDSmE-12
TEXT           think about it, if you have a heart attack, if...
START_FRAME                                                 6337
END_FRAME                                                   6687
Name: 541, dtype: object is longer than the video 1-xK5UtDSmE-12. Check what might be causing this.
  6%|█████████▉                                                                                                                                                                    | 7/122 [01:32<24:15, 12.65s/it]
  Warning: The end of the sentence VIDEO_ID                   11JT4jRNI-o-12
TEXT           again, so many variations.
START_FRAME                          2811
END_FRAME                            2862
Name: 88, dtype: object is longer than the video 11JT4jRNI-o-12. Check what might be causing this.
No matches found for video_id: 11JT4jRNI-o-12 and text: but it's definitely a great exercise, and i highly recommend it.
  9%|███████████████▌                                                                                                                                                             | 11/122 [02:04<17:31,  9.47s/it]
  Warning: The end of the sentence VIDEO_ID                                          1aJwX9nRlmk-12
TEXT           if you can keep it going for a minute, that's ...
START_FRAME                                                 5314
END_FRAME                                                   5469
Name: 1269, dtype: object is longer than the video 1aJwX9nRlmk-12. Check what might be causing this.
Warning: The end of the sentence VIDEO_ID                                          1aJwX9nRlmk-12
TEXT           let the breath be the emphasis for every movem...
START_FRAME                                                 5478
END_FRAME                                                   5635
Name: 1270, dtype: object is longer than the video 1aJwX9nRlmk-12. Check what might be causing this.
 15%|█████████████████████████▌                                                                                                                                                   | 18/122 [03:06<15:36,  9.01s/it]
 Warning: The end of the sentence VIDEO_ID                                          33jxeIIbBnM-12
TEXT           my name is dave andrews and i just showed you ...
START_FRAME                                                 6333
END_FRAME                                                   6715
Name: 254, dtype: object is longer than the video 33jxeIIbBnM-12. Check what might be causing this.
 17%|█████████████████████████████▊                                                                                                                                               | 21/122 [03:52<21:22, 12.69s/it]
 Warning: The end of the sentence VIDEO_ID              3HCjTYIijec-14
TEXT           it's a vicious cycle.
START_FRAME                     3984
END_FRAME                       4066
Name: 773, dtype: object is longer than the video 3HCjTYIijec-14. Check what might be causing this.
 18%|███████████████████████████████▏                                                                                                                                             | 22/122 [04:04<20:46, 12.46s/it]
 Warning: The end of the sentence VIDEO_ID         3TrMyzNWGpY-12
TEXT           i'm john graden.
START_FRAME                5342
END_FRAME                  5392
Name: 1493, dtype: object is longer than the video 3TrMyzNWGpY-12. Check what might be causing this.
Warning: The end of the sentence VIDEO_ID           3TrMyzNWGpY-12
TEXT           i hope that helps.
START_FRAME                  5396
END_FRAME                    5446
Name: 1494, dtype: object is longer than the video 3TrMyzNWGpY-12. Check what might be causing this.
 21%|████████████████████████████████████▊                                                                                                                                        | 26/122 [04:32<12:43,  7.95s/it]No matches found for video_id: 46Cwjrd4ua4-14 and text: now the next environmental factors that you want to look at are the date, and the year.
Semblance between both texts: 0.875
Warning: The end of the sentence VIDEO_ID                                          46Cwjrd4ua4-14
TEXT           take into effect all of those environmental fa...
START_FRAME                                                 4597
END_FRAME                                                   4743
Name: 873, dtype: object is longer than the video 46Cwjrd4ua4-14. Check what might be causing this.
 24%|█████████████████████████████████████████                                                                                                                                    | 29/122 [05:08<15:19,  9.88s/it]
 Warning: The end of the sentence VIDEO_ID             4CSSlWonj3E-14
TEXT           richard buccola: hi.
START_FRAME                    3049
END_FRAME                      3136
Name: 50, dtype: object is longer than the video 4CSSlWonj3E-14. Check what might be causing this.
 28%|████████████████████████████████████████████████▏                                                                                                                            | 34/122 [05:55<14:30,  9.89s/it]
 Warning: The end of the sentence VIDEO_ID       4XlVMRXLydg-14
TEXT                     any-
START_FRAME              4593
END_FRAME                4610
Name: 899, dtype: object is longer than the video 4XlVMRXLydg-14. Check what might be causing this.
Warning: The end of the sentence VIDEO_ID       4XlVMRXLydg-14
TEXT                      ok?
START_FRAME              4589
END_FRAME                4985
Name: 904, dtype: object is longer than the video 4XlVMRXLydg-14. Check what might be causing this.
Warning: The end of the sentence VIDEO_ID       4XlVMRXLydg-14
TEXT                      ok?
START_FRAME              4601
END_FRAME                4614
Name: 912, dtype: object is longer than the video 4XlVMRXLydg-14. Check what might be causing this.
 31%|█████████████████████████████████████████████████████▉                                                                                                                       | 38/122 [06:29<11:31,  8.23s/it]
 Warning: The end of the sentence VIDEO_ID                                          BMXB5nth8hA-14
TEXT           first of all, it shows the seller that you are...
START_FRAME                                                 3197
END_FRAME                                                   3632
Name: 561, dtype: object is longer than the video BMXB5nth8hA-14. Check what might be causing this.
Warning: The end of the sentence VIDEO_ID                                BMXB5nth8hA-14
TEXT           a prequalification is really important.
START_FRAME                                       3637
END_FRAME                                         3796
Name: 562, dtype: object is longer than the video BMXB5nth8hA-14. Check what might be causing this.
 39%|████████████████████████████████████████████████████████████████████                                                                                                         | 48/122 [07:47<10:09,  8.24s/it]
 Warning: The end of the sentence VIDEO_ID                               CzkLI34HFIg-14
TEXT           and i'll show you how to do that next.
START_FRAME                                      5142
END_FRAME                                        5232
Name: 203, dtype: object is longer than the video CzkLI34HFIg-14. Check what might be causing this.
 44%|████████████████████████████████████████████████████████████████████████████▌                                                                                                | 54/122 [08:51<12:29, 11.02s/it]
 Warning: The end of the sentence VIDEO_ID                                          DfnHNkTE7mE-12
TEXT           so yeah, do that, go rent it because residuals...
START_FRAME                                                 5378
END_FRAME                                                   5598
Name: 456, dtype: object is longer than the video DfnHNkTE7mE-12. Check what might be causing this.
Warning: The end of the sentence VIDEO_ID                                       DfnHNkTE7mE-12
TEXT           so, go rent the movie and check out the scene.
START_FRAME                                              5598
END_FRAME                                                5690
Name: 457, dtype: object is longer than the video DfnHNkTE7mE-12. Check what might be causing this.
 48%|██████████████████████████████████████████████████████████████████████████████████▏                                                                                          | 58/122 [09:39<12:34, 11.79s/it]
 Warning: The end of the sentence VIDEO_ID                                      Eh2AVkAQsxI-12
TEXT           and that is how to lighten your hair at home.
START_FRAME                                             5666
END_FRAME                                               5811
Name: 147, dtype: object is longer than the video Eh2AVkAQsxI-12. Check what might be causing this.

No matches found for video_id: f5EGPzGSCJs-15 and text: today we're going to talk about how to diagnose an ailment of your fish.
No matches found for video_id: f5EGPzGSCJs-15 and text: aquarium salt raises the electrolytes in the tank and it also helps reduce stress.
No matches found for video_id: f5EGPzGSCJs-15 and text: my name is michael and today we're talking about how to diagnose an ailment.

No matches found for video_id: f5EGPzGSCJs-16 and text: today we're going to talk about how to diagnose an ailment of your fish.
No matches found for video_id: f5EGPzGSCJs-16 and text: aquarium salt raises the electrolytes in the tank and it also helps reduce stress.
No matches found for video_id: f5EGPzGSCJs-16 and text: my name is michael and today we're talking about how to diagnose an ailment.

No matches found for video_id: fE6xxSbjVV8-15 and text: that's it.
No matches found for video_id: fE6xxSbjVV8-16 and text: that's it.




For the test partition:

  5%|████████▏                                                                                                                                                                     | 7/149 [01:18<26:32, 11.21s/it]Semblance between both texts: 0.9411764705882353
 17%|█████████████████████████████                                                                                                                                                | 25/149 [04:57<16:27,  7.96s/it]Warning: The end of the sentence VIDEO_ID         FzoUVr98JmQ-13
TEXT           allen diwan: hi.
START_FRAME                2484
END_FRAME                  2488
Name: 1053, dtype: object is longer than the video FzoUVr98JmQ-13. Check what might be causing this.
 19%|████████████████████████████████▌                                                                                                                                            | 28/149 [05:26<18:57,  9.40s/it]Warning: The end of the sentence VIDEO_ID                                          G06Irzcwxiw-15
TEXT           so if you can enjoy the moment that you're in ...
START_FRAME                                                 6758
END_FRAME                                                   6940
Name: 1614, dtype: object is longer than the video G06Irzcwxiw-15. Check what might be causing this.
Warning: The end of the sentence VIDEO_ID                                 G06Irzcwxiw-15
TEXT           you're having a good time along the way.
START_FRAME                                        6940
END_FRAME                                          6992
Name: 1615, dtype: object is longer than the video G06Irzcwxiw-15. Check what might be causing this.
Warning: The end of the sentence VIDEO_ID                                          G06Irzcwxiw-15
TEXT           and every moment's not good but i think trying...
START_FRAME                                                 7021
END_FRAME                                                   7302
Name: 1616, dtype: object is longer than the video G06Irzcwxiw-15. Check what might be causing this.
 23%|████████████████████████████████████████▋                                                                                                                                    | 35/149 [07:31<26:57, 14.19s/it]Semblance between both texts: 0.9090909090909091
 24%|█████████████████████████████████████████▊                                                                                                                                   | 36/149 [07:50<29:36, 15.72s/it]Warning: The end of the sentence VIDEO_ID                            G1GUMky8kWc-8
TEXT           so, it's going to sound like this.
START_FRAME                                  3094
END_FRAME                                    3241
Name: 1051, dtype: object is longer than the video G1GUMky8kWc-8. Check what might be causing this.
 26%|████████████████████████████████████████████                                                                                                                                 | 38/149 [08:10<24:29, 13.24s/it]Warning: The end of the sentence VIDEO_ID                            G1QiXuldOxM-8
TEXT           make sure you're wedging properly.
START_FRAME                                  3164
END_FRAME                                    3278
Name: 82, dtype: object is longer than the video G1QiXuldOxM-8. Check what might be causing this.
 26%|█████████████████████████████████████████████▎                                                                                                                               | 39/149 [08:21<23:03, 12.58s/it]Warning: The end of the sentence VIDEO_ID       G1hb5HugzVk-8
TEXT                     hi!
START_FRAME             3015
END_FRAME               3051
Name: 1552, dtype: object is longer than the video G1hb5HugzVk-8. Check what might be causing this.
 32%|██████████████████████████████████████████████████████▌                                                                                                                      | 47/149 [09:44<17:32, 10.32s/it]Semblance between both texts: 0.8269230769230769
 33%|████████████████████████████████████████████████████████▉                                                                                                                    | 49/149 [10:02<15:05,  9.05s/it]Semblance between both texts: 0.9073569482288828
 35%|████████████████████████████████████████████████████████████▍                                                                                                                | 52/149 [10:35<19:12, 11.88s/it]Warning: The end of the sentence VIDEO_ID                     G2dND014Ps4-14
TEXT           so that is pretty important.
START_FRAME                            3450
END_FRAME                              3465
Name: 534, dtype: object is longer than the video G2dND014Ps4-14. Check what might be causing this.
 40%|████████████████████████████████████████████████████████████████████▌                                                                                                        | 59/149 [12:11<17:21, 11.58s/it]Semblance between both texts: 0.891566265060241
 42%|███████████████████████████████████████████████████████████████████████▉                                                                                                     | 62/149 [12:45<16:19, 11.26s/it]Warning: The end of the sentence VIDEO_ID                             G3EYpadwqck-14
TEXT           that's why they have it in the foil.
START_FRAME                                    3240
END_FRAME                                      3300
Name: 1534, dtype: object is longer than the video G3EYpadwqck-14. Check what might be causing this.
 42%|█████████████████████████████████████████████████████████████████████████▏                                                                                                   | 63/149 [12:59<17:21, 12.11s/it]No matches found for video_id: G3EfBFwsOpE-14 and text: reverse layups take a lot of practice, timing, and coordination.
 48%|██████████████████████████████████████████████████████████████████████████████████▍                                                                                          | 71/149 [14:34<13:18, 10.24s/it]Warning: The end of the sentence VIDEO_ID                                          G3bMqicS4bQ-14
TEXT           so that--make sure that your modifications are...
START_FRAME                                                 5130
END_FRAME                                                   5247
Name: 249, dtype: object is longer than the video G3bMqicS4bQ-14. Check what might be causing this.
 49%|████████████████████████████████████████████████████████████████████████████████████▊                                                                                        | 73/149 [15:05<16:02, 12.66s/it]Warning: The end of the sentence VIDEO_ID                    G3g0-BeFN3c-14
TEXT           that's where her makeup is.
START_FRAME                           3480
END_FRAME                             3540
Name: 1403, dtype: object is longer than the video G3g0-BeFN3c-14. Check what might be causing this.
Warning: The end of the sentence VIDEO_ID                         G3g0-BeFN3c-14
TEXT           which is usually a neutral tone.
START_FRAME                                3510
END_FRAME                                  3570
Name: 1411, dtype: object is longer than the video G3g0-BeFN3c-14. Check what might be causing this.
Warning: The end of the sentence VIDEO_ID                         G3g0-BeFN3c-14
TEXT           and take care of them every day.
START_FRAME                                3392
END_FRAME                                  3456
Name: 1422, dtype: object is longer than the video G3g0-BeFN3c-14. Check what might be causing this.
 54%|██████████████████████████████████████████████████████████████████████████████████████████████                                                                               | 81/149 [17:05<16:02, 14.16s/it]Warning: The end of the sentence VIDEO_ID       _fZbAxSSbX4-13
TEXT                    paul?
START_FRAME              6942
END_FRAME                7005
Name: 323, dtype: object is longer than the video _fZbAxSSbX4-13. Check what might be causing this.
Warning: The end of the sentence VIDEO_ID       _fZbAxSSbX4-13
TEXT                     yes.
START_FRAME              6942
END_FRAME                6991
Name: 324, dtype: object is longer than the video _fZbAxSSbX4-13. Check what might be causing this.
 60%|████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                                    | 90/149 [18:27<08:47,  8.94s/it]Semblance between both texts: 0.9375
 72%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                               | 108/149 [21:48<07:36, 11.14s/it]No matches found for video_id: g0iNy-yPisM-8 and text: """you""."
 74%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                             | 110/149 [22:16<07:49, 12.05s/it]Warning: The end of the sentence VIDEO_ID       g0t4Wz5qsT8-12
TEXT                 massey!.
START_FRAME              4648
END_FRAME                4673
Name: 1296, dtype: object is longer than the video g0t4Wz5qsT8-12. Check what might be causing this.
 75%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                          | 112/149 [22:28<05:17,  8.59s/it]No matches found for video_id: g0yUlOaqL6k-15 and text: you can just add a little bit of that to your water and soak those plants in there.
No matches found for video_id: g0yUlOaqL6k-15 and text: use the wipe out or terrarium cleaner on any type of housing that you have in there.
No matches found for video_id: g0yUlOaqL6k-15 and text: it will work well for that and the food and water dishes i would just use a anti bacterial dish soap.
No matches found for video_id: g0yUlOaqL6k-15 and text: to make sure that we don't get any animals in digesting them cleaning supplies.
No matches found for video_id: g0yUlOaqL6k-15 and text: this type of bedding the sand requires a different type of cleaner.
No matches found for video_id: g0yUlOaqL6k-15 and text: to clean this out you would use something such as this it's a sand shovel.
No matches found for video_id: g0yUlOaqL6k-15 and text: what it will do is shift all the foreign objects right out of that sand and you can throw it out of the trash.
No matches found for video_id: g0yUlOaqL6k-15 and text: this is also very simple to clean it's not a long process and you only need to replace this bedding every six months as well, this type of bedding is very convenient for the long life span.
 77%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                        | 114/149 [22:40<04:04,  6.99s/it]Warning: The end of the sentence VIDEO_ID                             g1HXoDkax5A-16
TEXT           the nasal cavity is fairly flexible.
START_FRAME                                    3585
END_FRAME                                      3651
Name: 216, dtype: object is longer than the video g1HXoDkax5A-16. Check what might be causing this.
Warning: The end of the sentence VIDEO_ID                g1HXoDkax5A-16
TEXT           it's extremely painful.
START_FRAME                       3600
END_FRAME                         3750
Name: 224, dtype: object is longer than the video g1HXoDkax5A-16. Check what might be causing this.
Warning: The end of the sentence VIDEO_ID       g1HXoDkax5A-16
TEXT                gruesome?
START_FRAME              3600
END_FRAME                3750
Name: 227, dtype: object is longer than the video g1HXoDkax5A-16. Check what might be causing this.
Warning: The end of the sentence VIDEO_ID       g1HXoDkax5A-16
TEXT                     yes.
START_FRAME              3600
END_FRAME                3750
Name: 228, dtype: object is longer than the video g1HXoDkax5A-16. Check what might be causing this.
 77%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                       | 115/149 [23:01<06:19, 11.16s/it]Warning: The end of the sentence VIDEO_ID       g1HvmBOR7Y4-16
TEXT                 come on.
START_FRAME              3389
END_FRAME                3433
Name: 1825, dtype: object is longer than the video g1HvmBOR7Y4-16. Check what might be causing this.
Warning: The end of the sentence VIDEO_ID       g1HvmBOR7Y4-16
TEXT                good boy.
START_FRAME              3385
END_FRAME                3441
Name: 1826, dtype: object is longer than the video g1HvmBOR7Y4-16. Check what might be causing this.
Warning: The end of the sentence VIDEO_ID          g1HvmBOR7Y4-16
TEXT           keeps him coming.
START_FRAME                 3381
END_FRAME                   3442
Name: 1828, dtype: object is longer than the video g1HvmBOR7Y4-16. Check what might be causing this.
 78%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                      | 116/149 [23:16<06:49, 12.40s/it]Warning: The end of the sentence VIDEO_ID                                          g1ccEYTMGGY-15
TEXT           whether it be your height, whether it be your ...
START_FRAME                                                 3434
END_FRAME                                                   3643
Name: 1498, dtype: object is longer than the video g1ccEYTMGGY-15. Check what might be causing this.
Warning: The end of the sentence VIDEO_ID                                          g1ccEYTMGGY-15
TEXT           so you need to find a jean that's perfect for ...
START_FRAME                                                 3441
END_FRAME                                                   3565
Name: 1499, dtype: object is longer than the video g1ccEYTMGGY-15. Check what might be causing this.
Warning: The end of the sentence VIDEO_ID                                          g1ccEYTMGGY-15
TEXT           now my suggestion is to do some research on th...
START_FRAME                                                 3457
END_FRAME                                                   3621
Name: 1500, dtype: object is longer than the video g1ccEYTMGGY-15. Check what might be causing this.
Warning: The end of the sentence VIDEO_ID                                          g1ccEYTMGGY-15
TEXT           and what you can do is mix and match different...
START_FRAME                                                 3444
END_FRAME                                                   3522
Name: 1501, dtype: object is longer than the video g1ccEYTMGGY-15. Check what might be causing this.
Warning: The end of the sentence VIDEO_ID                               g1ccEYTMGGY-15
TEXT           write them down, then go to the store.
START_FRAME                                      3442
END_FRAME                                        3537
Name: 1502, dtype: object is longer than the video g1ccEYTMGGY-15. Check what might be causing this.
Warning: The end of the sentence VIDEO_ID                                          g1ccEYTMGGY-15
TEXT           have somebody help you pick out the jeans you ...
START_FRAME                                                 3438
END_FRAME                                                   3680
Name: 1503, dtype: object is longer than the video g1ccEYTMGGY-15. Check what might be causing this.
Warning: The end of the sentence VIDEO_ID                                          g1ccEYTMGGY-15
TEXT           my other piece of advice is bring a friend wit...
START_FRAME                                                 3446
END_FRAME                                                   3521
Name: 1504, dtype: object is longer than the video g1ccEYTMGGY-15. Check what might be causing this.
Warning: The end of the sentence VIDEO_ID                                          g1ccEYTMGGY-15
TEXT           because what you may think looks good on you, ...
START_FRAME                                                 3440
END_FRAME                                                   3557
Name: 1505, dtype: object is longer than the video g1ccEYTMGGY-15. Check what might be causing this.
Warning: The end of the sentence VIDEO_ID                                          g1ccEYTMGGY-15
TEXT           so some of my favorite brands that we are goin...
START_FRAME                                                 3457
END_FRAME                                                   3558
Name: 1506, dtype: object is longer than the video g1ccEYTMGGY-15. Check what might be causing this.
Warning: The end of the sentence VIDEO_ID                                          g1ccEYTMGGY-15
TEXT           the most important thing to pay attention to i...
START_FRAME                                                 3441
END_FRAME                                                   3627
Name: 1507, dtype: object is longer than the video g1ccEYTMGGY-15. Check what might be causing this.
Warning: The end of the sentence VIDEO_ID                        g1ccEYTMGGY-15
TEXT           how the jeans hit each of them.
START_FRAME                               3451
END_FRAME                                 3512
Name: 1508, dtype: object is longer than the video g1ccEYTMGGY-15. Check what might be causing this.
Warning: The end of the sentence VIDEO_ID                                          g1ccEYTMGGY-15
TEXT           some jeans are too tight, some jeans are not t...
START_FRAME                                                 3430
END_FRAME                                                   3527
Name: 1509, dtype: object is longer than the video g1ccEYTMGGY-15. Check what might be causing this.
Warning: The end of the sentence VIDEO_ID                                          g1ccEYTMGGY-15
TEXT           once you find that brand you can buy five to s...
START_FRAME                                                 3435
END_FRAME                                                   3568
Name: 1510, dtype: object is longer than the video g1ccEYTMGGY-15. Check what might be causing this.
Warning: The end of the sentence VIDEO_ID                                          g1ccEYTMGGY-15
TEXT           we're going to show you the different styles a...
START_FRAME                                                 3437
END_FRAME                                                   3545
Name: 1511, dtype: object is longer than the video g1ccEYTMGGY-15. Check what might be causing this.
Warning: The end of the sentence VIDEO_ID                                          g1ccEYTMGGY-15
TEXT           dark or light, different washes, different hol...
START_FRAME                                                 3444
END_FRAME                                                   3611
Name: 1512, dtype: object is longer than the video g1ccEYTMGGY-15. Check what might be causing this.
Warning: The end of the sentence VIDEO_ID                                          g1ccEYTMGGY-15
TEXT           so come along with me as we try on several dif...
START_FRAME                                                 3436
END_FRAME                                                   3626
Name: 1513, dtype: object is longer than the video g1ccEYTMGGY-15. Check what might be causing this.
 81%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                 | 120/149 [23:59<05:27, 11.29s/it]Warning: The end of the sentence VIDEO_ID       g1xdqxCZxTg-12
TEXT                  aliens?
START_FRAME              2705
END_FRAME                2720
Name: 784, dtype: object is longer than the video g1xdqxCZxTg-12. Check what might be causing this.
Warning: The end of the sentence VIDEO_ID       g1xdqxCZxTg-12
TEXT                    okay.
START_FRAME              2702
END_FRAME                2717
Name: 791, dtype: object is longer than the video g1xdqxCZxTg-12. Check what might be causing this.
 81%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                | 121/149 [24:12<05:35, 11.98s/it]Warning: The end of the sentence VIDEO_ID       g1z6HOJ0yRw-12
TEXT                    okay?
START_FRAME              2460
END_FRAME                2475
Name: 899, dtype: object is longer than the video g1z6HOJ0yRw-12. Check what might be causing this.
 82%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                               | 122/149 [24:21<04:54, 10.90s/it]Semblance between both texts: 0.8376068376068376
 83%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                            | 124/149 [24:49<05:03, 12.15s/it]Warning: The end of the sentence VIDEO_ID                                  g2NA_eBUcH8-15
TEXT           you're not playing with any other player.
START_FRAME                                         5209
END_FRAME                                           5327
Name: 406, dtype: object is longer than the video g2NA_eBUcH8-15. Check what might be causing this.
Warning: The end of the sentence VIDEO_ID       g2NA_eBUcH8-15
TEXT               good luck.
START_FRAME              5214
END_FRAME                5269
Name: 411, dtype: object is longer than the video g2NA_eBUcH8-15. Check what might be causing this.
 84%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                           | 125/149 [25:20<07:08, 17.85s/it]Warning: The end of the sentence VIDEO_ID                                          g2QdwYqm8pg-16
TEXT           so once again, now i'm going to do another clo...
START_FRAME                                                 2940
END_FRAME                                                   3055
Name: 87, dtype: object is longer than the video g2QdwYqm8pg-16. Check what might be causing this.
Warning: The end of the sentence VIDEO_ID                    g2QdwYqm8pg-16
TEXT           again i want to make faces.
START_FRAME                           2940
END_FRAME                             3009
Name: 89, dtype: object is longer than the video g2QdwYqm8pg-16. Check what might be causing this.
 85%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                         | 127/149 [25:48<05:55, 16.16s/it]Semblance between both texts: 0.835820895522388
Semblance between both texts: 0.8311688311688312
 87%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                      | 130/149 [26:30<04:54, 15.51s/it]Warning: The end of the sentence VIDEO_ID                   g2o-GFdGOJE-12
TEXT           here we go, you know what?
START_FRAME                          5190
END_FRAME                            5220
Name: 1778, dtype: object is longer than the video g2o-GFdGOJE-12. Check what might be causing this.
 89%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                   | 132/149 [27:04<04:28, 15.77s/it]Semblance between both texts: 0.9299363057324841
 93%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎            | 138/149 [28:09<02:06, 11.47s/it]Warning: The end of the sentence VIDEO_ID           g3PBeTb1TCw-8
TEXT           so one more time.
START_FRAME                 3802
END_FRAME                   3856
Name: 501, dtype: object is longer than the video g3PBeTb1TCw-8. Check what might be causing this.
Warning: The end of the sentence VIDEO_ID                                       g3PBeTb1TCw-8
TEXT           here, come up and hook it all the way around.
START_FRAME                                             3804
END_FRAME                                               3988
Name: 502, dtype: object is longer than the video g3PBeTb1TCw-8. Check what might be causing this.
 95%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊         | 141/149 [28:44<01:33, 11.69s/it]Warning: The end of the sentence VIDEO_ID                                 g3X3XE6M2_A-8
TEXT           now we move into ryote tori tenchinage.
START_FRAME                                       3570
END_FRAME                                         3600
Name: 1136, dtype: object is longer than the video g3X3XE6M2_A-8. Check what might be causing this.
Warning: The end of the sentence VIDEO_ID                    g3X3XE6M2_A-8
TEXT           this is where he's strong.
START_FRAME                          3570
END_FRAME                            3600
Name: 1139, dtype: object is longer than the video g3X3XE6M2_A-8. Check what might be causing this.
Warning: The end of the sentence VIDEO_ID                                           g3X3XE6M2_A-8
TEXT           if he pushes he moves me, if he pulls he moves...
START_FRAME                                                 3570
END_FRAME                                                   3600
Name: 1140, dtype: object is longer than the video g3X3XE6M2_A-8. Check what might be causing this.
Warning: The end of the sentence VIDEO_ID       g3X3XE6M2_A-8
TEXT               this way.
START_FRAME             3570
END_FRAME               3600
Name: 1145, dtype: object is longer than the video g3X3XE6M2_A-8. Check what might be causing this.
 99%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋  | 147/149 [30:07<00:27, 13.60s/it]

No matches found for video_id: g3sLd8JupoQ-8 and text: and then that gives you a line that you can cut on.

'''