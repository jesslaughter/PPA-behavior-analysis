"""
AutoPlacePrefAnalysis_600seconds.py
Authors: Jessica Slaughter, Yeji Kim, Polina Anikeeva  
Date: June 25, 2025  
Last Updated: August 3rd, 2025  
Description:
    This script processes the CSV output of DeepLabCut video analyses to quantify 
    mouse occupancy in predefined regions of interest during a place preference assay. 
    Specifically, it computes the time each subject spends in striped, middle, and 
    checkered zones based on tracked bodypart positions and an adjustable confidence threshold. 
    Videos must be at least 10 minutes long or 600 seconds in length.
    Inputs:
        - DeepLabCut CSV tracking files
        - Corresponding video files (.mp4) for timing and frame rate
    Outputs:
        - A summary CSV file containing time each subject spent in each ROI
Dependencies:
    - pandas
    - numpy
    - opencv-python (cv2)
    - datetime
    - os
"""

# Import libraries
import pandas as pd 
import numpy as np 
import cv2  
from datetime import datetime
import os

# USERS CHOICE
    # 'A' = determine mouse's location based off middle of it's body
    # 'B' = determine mouse's location based off front of it's body
    # 'C' = determine mouse's location based off feet // this one is the best because it is most closely related to the refernence point 
analysis_choice = 'C'

# Folder containing the videos to analysis
foldername = f"{os.getcwd()}/FinalData"

# Save name of all csv files in folder
file_array = []
for file in os.listdir(f'{foldername}'): 
    if file.endswith('.csv'):
        file_array.append(file)

# Sort files alphabetically/sequentially 
file_array = sorted(file_array)
# Define minimum confidence value of DLC predictions
p_cutoff = 0.6

# Initialize a DataFrame to store the results
    # Each row corresponds to one video file. 
    # The first column will hold file name
    # Next three columns will hold total time spent in striped, middle, and checkered area
    # The last column will hold the total duration of the experiment
results = pd.DataFrame({'File Name': [""]* len(file_array),
    'Seconds in Striped Area': [0.0] * len(file_array),
    'Seconds in Middle Area': [0.0] * len(file_array),
    'Seconds in Checkered Area': [0.0] * len(file_array),
    'Total Duration': [0.0] * len(file_array)
})


print("This might take a while. Please wait.")

# Index for file order
file_num = 0
for file in file_array:
    # Count files processed
    file_num += 1
    # Get experimental filename ID
    mouse_ID = file.split("DLC")[0]
    # Load the CSV file
    try:
        df = pd.read_csv(f'{foldername}/{file}', header=[1, 2])
    except FileNotFoundError:
        print(f"File not found: {file}")
        results.loc[file_num-1, 'File Name'] = mouse_ID
        results.loc[file_num-1, 'Seconds in Striped Area'] = 0.0
        results.loc[file_num-1, 'Seconds in Middle Area'] = 0.0
        results.loc[file_num-1, 'Seconds in Checkered Area'] = 0.0
        results.loc[file_num-1, 'Total Duration'] = 0.0
        continue


    # Define frames per second
    vidname = (f'{foldername}/{mouse_ID}.mp4')
    video =  cv2.VideoCapture(vidname)
    fps = video.get(cv2.CAP_PROP_FPS)
    if fps == 0.0:
        raise ValueError("FPS is zero — check your video file path.")

    # If the likelihood of boundary points are greater than p-stat cutoff value 
    check_left_filter = df[('check_left', 'likelihood')] >= p_cutoff
    check_right_filter = df[('check_right', 'likelihood')] >= p_cutoff

    stripe_left_filter = df[('stripe_left', 'likelihood')] >= p_cutoff
    stripe_right_filter = df[('stripe_right', 'likelihood')] >= p_cutoff

    # Then, determine the average coordinates of high-confidence boundary points
    checkered_Y = np.mean([df.loc[check_left_filter, ('check_left', 'y')].mean(), 
                        df.loc[check_right_filter, ('check_right', 'y')].mean()])


    striped_Y = np.mean([df.loc[stripe_left_filter, ('stripe_left', 'y')].mean(), 
                        df.loc[stripe_right_filter, ('stripe_right', 'y')].mean()])


    # Create empty array for effective position
    # Hold boolean per frame, evaluates to True if mouse is in specified ROI during frame
    in_stripe = []
    in_mid = []
    in_check = []

    # Loop through the first 600 seconds of frames to classify all ROI occupancy
    fps = int(fps)
    max_frames = 600 * fps if 600 * fps <= df.shape[0] else df.shape[0]
    for frame in range(max_frames):
        # Decide mouse position with y position from highest-priority visible bodypart
        temp_y = np.nan
        y = np.nan
        valid_bp = False
        
        # Analysis_choice 'A' == Center of body
        if analysis_choice == 'A':
            all_bps =[]
            average_bp = 0
            bodyparts_choices = ['tail', 'right_ear', 'left_ear']
            for bp in bodyparts_choices:
                if df[bp,'likelihood'][frame] >= p_cutoff:
                    temp_y = df.at[frame, (bp, 'y')]
                    all_bps.append(temp_y)
                    valid_bp = True
            # Average high confident features
            if valid_bp:
                average_bp = np.mean(all_bps)
                y = average_bp
        # Analysis_choice 'B' == Center of front of body
        elif analysis_choice == 'B': 
            all_bps =[]
            average_bp = 0
            bodyparts_choices = ['left_front', 'right_front', 'right_ear', 'left_ear', 'nose']
            for bp in bodyparts_choices:
                if df[bp,'likelihood'][frame] >= p_cutoff:
                    temp_y = df.at[frame, (bp, 'y')]
                    all_bps.append(temp_y)
                    valid_bp = True
            # Average high confident features
            if valid_bp:
                average_bp = np.mean(all_bps)
                y = average_bp
        # Analysis_choice 'C' == Based on paws
        else:
            all_bps =[]
            average_bp = 0
            # Ranking body parts based on how likely they are to be a good indicator of location
            bodyparts_choices_priority = ['left_hind', 'right_hind']
            bodyparts_choices_secondary = ['left_front', 'right_front']
            # Location is average of hind paws
            for bp in bodyparts_choices_priority:
                if df[bp,'likelihood'][frame] >= p_cutoff:
                    temp_y = df.at[frame, (bp, 'y')]
                    all_bps.append(temp_y)
                    valid_bp = True
            # If hind paws not visible, location is average of front paws
            if ~valid_bp:
                for bp in bodyparts_choices_secondary:
                    if df[bp,'likelihood'][frame] >= p_cutoff:
                        temp_y = df.at[frame, (bp, 'y')]
                        all_bps.append(temp_y)
                        valid_bp = True 
            # Average high confident features
            if valid_bp:
                average_bp = np.mean(all_bps)
                y = average_bp

        # Classify region based on X position
        if (valid_bp):
            # Bases on y position, determine ROI occupancy 
            temp_stripe = (y <= striped_Y)
            temp_mid = (y > striped_Y) and (y <= checkered_Y)
            temp_check = (not temp_stripe) and (not temp_mid)
            # Check for validating mouse movement, must go through middle to get to striped or checkered area
            if (frame >= 1):
                if (temp_stripe):
                    # If detected in stripe area
                    if(in_mid[frame-1] or in_stripe[frame-1]):
                        # Must have perviously been in middle or striped region
                        in_stripe.append(temp_stripe)
                        in_mid.append(temp_mid)
                        in_check.append(temp_check)
                    else:
                        # Mouse cannot go from checkered to striped so stays consistent with previous ROI
                        in_stripe.append(in_stripe[frame-1])
                        in_mid.append(in_mid[frame-1])
                        in_check.append(in_check[frame-1])
                elif (temp_check):
                    # If detected in checkered area
                    if(in_mid[frame-1] or in_check[frame-1]):
                        # Must have perviously been in middle or checkered region
                        in_stripe.append(temp_stripe)
                        in_mid.append(temp_mid)
                        in_check.append(temp_check)
                    else:
                        # Mouse cannot go from striped to checkered so stays consistent with previous ROI
                        in_stripe.append(in_stripe[frame-1])
                        in_mid.append(in_mid[frame-1])
                        in_check.append(in_check[frame-1])
                else:
                    # If detected in middle area
                    in_stripe.append(temp_stripe)
                    in_mid.append(temp_mid)
                    in_check.append(temp_check)
            else:
                in_stripe.append(temp_stripe)
                in_mid.append(temp_mid)
                in_check.append(temp_check)
        else:
            if (frame > 0):
                # If no body part was detected, Mouse stays in previous ROI
                in_stripe.append(in_stripe[frame-1])
                in_mid.append(in_mid[frame-1])
                in_check.append(in_check[frame-1])
            else:   
                # Default to checkered if no detection 
                in_stripe.append(False)
                in_mid.append(False)
                in_check.append(True)

    # Calulation video duration
    frame_count = len(in_stripe)
    duration = frame_count / fps

    # Add the results to the dataframe in the format specified above.
    results.loc[file_num-1, 'File Name'] = mouse_ID

    # Calculate time by summing frames per area, then dividing by frames per second
    results.loc[file_num-1, 'Seconds in Striped Area'] = sum(in_stripe) / fps
    results.loc[file_num-1, 'Seconds in Middle Area'] = sum(in_mid) / fps
    results.loc[file_num-1, 'Seconds in Checkered Area'] = sum(in_check) / fps
    results.loc[file_num-1, 'Total Duration'] = duration
        


# Save the results dataframe to a CSV file.
now = datetime.now()
timestamp = now.strftime("Date_%Y-%m-%d_Time_%H-%M")
# Create output folder
output_path = f'{os.getcwd()}/AnalyzedFinalData_600sec_{timestamp}'
os.makedirs(output_path, exist_ok=True)
# Paths for each result file into output directory
resFile = os.path.join(output_path, f'Results_Place_Preference_600sec_p{p_cutoff}_{timestamp}.csv')

# Save to correct directory
results.to_csv(resFile, index=False)

print("Analysis complete.")
print(f"Results saved to: {output_path}")