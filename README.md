# PPA-behavior-analysis
A reproducible deep-learning pipeline that imports computer vision tracking data of mouse position from video assay, segments regions of interest, and computes region of interest occupancy time during place preference assays conducted with wireless neuromodulation.

This repository contains the Python script 'AutoPlacePrefAnalysis_600seconds.py', which processes DeepLabCut (DLC) tracking data to quantify mouse occupancy in 3 visually distinct regions labeled, striped, middle, and checkered regions during a place preference assay. The DLC analysis was conducted with the 'config_place_pref_assay.yaml' file. The analysis uses frame-wise body-part likelihoods to determine mouse position with an adjustable confidence threshold, and enforces logical movement (e.g., mice must pass through the middle region when switching between the two preference zones).

## Data Folder Organization

Name the folder containing all DLC CSV files and video files: FinalData/

Inside FinalData/, place: All DLC .csv files, corresponding .mp4 videos 


## Data file name

DLC tracking output files should follow: <MouseID>DLC...csv

Corresponding videos should follow: <MouseID>.mp4

(e.g., Mouse1.mp4, Mouse1DLC_resnet50_place_pref_assay.csv):


## System Requirements

**Hardware**
A standard computer with sufficient RAM for CSV and video processing.

**Software**
No OS restrictions. Tested on: macOS (Apple M2 chip)

**Required Python packages:**

pandas

numpy

opencv-python

datetime (standard library)

os (standard library)


## Workflow of AutoPlacePrefAnalysis_600seconds.py:

Loads DLC CSV tracking files and corresponding .mp4 video files.

Extracts frame rate directly from each video.

Determines the mouse’s location using one of three user-selectable methods: A- Body center, B- Front-of-body, C- Paws (recommended, default)
  
Computes the Y-boundaries of the striped and checkered ROIs based on reliably detected boundary markers.

Classifies each frame into one of three regions.

Enforces valid region transitions.

Outputs a summary table reporting seconds spent in each region for each subject.

#### Inputs: 
The DLC CSV files and corresponding .mp4 videos (Assumption, each video is >= 600 seconds.)
#### Outputs: 
A summary CSV reporting the number of seconds each subject spends in each region and the total analyzed duration
The summary CSV file is named Results_Place_Preference_600sec_p{p_cutoff}_{timestamp}.csv and will be located in the {pwd}/AnalyzedFinalData_600sec_{timestamp}. The output CSV file is formated every row is a different subject and the columns are File Name, Seconds in Striped Area, Seconds in Middle Area, Seconds in Checkered Area, and Total Duration.
