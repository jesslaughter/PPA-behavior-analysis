# PPA-behavior-analysis
A reproducible deep-learning pipeline that imports computer vision tracking data of mouse position from video assay, segments regions of interest, and computes region of interest occupancy time during place preference assays conducted with wireless neuromodulation.

This repository contains the Python script AutoPlacePrefAnalysis_600seconds.py, which processes DeepLabCut (DLC) tracking data to quantify mouse occupancy in 3 visually distinct regions labeled, striped, middle, and checkered regions during a place preference assay.

The analysis uses frame-wise body-part likelihoods to determine mouse position with an adjustable confidence threshold, and enforces logical movement (e.g., mice must pass through the middle region when switching between the two preference zones).


Inputs: The DLC CSV files and corresponding .mp4 videos (Assumption, each video is >= 600 seconds.)
Outputs: A summary CSV reporting the number of seconds each subject spends in each region and the total analyzed duration

The summary CSV file is named Results_Place_Preference_600sec_p{p_cutoff}_{timestamp}.csv and will be located in the {pwd}/AnalyzedFinalData_600sec_{timestamp}. The output CSV file is formated every row is a different subject and the columns are File Name, Seconds in Striped Area, Seconds in Middle Area, Seconds in Checkered Area, and Total Duration.


 
Workflow of AutoPlacePrefAnalysis_600seconds.py:

Loads DLC CSV tracking files and corresponding .mp4 video files.

Extracts frame rate directly from each video.

Determines the mouse’s location using one of three user-selectable methods: A- Body center, B- Front-of-body, C- Paws (recommended, default)
  
Computes the Y-boundaries of the striped and checkered ROIs based on reliably detected boundary markers.

Classifies each frame into one of three regions.

Enforces valid region transitions.

Outputs a summary table reporting seconds spent in each region for each subject.

 

Additionally, this reposititory contains our trained DLC model, a sample our experimental dataset generated in the Bioelectronics Lab at MIT, and the corresponding output to illustrate proper input formatting and to show the expected output structure. All data is included strictly for instructional use.
