Scripts for 'netmats prediction' project
------------------------------------------------------

The goal of this project is to predict a 'ground truth' netmats, i.e., a partial correlation matrix developed from lots of brain scanning data (e.g., 60 minutes), where we assume noise is minimal due to the scanning time. To do this, we run the scripts contained in this project which will use different things to predict depending on what we ask for, including a netmats from a small chunk of time (e.g., 5 minutes), as well as HMM ffeatures developed on a small chunk of time.

The scripts are as follows:

- **00_train_hmm_chunk_group.py: Train a HMM on HCP time series data, which had been divided up into user specified chunks of time.
- **01_static_netmats_prepaaration.py: Generate a 'ground truth' netmat based on the whole time series.
- **02_netmats_edge_prediction.py: Predict the ground truth netmats (one edge at a time) using user specified features.
- **03_combine_edge_predictions.py: Combine the predictions from script 02...

Note that the scripts in `job_submission` can be used to submit jobs to the BMRC cluster. 

Note you need to activate the conda environment before running these scripts:

    conda activate osld 

