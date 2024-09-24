# Netmats Prediction Project

## Overview
This project predicts "ground truth" partial correlation matrices (netmats) derived from extensive brain scanning data. By leveraging various predictive features, such as Hidden Markov Model (HMM) features, this project provides predictions based on shorter time spans (e.g., 5 minutes).

## Usage

- **00_train_hmm_chunk_group.py: Train a HMM on HCP time series data, which had been divided up into user specified chunks of time.
- **01_static_netmats_prepaaration.py: Generate a 'ground truth' netmat based on the whole time series.
- **02_netmats_edge_prediction.py: Predict the ground truth netmats (one edge at a time) using user specified features.
- **03_combine_edge_predictions.py: Combine the predictions from script 02...
- **04_weighted_covs_asis.py: Weighted the covariances by their respective FOs

Scripts in `job_submission` can be used to submit jobs to the BMRC cluster. 

Note you need to activate the conda environment before running scripts running an HMM:

    conda activate osld
