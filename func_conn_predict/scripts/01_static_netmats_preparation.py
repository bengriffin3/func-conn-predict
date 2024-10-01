"""
    Static netmats calculations for the 'ground truth' for HCP subjects.

    We calculate the 'ground truth' netmats for each subject by calculating the 
    (partial correlation r2z) netmats for each of the four 15-minute sessions 
    and then taking the average across 4 sessions. 

    We could just compute it over all 60 minutes too, but the average has been done
    in a previous paper (because the sessions are recorded separately, so averaging
    helps to reduce noise).
"""

#%% Import modules
import os
import sys
import logging
import argparse
import numpy as np
from scipy.stats import pearsonr
from tqdm import trange
from sklearn.metrics import r2_score
from osl_dynamics.data import Data

# Import custom classes
base_dir = "/gpfs3/well/win-fmrib-analysis/users/psz102/nets-predict"
proj_dir = f"{base_dir}/nets_predict"
sys.path.append(base_dir)
from nets_predict.classes.partial_correlation import PartialCorrelationAnalysis, PartialCorrelation, ProjectSetup, PartialCorrelationCalculation, GroundTruthPreparation
from nets_predict.classes.hmm import TimeSeriesProcessing

#%% Set up logger
_logger = logging.getLogger("fc_prediction")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#%% Parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Calculate static netmats for HCP subjects")
    parser.add_argument("n_ICs", type=int, help='Number of IC components of brain parcellation', choices=[25, 50, 100])
    parser.add_argument("n_chunks", type=int, help='How many chunks to divide the time series into?')
    parser.add_argument("--n_session", default=4, type=int, help='Number of sessions the data was recorded across') # Set to 1 to use all 60 minutes as "ground truth"
    # parser.add_argument("--apply_filter", default=0, type=int, help='Apply Butterworth filter to time series?', choices=[0, 1])
    # parser.add_argument("--sampling_freq", default=1.389, type=float, help='Sampling frequency for Butterworth filter (1.389 for HCP)')
    # parser.add_argument("--low_freq", default=0.05, type=float, help='Frequency cutoff for highpass filtering')
    return parser.parse_args()

project_setup = ProjectSetup()
partial_correlation_calculation = PartialCorrelationCalculation()
ground_truth_preparation = GroundTruthPreparation()

#%% Main script execution
if __name__ == "__main__":
    # Parse arguments and initialize parameters
    args = parse_arguments()
    n_ICs, n_chunks, n_session, n_edge = project_setup.initialize_parameters(args)
    
    # Initialise class instance
    time_series_processing = TimeSeriesProcessing()

    #%% Set directories
    [static_dir, ground_truth_dir] = project_setup.setup_directories(proj_dir, n_ICs)

    #%% Load and prepare data
    with open(f"{proj_dir}/data/data_files_ICA{n_ICs}.txt", "r") as file:
        inputs = file.read().split("\n")

    data = Data(inputs, load_memmaps=False, n_jobs=8)
    data.standardize()
    #data = time_series_processing.remove_bad_components(data) # don't have the bad components for HCP

    # develop and save ground truth matrix
    [ground_truth_matrix_partial, ground_truth_matrix_full] = ground_truth_preparation.prepare_ground_truth_matrices(data, n_session, ground_truth_dir)

    # Flatten partial correlation and full covairance matrices
    [ground_truth_matrix_partial_flatten, ground_truth_matrix_full_flatten] = ground_truth_preparation.flatten_matrices(ground_truth_matrix_partial, ground_truth_matrix_full)

    [partial_correlations_chunk, full_covariances_chunk, partial_correlations_chunk_flatten, full_covariances_chunk_flatten] = partial_correlation_calculation.calculate_partial_correlations_chunks(data, n_chunks)

    # Compute metrics
    [metrics, chunk_matrices] = partial_correlation_calculation.compute_metrics(n_edge, n_chunks, ground_truth_matrix_partial_flatten, ground_truth_matrix_full_flatten, partial_correlations_chunk_flatten, full_covariances_chunk_flatten)

    # Save results
    save_dir = f"{proj_dir}/results/ICA_{n_ICs}/edge_prediction/{n_chunks}_chunks/combined"
    partial_correlation_calculation.save_results(static_dir, save_dir, n_chunks, partial_correlations_chunk, full_covariances_chunk, metrics, chunk_matrices)