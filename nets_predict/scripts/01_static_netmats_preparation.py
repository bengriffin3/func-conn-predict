""" 
    Static netmats calculations for the 'ground truth' for HCP subjects.

    We calculate the 'ground truth' netmats for each subject by calculating the 
    (partial correlation r2z) netmats for each of the four 15 minutes sessions 
    and then taking the average across 4 sessions. 

    We could just find it over all 60 minutes too but Steve has done the average 
    in a previous paper (because the sessions are recoreded separately so by taking
    the average we are trying to remove the noise).

    I think meaning across covariances for each of the 4 sessions is equivalent
    to just treating it at one 60 minute session.

"""
#%% Import modules
import os
from osl_dynamics.data import Data
import argparse
from nets_predict.classes.partial_correlation import PartialCorrelationClass
import numpy as np
from scipy.stats import pearsonr
from tqdm import trange

#%% Parse command line arguments and intialise classes
parser = argparse.ArgumentParser()
parser.add_argument("n_ICs", type=int, help='No. IC components of brain parcellation', choices = [25, 50])
parser.add_argument("n_chunks", type=int, help='How many chunks do you want to divide the time series into?')
parser.add_argument("--n_session", default=4, type=int, help='How many sessions was the data recorded across?') # Set to 1 to develop ground truth over all 60 mins

args = parser.parse_args()
n_ICs = args.n_ICs
n_chunks = args.n_chunks
if args.n_session:
    n_session = args.n_session

n_edge = int((n_ICs * (n_ICs - 1))/2)

# Initialise class instance
PartialCorrClass = PartialCorrelationClass()

#%% Set directories
proj_dir = "/well/win-fmrib-analysis/users/psz102/nets_project/nets_predict"
static_dir = f"{proj_dir}/results/ICA_{n_ICs}/static"
ground_truth_dir = f"{proj_dir}/results/ICA_{n_ICs}/ground_truth"
os.makedirs(static_dir, exist_ok=True)
os.makedirs(ground_truth_dir, exist_ok=True)

#%% Load and prepare data
with open(f"{proj_dir}/data/data_files_ICA{n_ICs}.txt", "r") as file:
    inputs = file.read().split("\n")

data = Data(inputs, load_memmaps=False, n_jobs=8)
data.standardize()

# develop and ground truth matrix
ground_truth_matrix_partial, ground_truth_matrix_full = PartialCorrClass.get_ground_truth_matrix(data, n_session)
np.save(f"{ground_truth_dir}/ground_truth_partial_mean_{n_session}_sessions.npy", ground_truth_matrix_partial)
np.save(f"{ground_truth_dir}/ground_truth_full_mean_{n_session}_sessions.npy", ground_truth_matrix_full)
ground_truth_matrix_partial_flatten = PartialCorrClass.extract_upper_off_main_diag(ground_truth_matrix_partial)
ground_truth_matrix_full_flatten = PartialCorrClass.extract_upper_off_main_diag(ground_truth_matrix_full)

# Determine the partial correlation matrix for each chunk of time and flatten
time_series_chunk =  PartialCorrClass.split_time_series(data, n_chunks) # creates array of size (n_sub x n_chunk x n_IC x n_IC)
partial_correlations_chunk, full_covariances_chunk = PartialCorrClass.get_partial_correlation_chunk(time_series_chunk, n_chunks) # get partial correlation matrix
partial_correlations_chunk_flatten = PartialCorrClass.extract_upper_off_main_diag(partial_correlations_chunk) # extract upper diagonal (excluding main diagonal)
full_covariances_chunk_flatten = PartialCorrClass.extract_upper_off_main_diag(full_covariances_chunk) # extract upper diagonal (excluding main diagonal)

# determine accuracy of the chunked up partial corr vs ground truth partial corr
accuracy_per_edge_nm_icov_pm_icov = np.zeros((n_chunks, n_edge))
accuracy_per_edge_nm_icov_pm_icov[:] = np.nan
accuracy_per_edge_nm_cov_pm_icov = np.zeros((n_chunks, n_edge))
accuracy_per_edge_nm_cov_pm_icov[:] = np.nan
accuracy_per_edge_nm_icov_pm_cov = np.zeros((n_chunks, n_edge))
accuracy_per_edge_nm_icov_pm_cov[:] = np.nan
accuracy_per_edge_nm_cov_pm_cov = np.zeros((n_chunks, n_edge))
accuracy_per_edge_nm_cov_pm_cov[:] = np.nan

for edge in trange(n_edge, desc='Getting accuracy per edge:'):
    for chunk in range(n_chunks):
        accuracy_per_edge_nm_icov_pm_icov[chunk,edge] = pearsonr(partial_correlations_chunk_flatten[:,chunk,edge], ground_truth_matrix_partial_flatten[:,edge])[0]
        accuracy_per_edge_nm_cov_pm_icov[chunk,edge] = pearsonr(full_covariances_chunk_flatten[:,chunk,edge], ground_truth_matrix_partial_flatten[:,edge])[0]
        accuracy_per_edge_nm_icov_pm_cov[chunk,edge] = pearsonr(partial_correlations_chunk_flatten[:,chunk,edge], ground_truth_matrix_full_flatten[:,edge])[0]
        accuracy_per_edge_nm_cov_pm_cov[chunk,edge] = pearsonr(full_covariances_chunk_flatten[:,chunk,edge], ground_truth_matrix_full_flatten[:,edge])[0]

# save partial correlation matrices and accuracy
save_dir = f"{proj_dir}/results/ICA_{n_ICs}/edge_prediction/{n_chunks}_chunks/combined"
os.makedirs(save_dir, exist_ok=True)
np.save(f"{static_dir}/partial_correlations_{n_chunks}_chunks.npy", partial_correlations_chunk)
np.save(f"{static_dir}/full_covariances_{n_chunks}_chunks.npy", full_covariances_chunk)

# save edge accuracies
np.savez(f"{save_dir}/edge_prediction_all_nm_icov_pm_icov_chunks_{n_chunks}_features_used_actual.npz", accuracy_per_edge=accuracy_per_edge_nm_icov_pm_icov)
np.savez(f"{save_dir}/edge_prediction_all_nm_cov_pm_icov_chunks_{n_chunks}_features_used_actual.npz", accuracy_per_edge=accuracy_per_edge_nm_cov_pm_icov)
np.savez(f"{save_dir}/edge_prediction_all_nm_icov_pm_cov_chunks_{n_chunks}_features_used_actual.npz", accuracy_per_edge=accuracy_per_edge_nm_icov_pm_cov)
np.savez(f"{save_dir}/edge_prediction_all_nm_cov_pm_cov_chunks_{n_chunks}_features_used_actual.npz", accuracy_per_edge=accuracy_per_edge_nm_cov_pm_cov)

