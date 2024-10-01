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
import argparse
from nets_predict.classes.partial_correlation import PartialCorrelation
import numpy as np
from scipy.stats import pearsonr
from tqdm import trange
import pickle

#%% Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("n_ICs", type=int, help='No. IC components of brain parcellation', choices = [25, 50])
parser.add_argument("n_chunks", type=int, help='How many chunks do you want to divide the time series into?')
parser.add_argument("n_states", type=int, help='No. HMM states', choices = [3, 6, 8, 9, 10, 12, 15])
parser.add_argument("run", type=int, help='HMM run')
parser.add_argument("trans_prob_diag", type=int, help='Prior on transition probability matrix')
parser.add_argument('--model_mean', default=True, action=argparse.BooleanOptionalAction, help='add flag --model_mean to model the mean, and add flag --no-model_mean to not model the mean')
parser.add_argument("--n_session", default=4, type=int, help='How many sessions was the data recorded across?') # Set to 1 to develop ground truth over all 60 mins

args = parser.parse_args()
n_ICs = args.n_ICs
n_chunks = args.n_chunks
run = args.run
n_states = args.n_states
trans_prob_diag = args.trans_prob_diag
model_mean = args.model_mean
if args.n_session:
    n_session = args.n_session

n_edge = int((n_ICs * (n_ICs - 1))/2)

#%% Set directories
proj_dir = "/well/win-fmrib-analysis/users/psz102/nets-predict/nets_predict"
static_dir = f"{proj_dir}/results/ICA_{n_ICs}/static"
ground_truth_dir = f"{proj_dir}/results/ICA_{n_ICs}/ground_truth"

#%% Load ground truth matrix
ground_truth_matrix_partial = np.load(f"{ground_truth_dir}/ground_truth_partial_mean_4_sessions.npy")

partial_correlation = PartialCorrelation(ground_truth_matrix_partial)
ground_truth_matrix_partial_flatten = partial_correlation.extract_upper_off_main_diag()

# here we load the HMM feature dictionary, including the weighted covs
dynamic_dir = f"{proj_dir}/results/ICA_{n_ICs}/dynamic/run{run:02d}_states{n_states:02d}_DD{trans_prob_diag:06d}_model_mean_{model_mean}/{n_chunks}_chunks"
with open(f'{dynamic_dir}/hmm_features_{n_chunks}_chunks.pickle', 'rb') as file:
    hmm_features_dict = pickle.load(file)


# determine accuracy of the chunked up partial corr vs ground truth partial corr
n_sub = ground_truth_matrix_partial.shape[0]
accuracy_per_edge_nm_icov_pm_icov_weighted_covs = np.zeros((n_chunks, n_edge))
accuracy_per_edge_nm_icov_pm_icov_weighted_icovs = np.zeros((n_chunks, n_edge))
weighted_covs_flatten = np.zeros((n_sub, n_chunks, n_edge))
weighted_icovs_flatten = np.zeros((n_sub, n_chunks, n_edge))

for chunk in range(n_chunks):
    # covs
    weighted_covs = hmm_features_dict[chunk]['weighted_covs_chunk']
    weighted_covs_sum = np.sum(weighted_covs, axis=1)
    partial_correlation = PartialCorrelation(weighted_covs_sum)
    weighted_covs_flatten[:, chunk, :] = PartialCorrClass.extract_upper_off_main_diag()

    #icovs
    weighted_icovs = hmm_features_dict[chunk]['weighted_icovs_chunk']
    weighted_icovs = np.nan_to_num(weighted_icovs) # icovs gives nan diagonal
    weighted_icovs_sum = np.sum(weighted_icovs, axis=1)
    partial_correlation = PartialCorrelation(weighted_icovs_sum)
    weighted_icovs_flatten[:, chunk, :] = PartialCorrClass.extract_upper_off_main_diag()

    for edge in trange(n_edge, desc='Getting accuracy per edge:'):
        accuracy_per_edge_nm_icov_pm_icov_weighted_covs[chunk,edge] = pearsonr(weighted_covs_flatten[:,chunk, edge], ground_truth_matrix_partial_flatten[:,edge])[0]
        accuracy_per_edge_nm_icov_pm_icov_weighted_icovs[chunk,edge] = pearsonr(weighted_icovs_flatten[:,chunk, edge], ground_truth_matrix_partial_flatten[:,edge])[0]

# save partial correlation matrices and accuracy
save_dir = f"{proj_dir}/results/ICA_{n_ICs}/edge_prediction/{n_chunks}_chunks/combined"
os.makedirs(save_dir, exist_ok=True)

# save edge accuracies
np.savez(f"{save_dir}/edge_prediction_all_nm_icov_pm_icov_chunks_{n_chunks}_features_used_weighted_covs_states_{n_states}_model_mean_{model_mean}_actual.npz", accuracy_per_edge=accuracy_per_edge_nm_icov_pm_icov_weighted_covs, netmats_flatten = weighted_covs_flatten.transpose(1, 0, 2))
np.savez(f"{save_dir}/edge_prediction_all_nm_icov_pm_icov_chunks_{n_chunks}_features_used_weighted_icovs_states_{n_states}_model_mean_{model_mean}_actual.npz", accuracy_per_edge=accuracy_per_edge_nm_icov_pm_icov_weighted_icovs, netmats_flatten = weighted_icovs_flatten.transpose(1, 0, 2))
# %%
