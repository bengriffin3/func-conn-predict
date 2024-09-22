""" Given a netmats from a chunk of the time series of HCP subjects, a previous script has generated a prediction for a specific edge
    of a netmats for the entire length of the time series (i.e., the 'ground truth' netmats)

    This script now combines all the edges for a given netmats.

    For example, for a 25 x 25 netmats, we combine the unique edges = n(n-1)/2 (note the diagonal of a partial correlation matrix are 0
    so we don't need to calcalate them.

    Note that while some of the edges work instantly, some take up to 20+ mins, which is why we do 1 edge at a time rather than putting them in a for loop.

"""
#%% Import modules
import os
import numpy as np
import argparse
from nets_predict.classes.hmm import HiddenMarkovModelClass
from nets_predict.classes.partial_correlation import PartialCorrelationClass
from tqdm import trange
from sklearn.metrics import r2_score

# intialise class
HMMClass = HiddenMarkovModelClass()
PCCClass = PartialCorrelationClass()


#%% Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("n_ICs", type=int, help='No. IC components of brain parcellation', choices = [25, 50])
parser.add_argument("n_chunk", type=int, help='How many chunks was the data divided into?', choices = [2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60, 120])
parser.add_argument("feature_type", type=str, help='Which features were used to run the prediction? fc, pc, means, tpms_ss, tpms_ss_only, static')
parser.add_argument("--prediction_model", default = 'elastic_net', type = str, help = 'Which prediction method to use?', choices = ['elastic_net','xgboost'])
parser.add_argument("--apply_filter", default=0, type=int, help='Was a Butterworth filter applied to the data?', choices = [0, 1])
parser.add_argument("--low_freq", default=0, type=float, help='What frequency filter was used for the highpass?')
parser.add_argument("--n_fold", default=10, type=int, help='How many folds were used for the cross-validation?', choices = [10])
parser.add_argument("--n_sub", default=1003, type=int, help='How many subjects were predictions run for?', choices = [1003])
parser.add_argument("--n_states", default=0, type=int, help='How many states were used for the HMM?')
parser.add_argument("--network_matrix", default="icov", type=str, help='Which matrix to use as features?', choices = ['cov', 'icov'])
parser.add_argument("--prediction_matrix", default="icov", type=str, help='Why matrix to predict?', choices = ['cov', 'icov'])

args = parser.parse_args()
n_ICs = args.n_ICs # 25
n_chunk = args.n_chunk # 12
feature_type = args.feature_type
n_fold = args.n_fold
n_sub = args.n_sub
n_states = args.n_states
network_matrix = args.network_matrix
prediction_matrix = args.prediction_matrix
apply_filter = args.apply_filter
prediction_model = args.prediction_model
if apply_filter==1:
   low_freq = args.low_freq
   low_freq = float(np.round(low_freq, 3))

#%% Set directories
proj_dir = '/gpfs3/well/win-fmrib-analysis/users/psz102/nets-predict/nets_predict'
load_dir = f"{proj_dir}/results/ICA_{n_ICs}/edge_prediction/{n_chunk}_chunks"
save_dir = f"{load_dir}/combined"
os.makedirs(save_dir, exist_ok=True)

#%% Calculating stats for dynamic predictions
model_mean = True
n_edge = int((n_ICs*(n_ICs-1))/2)
n_feat = HMMClass.determine_n_features(feature_type, n_ICs, n_states)

# load ground truth
static_dir = f"{proj_dir}/results/ICA_{n_ICs}"
if prediction_matrix == 'icov':
    ground_truth = np.load(f"{static_dir}/ground_truth/ground_truth_partial_mean_4_sessions.npy") 
elif prediction_matrix == 'cov':
    ground_truth = np.load(f"{static_dir}/ground_truth/ground_truth_full_mean_4_sessions.npy") 
    
ground_truth_edges = PCCClass.extract_upper_off_main_diag(ground_truth)

# initialise arrays
alpha = np.zeros((n_chunk, n_fold, n_edge))
alpha[:] = np.nan
l1_ratio = np.zeros((n_chunk, n_fold, n_edge))
l1_ratio[:] = np.nan
corr_y = np.zeros((n_chunk, n_fold, n_edge))
corr_y[:] = np.nan
predict_y = np.zeros((n_chunk, n_sub, n_edge))
predict_y[:] = np.nan
accuracy_per_edge = np.zeros((n_chunk, n_edge))
accuracy_per_edge[:] = np.nan
beta = np.zeros((n_chunk, n_fold, n_edge, n_feat))
beta[:] = np.nan
r2_accuracy_per_edge = np.zeros((n_chunk, n_edge))
r2_accuracy_per_edge[:] = np.nan


#for edge in trange(0, n_edge, desc='Saving info for edges...'):
for edge in range(0, n_edge, 2):
    
    if prediction_model=='xgboost':# or prediction_model=='elastic_net':
        edge_prediction_vars = np.load(f'{load_dir}/edge_prediction_{edge}_nm_{network_matrix}_pm_{prediction_matrix}_chunks_{n_chunk}_features_used_{feature_type}_states_{n_states}_model_mean_{model_mean}_model_{prediction_model}.npz')
    else: 
        if apply_filter==1:
            edge_prediction_vars = np.load(f'{load_dir}/edge_prediction_{edge}_nm_{network_matrix}_pm_{prediction_matrix}_chunks_{n_chunk}_features_used_{feature_type}_states_{n_states}_model_mean_{model_mean}_low_freq_{low_freq}'.replace('.', '_')+'.npz')
        elif apply_filter==0:
            edge_prediction_vars = np.load(f'{load_dir}/edge_prediction_{edge}_nm_{network_matrix}_pm_{prediction_matrix}_chunks_{n_chunk}_features_used_{feature_type}_states_{n_states}_model_mean_{model_mean}.npz')


    # predicted_edges_all[j, :,:,edge] = a['predict_y']
    alpha[:, :, edge] = edge_prediction_vars['alpha']
    l1_ratio[:, :, edge] = edge_prediction_vars['l1_ratio']
    corr_y[:, :, edge] = edge_prediction_vars['corr_y']
    predict_y[:, :, edge] = edge_prediction_vars['predict_y']
    accuracy_per_edge[:, edge] = edge_prediction_vars['prediction_accuracy_chunk'].flatten()

    for ifold in range(n_fold):
        for chunk in range(n_chunk):
            first_nan_index = np.where(np.isnan(edge_prediction_vars['beta'][chunk, ifold, :]))[0][0]
            n_feat = first_nan_index 
            beta[chunk, ifold, edge, 0:n_feat] = edge_prediction_vars['beta'][chunk, ifold, 0:n_feat]
        



for edge in trange(0, n_edge, desc='Saving info for edges...'):
    print(edge)
    for chunk in range(n_chunk): # n_chunk
        if sum(np.isnan(predict_y[chunk, :, edge]))>0:
            continue
        else: 
            r2_accuracy_per_edge[chunk, edge] = r2_score(ground_truth_edges[:,edge], predict_y[chunk, :, edge])



if prediction_model=='xgboost':# or prediction_model=='elastic_net':
    # if apply_filter==1:
    #     np.savez(f"{save_dir}/edge_prediction_all_nm_{network_matrix}_pm_{prediction_matrix}_chunks_{n_chunk}_features_used_{feature_type}_states_{n_states}_model_mean_{model_mean}_low_freq_{low_freq}".replace('.', '_')+f"_with_r2_model_{prediction_model}.npz", 
    #                 alpha=alpha, l1_ratio=l1_ratio, corr_y=corr_y, predict_y=predict_y, beta=beta, accuracy_per_edge=accuracy_per_edge, r2_accuracy_per_edge=r2_accuracy_per_edge) 
    # else:
    np.savez(f"{save_dir}/edge_prediction_all_nm_{network_matrix}_pm_{prediction_matrix}_chunks_{n_chunk}_features_used_{feature_type}_states_{n_states}_model_mean_{model_mean}_with_r2_model_{prediction_model}.npz", 
                alpha=alpha, l1_ratio=l1_ratio, corr_y=corr_y, predict_y=predict_y, beta=beta, accuracy_per_edge=accuracy_per_edge, r2_accuracy_per_edge=r2_accuracy_per_edge)     
else:
    if apply_filter==1:
        np.savez(f"{save_dir}/edge_prediction_all_nm_{network_matrix}_pm_{prediction_matrix}_chunks_{n_chunk}_features_used_{feature_type}_states_{n_states}_model_mean_{model_mean}_low_freq_{low_freq}".replace('.', '_')+".npz", 
                alpha=alpha, l1_ratio=l1_ratio, corr_y=corr_y, predict_y=predict_y, beta=beta, accuracy_per_edge=accuracy_per_edge)
        np.savez(f"{save_dir}/edge_prediction_all_nm_{network_matrix}_pm_{prediction_matrix}_chunks_{n_chunk}_features_used_{feature_type}_states_{n_states}_model_mean_{model_mean}_with_r2_low_freq_{low_freq}".replace('.', '_')+".npz", 
                alpha=alpha, l1_ratio=l1_ratio, corr_y=corr_y, predict_y=predict_y, beta=beta, accuracy_per_edge=accuracy_per_edge, r2_accuracy_per_edge=r2_accuracy_per_edge)
    elif apply_filter==0:
        np.savez(f"{save_dir}/edge_prediction_all_nm_{network_matrix}_pm_{prediction_matrix}_chunks_{n_chunk}_features_used_{feature_type}_states_{n_states}_model_mean_{model_mean}.npz", 
                alpha=alpha, l1_ratio=l1_ratio, corr_y=corr_y, predict_y=predict_y, beta=beta, accuracy_per_edge=accuracy_per_edge)
        np.savez(f"{save_dir}/edge_prediction_all_nm_{network_matrix}_pm_{prediction_matrix}_chunks_{n_chunk}_features_used_{feature_type}_states_{n_states}_model_mean_{model_mean}_with_r2.npz", 
                alpha=alpha, l1_ratio=l1_ratio, corr_y=corr_y, predict_y=predict_y, beta=beta, accuracy_per_edge=accuracy_per_edge, r2_accuracy_per_edge=r2_accuracy_per_edge)


# r2_per_edge_nm_icov_pm_icov_version2 = np.zeros((n_chunk, n_edge))
# r2_per_edge_nm_icov_pm_icov_version2_pred = np.zeros((n_chunk, n_edge))

# ground_truth_matrix_partial_flatten = feature_prediction_dict['actual']['ground_truth_matrix_partial_flatten']
# netmats_flatten = feature_prediction_dict['actual']['netmats_flatten']

# for edge in range(n_edge):
#     print(edge)
#     for chunk in range(n_chunk):
#         r2_per_edge_nm_icov_pm_icov_version2[chunk,edge] = r2_score(ground_truth_matrix_partial_flatten[:,edge], feature_prediction_dict['actual']['netmats_flatten'][chunk,:,edge])
#         r2_per_edge_nm_icov_pm_icov_version2_pred[chunk,edge] = r2_score(ground_truth_matrix_partial_flatten[:,edge], feature_prediction_dict['static']['predict_y'][chunk,:,edge])
# delete files which made up the individual edge predictions which we have now combined





