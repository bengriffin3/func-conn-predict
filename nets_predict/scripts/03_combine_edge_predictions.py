#%% Import modules
import os
import numpy as np
import argparse
from nets_predict.classes.hmm import HiddenMarkovModelClass
import logging
from tqdm import trange

HMMClass = HiddenMarkovModelClass()
_logger = logging.getLogger("Chunk_project")


#%% Parse command line arguments and intialise classes
parser = argparse.ArgumentParser()
parser.add_argument("n_ICs", type=int, help='No. IC components of brain parcellation', choices = [25, 50])
parser.add_argument("n_chunk", type=int, help='How many chunks was the data divided into?', choices = [4, 12])
parser.add_argument("feature_type", type=str, help='Which features were used to run the prediction?')
parser.add_argument("--n_fold", default=10, type=int, help='How many folds were used for the cross-validation?')
parser.add_argument("--n_sub", default=1003, type=int, help='How many subjects were predictions run for?')
parser.add_argument("--n_states", default=0, type=int, help='How many states were used for the HMM?')
parser.add_argument("--network_matrix", default="icov", type=str, help='Which matrix to use as features?')
parser.add_argument("--prediction_matrix", default="icov", type=str, help='Why matrix to predict?')

args = parser.parse_args()
n_ICs = args.n_ICs # 25
n_chunk = args.n_chunk # 12
feature_type = args.feature_type
n_fold = args.n_fold
n_sub = args.n_sub
n_states = args.n_states
network_matrix = args.network_matrix
prediction_matrix = args.prediction_matrix

proj_dir = '/gpfs3/well/win-fmrib-analysis/users/psz102/nets_project/nets_predict'
load_dir = f"{proj_dir}/results/ICA_{n_ICs}/edge_prediction/{n_chunk}_chunks"
save_dir = f"{load_dir}/combined"
os.makedirs(save_dir, exist_ok=True)

#%% Calculating stats for dynamic predictions

model_mean = True
n_edge = int((n_ICs*(n_ICs-1))/2)


n_feat = HMMClass.determine_n_features(feature_type, n_ICs, n_states)

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

for edge in trange(0, n_edge, desc='Saving info for edges...'):

    edge_prediction_vars = np.load(f'{load_dir}/edge_prediction_{edge}_nm_{network_matrix}_pm_{prediction_matrix}_chunks_{n_chunk}_features_used_{feature_type}_states_{n_states}_model_mean_{model_mean}.npz')

    # predicted_edges_all[j, :,:,edge] = a['predict_y']
    alpha[:, :, edge] = edge_prediction_vars['alpha']
    l1_ratio[:, :, edge] = edge_prediction_vars['l1_ratio']
    corr_y[:, :, edge] = edge_prediction_vars['corr_y']
    predict_y[:, :, edge] = edge_prediction_vars['predict_y']
    accuracy_per_edge[:, edge] = edge_prediction_vars['prediction_accuracy_chunk'].flatten()

    beta[:,:,edge, :] = edge_prediction_vars['beta']
        

np.savez(f"{save_dir}/edge_prediction_all_nm_{network_matrix}_pm_{prediction_matrix}_chunks_{n_chunk}_features_used_{feature_type}_states_{n_states}_model_mean_{model_mean}.npz", 
        alpha=alpha, l1_ratio=l1_ratio, corr_y=corr_y, predict_y=predict_y, beta=beta, accuracy_per_edge=accuracy_per_edge)

# delete files which made up the individual edge predictions which we have now combined





