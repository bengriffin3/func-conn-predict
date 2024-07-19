""" Given a netmats from a chunk of the time series of HCP subjects, this script generated a prediction for a specific edge
    of a netmats for the entire length of the time series (i.e., the 'ground truth' netmats)

    Note that while some of the edges work instantly, some take up to 20+ mins, which is why we do 1 edge at a time rather than putting them in a for loop.

"""
#%% import modules
import os
import argparse
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import RepeatedKFold, KFold
import logging
from tqdm import trange
from nets_predict.classes.hmm import HiddenMarkovModelClass
import pickle
from nets_predict.classes.partial_correlation import PartialCorrelationClass

print("WHY IS THIS SCRIPT DIFFERENT TO THE OTHER ONE \n \n \n \n \n ")

#%% Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("n_ICs", type = int, help = 'No. ICs of parcellation', choices = [25, 50])
parser.add_argument("network_edge", type = int, help = 'Network edge to predict')
parser.add_argument("network_matrix", type = str, help = 'Netmats matrix to use as predictor', choices = ['icov', 'cov'])
parser.add_argument("prediction_matrix", type = str, help = 'Netmats matrix to predict', choices = ['icov', 'cov'])
parser.add_argument("n_chunk", type = int, help = 'No. chunks data has been split in to')
parser.add_argument("features_to_use", type = str, help = 'Which dynamic params (if any) shall we also use to generate predictions?', choices = ['static','fc','pc','means','tpms_ss', 'all', 'tpms_ss_only'])
parser.add_argument("--run", type = int, default = 1)
parser.add_argument("--n_states", type = int, default = 0, choices = [3,6,9,12,15,8,10])
parser.add_argument("--trans_prob_diag", type = int, default = 10)
parser.add_argument('--model_mean', default = True, action=argparse.BooleanOptionalAction, help = 'add flag --model_mean to model the mean, and add flag --no-model_mean to not model the mean') 
parser.add_argument("--n_folds", type = int, default = 10)

args = parser.parse_args()
n_ICs = args.n_ICs
network_edge = args.network_edge
network_matrix = args.network_matrix
prediction_matrix = args.prediction_matrix
n_chunk = args.n_chunk
n_folds = args.n_folds
features_to_use = args.features_to_use

#if features_to_use!='static':
run=args.run
n_states=args.n_states
trans_prob_diag=args.trans_prob_diag
model_mean=args.model_mean

n_folds = args.n_folds

# initialise logger and classes
logger = logging.getLogger("Chunk_project")
HMMClass = HiddenMarkovModelClass()
PartialCorrClass = PartialCorrelationClass()

#%% Set directories
proj_dir = "/well/win-fmrib-analysis/users/psz102/nets-predict/nets_predict"
results_dir = f"{proj_dir}/results/ICA_{n_ICs}"
chunk_save_dir = f"{results_dir}/edge_prediction/{n_chunk}_chunks"
os.makedirs(chunk_save_dir, exist_ok=True)

#%% Load features to predict with
# load static partial correlation matrix for chunked time series
if network_matrix == 'icov':
    netmats = np.load(f"{results_dir}/static/partial_correlations_{n_chunk}_chunks.npy") 
elif network_matrix == 'cov':
    netmats = np.load(f"{results_dir}/static/full_covariances_{n_chunk}_chunks.npy") 
    #pass # not using full covariances anymore


# load dynamic features (even if not used)
if features_to_use=='static':
    hmm_features_dict= [1,2,3,4,5,6,7,8,9,10,11,12,13]
else:
    dynamic_dir = f"{results_dir}/dynamic/run{run:02d}_states{n_states:02d}_DD{trans_prob_diag:06d}_model_mean_{model_mean}/{n_chunk}_chunks"
    with open(f'{dynamic_dir}/hmm_features_{n_chunk}_chunks.pickle', 'rb') as file:
        hmm_features_dict = pickle.load(file)


for edge in range(network_edge, network_edge+5):
    single_edge_prediction = f"{chunk_save_dir}/edge_prediction_{edge}_nm_{network_matrix}_pm_{prediction_matrix}_chunks_{n_chunk}_features_used_{features_to_use}_states_{n_states}_model_mean_{model_mean}.npz"
    joint_edge_prediction = f"{chunk_save_dir}/combined/edge_prediction_all_nm_{network_matrix}_pm_{prediction_matrix}_chunks_{n_chunk}_features_used_{features_to_use}_states_{n_states}_model_mean_{model_mean}.npz"
    if os.path.isfile(single_edge_prediction) or os.path.isfile(joint_edge_prediction):
        print(f"Already developed edge {edge}")
        #exit()
        continue
    else:
        print("Prediction not developed, running script...")

    # load and vectorise the ground truth matrix and then select a network edge to predict (for all subjects)
    if prediction_matrix == 'icov':
        ground_truth = np.load(f"{results_dir}/ground_truth/ground_truth_partial_mean_4_sessions.npy") 
    elif prediction_matrix == 'cov':
        ground_truth = np.load(f"{results_dir}/ground_truth/ground_truth_full_mean_4_sessions.npy") 
    
    ground_truth_edges = PartialCorrClass.extract_upper_off_main_diag(ground_truth)
    y = ground_truth_edges[:, edge]


    # define model evaluation method (this is to determine the hyperparameters) so we use 5 folds here for hyperparameter optimzation but 10 later on for the prediction
    cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=1)
    ratios = np.arange(0, 1, 0.1)
    if features_to_use=='tpms_ss_only':
        alphas = [0.0] # use linear regression because so few features
    else: 
        alphas = [1e-1, 0.0, 1.0, 10.0]

    model = ElasticNetCV(l1_ratio=ratios, alphas=alphas, cv=cv, n_jobs=-1, precompute=False)

    # intialise arrays
    n_features = HMMClass.determine_n_features(features_to_use, n_ICs, n_states)    
    n_sub = netmats.shape[0]
    alpha = np.zeros((n_chunk,n_folds))
    alpha[:] = np.nan 
    l1_ratio = np.zeros((n_chunk,n_folds))
    l1_ratio[:] = np.nan
    beta = np.zeros((n_chunk,n_folds,n_features))
    beta[:] = np.nan
    corr_y = np.zeros((n_chunk,n_folds))
    corr_y[:] = np.nan 
    predict_y_all = np.zeros((n_chunk,n_sub))
    predict_y_all[:] = np.nan
    prediction_accuracy_chunk = np.zeros((n_chunk, 1))

    # diagonal elements of netmats are all 0 so don't bother predicting these 
    logger.info(f"Running predictions using {n_features} features")

    # define Kfold cross-validation
    kf = KFold(n_splits=n_folds)

    for i in trange(n_chunk, desc='Chunks'):
        print(f"{chunk_save_dir}/edge_prediction_{edge}_nm_{network_matrix}_pm_{prediction_matrix}_chunks_{n_chunk}_features_used_{features_to_use}_states_{n_states}_model_mean_{model_mean}_chunk{i}.npz") 
        print(f"Checking if chunk {i} has been run before...")
        if os.path.isfile(f"{chunk_save_dir}/edge_prediction_{edge}_nm_{network_matrix}_pm_{prediction_matrix}_chunks_{n_chunk}_features_used_{features_to_use}_states_{n_states}_model_mean_{model_mean}_chunk{i}.npz"):
            print(f"Chunk {i} already run, continuing to next chunk... \n \n \n")
            np.load(f"{chunk_save_dir}/edge_prediction_{edge}_nm_{network_matrix}_pm_{prediction_matrix}_chunks_{n_chunk}_features_used_{features_to_use}_states_{n_states}_model_mean_{model_mean}_chunk{i}.npz")
            continue

        X = HMMClass.get_predictor_features(netmats[:,i,:,:], hmm_features_dict[i], features_to_use) 

        for fold, (train_index, test_index) in enumerate(kf.split(X)):
            print(f"Fold {fold}:")

            # set up cross-validation folds
            X_train = X[train_index,:]
            y_train = y[train_index]
            X_test = X[test_index,:]
            y_test = y[test_index]

            # standardise X
            X_train, X_test = HMMClass.standardise_train_apply_to_test(X_train, X_test)

            # centre response
            my = np.mean(y_train)
            y_train = y_train - my

            # Train the model using the training set
            model.fit(X_train, y_train)

            # save chosen configuration
            alpha[i,fold] = model.alpha_
            l1_ratio[i,fold] = model.l1_ratio_
            beta[i,fold,:] = model.coef_

            #return correlation
            y_pred, correlation = HMMClass.evaluate_model(model, X_test, y_test, my)
            predict_y_all[i,test_index] = y_pred
            corr_y[i,fold] = correlation
            
            print(corr_y[i,fold])

        prediction_accuracy_chunk[i] = pearsonr(np.squeeze(predict_y_all[i,:]), np.squeeze(y))[0]


        chunk_file_name = f"{chunk_save_dir}/edge_prediction_{edge}_nm_{network_matrix}_pm_{prediction_matrix}_chunks_{n_chunk}_features_used_{features_to_use}_states_{n_states}_model_mean_{model_mean}_chunk{i}.npz"
        np.savez(chunk_file_name, alpha=alpha, l1_ratio=l1_ratio, corr_y=corr_y, predict_y=predict_y_all, chunk=i, beta=beta, prediction_accuracy_chunk=prediction_accuracy_chunk)
        #previous_chunk_file_name = f"{chunk_save_dir}/edge_prediction_{edge}_nm_{network_matrix}_pm_{prediction_matrix}_chunks_{n_chunk}_features_used_{features_to_use}_states_{n_states}_model_mean_{model_mean}_chunk{i-1}.npz"
        #os.remove(previous_chunk_file_name)
    
    # save the proper edges here
    print(corr_y) 
    logger.info(f"Saving edge pred to: {chunk_save_dir}")
    np.savez(f"{chunk_save_dir}/edge_prediction_{edge}_nm_{network_matrix}_pm_{prediction_matrix}_chunks_{n_chunk}_features_used_{features_to_use}_states_{n_states}_model_mean_{model_mean}.npz", alpha=alpha, l1_ratio=l1_ratio, corr_y=corr_y, predict_y=predict_y_all, chunk=i, beta=beta, prediction_accuracy_chunk=prediction_accuracy_chunk)
