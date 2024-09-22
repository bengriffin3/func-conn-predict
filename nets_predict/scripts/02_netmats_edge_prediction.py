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
from sklearn.model_selection import RepeatedKFold, KFold, GridSearchCV
import logging
from tqdm import trange
from nets_predict.classes.hmm import HiddenMarkovModelClass
import pickle
from nets_predict.classes.partial_correlation import PartialCorrelationClass
import time
import xgboost as xgb

start_time = time.time()

#%% Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("n_ICs", type = int, help = 'No. ICs of parcellation', choices = [25, 50])
parser.add_argument("network_edge", type = int, help = 'Network edge to predict')
parser.add_argument("network_matrix", type = str, help = 'Netmats matrix to use as predictor', choices = ['icov', 'cov'])
parser.add_argument("prediction_matrix", type = str, help = 'Netmats matrix to predict', choices = ['icov', 'cov'])
parser.add_argument("n_chunk", type = int, help = 'No. chunks data has been split in to')
parser.add_argument("features_to_use", type = str, help = 'Which dynamic params (if any) shall we also use to generate predictions?', choices = ['static','fc','pc','means','tpms_ss', 'all', 'tpms_ss_only', 'weighted_covs', 'weighted_icovs', 'static_pca', 'static_connecting_edges', 'static_self_edge_only'])
parser.add_argument("--prediction_model", default = 'elastic_net', type = str, help = 'Which prediction method to use?', choices = ['elastic_net','xgboost'])
parser.add_argument("--apply_filter", type = int, default = 0)
parser.add_argument("--low_freq", default=0, type=float, help='Level of frequency to be cutoff when highpass filtering?')
parser.add_argument("--run", type = int, default = 1)
parser.add_argument("--n_states", type = int, default = 0, choices = [3,6,9,12,15,8,10])
parser.add_argument("--trans_prob_diag", type = int, default = 10)
parser.add_argument('--model_mean', default = True, action=argparse.BooleanOptionalAction, help = 'add flag --model_mean to model the mean, and add flag --no-model_mean to not model the mean') 
parser.add_argument("--n_folds", type = int, default = 10)
parser.add_argument('--pca', default = False, action=argparse.BooleanOptionalAction, help = 'add flag --pca to perform PCA on feature matrix, and add flag --no-pca to not') 

args = parser.parse_args()
n_ICs = args.n_ICs
network_edge = args.network_edge
network_matrix = args.network_matrix
prediction_matrix = args.prediction_matrix
n_chunk = args.n_chunk
n_folds = args.n_folds
features_to_use = args.features_to_use
apply_filter = args.apply_filter
prediction_model = args.prediction_model
if apply_filter==1:
    low_freq = args.low_freq
    low_freq = float(np.round(low_freq, 3))
elif apply_filter==0:
    pass

#if features_to_use!='static':
run=args.run
n_states=args.n_states
trans_prob_diag=args.trans_prob_diag
model_mean=args.model_mean
pca=args.pca
pca_model = 0

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
if apply_filter==1:
    if network_matrix == 'icov':
        netmats = np.load(f"{results_dir}/static/partial_correlations_{n_chunk}_chunks_low_freq_{low_freq}".replace('.', '_')+".npy") 
    elif network_matrix == 'cov':
        netmats = np.load(f"{results_dir}/static/full_covariances_{n_chunk}_chunks_low_freq_{low_freq}".replace('.', '_')+".npy") 
elif apply_filter==0:
    if network_matrix == 'icov':
        netmats = np.load(f"{results_dir}/static/partial_correlations_{n_chunk}_chunks.npy") 
    elif network_matrix == 'cov':
        netmats = np.load(f"{results_dir}/static/full_covariances_{n_chunk}_chunks.npy") 


# load dynamic features (even if not used)
#if features_to_use=='static' or features_to_use=='static_pca' or features_to_use=='static_connecting_edges':
if "static" in features_to_use:
    hmm_features_dict= [i for i in range(1, 122)]
else:
    dynamic_dir = f"{results_dir}/dynamic/run{run:02d}_states{n_states:02d}_DD{trans_prob_diag:06d}_model_mean_{model_mean}/{n_chunk}_chunks"
    with open(f'{dynamic_dir}/hmm_features_{n_chunk}_chunks.pickle', 'rb') as file:
        hmm_features_dict = pickle.load(file)


for edge in range(network_edge, network_edge+2):
    if apply_filter==1:
        single_edge_prediction = f"{chunk_save_dir}/edge_prediction_{edge}_nm_{network_matrix}_pm_{prediction_matrix}_chunks_{n_chunk}_features_used_{features_to_use}_states_{n_states}_model_mean_{model_mean}_low_freq_{low_freq}".replace('.', '_')+".npz"
        joint_edge_prediction = f"{chunk_save_dir}/combined/edge_prediction_all_nm_{network_matrix}_pm_{prediction_matrix}_chunks_{n_chunk}_features_used_{features_to_use}_states_{n_states}_model_mean_{model_mean}_low_freq_{low_freq}".replace('.', '_')+".npz"
    elif apply_filter==0:
        single_edge_prediction = f"{chunk_save_dir}/2edge_prediction_{edge}_nm_{network_matrix}_pm_{prediction_matrix}_chunks_{n_chunk}_features_used_{features_to_use}_states_{n_states}_model_mean_{model_mean}.npz"
        joint_edge_prediction = f"{chunk_save_dir}/combined/2edge_prediction_all_nm_{network_matrix}_pm_{prediction_matrix}_chunks_{n_chunk}_features_used_{features_to_use}_states_{n_states}_model_mean_{model_mean}.npz"

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

    if prediction_model == 'elastic_net':
        model = ElasticNetCV(l1_ratio=ratios, alphas=alphas, cv=cv, n_jobs=-1, precompute=False)
    elif prediction_model == 'xgboost':
        model = xgb.XGBRegressor(n_jobs=1)
        # Define the parameter grid for hyperparameter tuning - 4 chunk, all edges, 0.7176705608488784, 0.8449748969562919
        # param_grid = {
        #     'alpha': [0, 0.1, 1, 2],  # Focus more on regularization
        #     'lambda': [1, 5, 10],
        #     'gamma': [0, 5, 10],
        #     'learning_rate': [0.1,0.05],  # Lower learning rates
        #     'max_depth': [3, 5, 7],  # Lower max depth
        #     'min_child_weight': [1, 3, 5], 
        #     'subsample': [0.6, 0.8], 
        #     'colsample_bytree': [0.6, 0.8], 
        #     #'n_estimators': [50, 100, 200]  # Slightly reduce the number of trees
        # }

        # working okay for 4 chunks
        # Define the parameter grid for hyperparameter tuning - 4 chunk, all edges, 0.7330292123166278 0.7770012537121369, but then 0.67..., then 0.75...
        param_grid = {
            'alpha': [1, 2, 5, 10],  # Focus more on regularization
            'lambda': [5, 10, 20],
            'gamma': [5, 10],
            'learning_rate': [0.05],  # Lower learning rates
            'max_depth': [1, 2],  # Lower max depth
            'min_child_weight': [3, 5], 
            'subsample': [0.4, 0.6], 
            'colsample_bytree': [0.4, 0.6], 
            'n_estimators': [50, 100, 200]  # Slightly reduce the number of trees
        }

        # This resulting in bad training (0.8) vs test (0.5)
        #         param_grid = {
        #     'alpha': [0, 0.1, 0.5, 1, 2, 5],
        #     'lambda': [1, 1.5, 2, 5, 10],
        #     'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        #     'max_depth': [1, 2, 3, 5, 7],
        #     'min_child_weight': [1, 3, 5],
        #     'subsample': [0.6, 0.8, 1.0], #[0.6, 0.8, 1.0],
        #     'colsample_bytree': [0.6, 0.8, 1.0],#, 0.8, 1.0],
        #     'n_estimators': [100, 200, 500]
        # }

        # this specific configuration results in training (0.58) vs test (0.53)
        # param_grid = {
        #     'alpha': [10], # higher = less likely to overfit
        #     'lambda': [10], # higher = less likely to overfit
        #     'learning_rate': [0.05], # higher = more likely to overfit
        #     'max_depth': [10], # higher = more likely to overfit
        #     'min_child_weight': [5], # higher = less likely to overfit
        #     'subsample': [0.7], # higher = more likely to overfit
        #     'colsample_bytree': [0.7], # higher = more likely to overfit
        #     'n_estimators': [50], #[100, 200, 500] # higher =more likely to overfit # Number of trees to fit
        #     'gamma': [10]
        # }

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
    corr_y_train = np.zeros((n_chunk,n_folds))
    corr_y_train[:] = np.nan 
    predict_y_all = np.zeros((n_chunk,n_sub))
    predict_y_all[:] = np.nan
    prediction_accuracy_chunk = np.zeros((n_chunk, 1))

    # diagonal elements of netmats are all 0 so don't bother predicting these 
    logger.info(f"Running predictions using {n_features} features")

    # define Kfold cross-validation
    kf = KFold(n_splits=n_folds)

    logger.info("Starting Chunk Loop...")

    for i in range(n_chunk):#, desc='Chunks'):
        logger.info(f"Checking if chunk {i} has been run before...") #print(f"{filename}") 
        filename = f"{chunk_save_dir}/edge_prediction_{edge}_nm_{network_matrix}_pm_{prediction_matrix}_chunks_{n_chunk}_features_used_{features_to_use}_states_{n_states}_model_mean_{model_mean}_chunk{i}.npz"

        #  here we check if the chunk has previously been run. If it has, load it and continue. If it hasn't (or the file is a bad file), then delete the file and continue
        if os.path.isfile(f"{filename}"):
            try:
                np.load(f"{filename}")
            except Exception as error:
                print(error)
                cmd = f"rm {filename}"
                print(cmd)
                os.system(cmd)
            else:
                print(f"Chunk {i} already run, continuing to next chunk... \n \n \n")
                np.load(f"{filename}")
                continue

        X = HMMClass.get_predictor_features(netmats[:,i,:,:], hmm_features_dict[i], features_to_use, edge)
        
        # form self edge + pca on remaining edges
        if features_to_use == 'static_pca':
            X = HMMClass.self_predict_plus_pca(X, edge)
        # just use the single self edge
        elif features_to_use == 'static_self_edge_only':
            X = X[:, edge].reshape(-1, 1)
        
        n_features = X.shape[1]

        for fold, (train_index, test_index) in enumerate(kf.split(X)):
            logger.info(f"Fold {fold}")

            # set up cross-validation folds
            X_train = X[train_index,:]
            y_train = y[train_index]
            X_test = X[test_index,:]
            y_test = y[test_index]

            # standardise X
            X_train, X_test = HMMClass.standardise_train_apply_to_test(X_train, X_test)
            
            # perform PCA
            n_features_dynamic = n_features - n_ICs*(n_ICs-1)/2
            if pca and n_features_dynamic > n_states*n_ICs:
                X_train, X_test, pca_model = HMMClass.pca_dynamic_only(X_train, X_test, n_ICs)
                n_features = X_train.shape[1]
                logger.info(f"Running predictions using {n_features} PCA features")
            else:
                pca=0

            # centre response
            my = np.mean(y_train)
            y_train = y_train - my

            if prediction_model=='xgboost':
                # Define the GridSearchCV object
                grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1) # cv=kf,

                # Fit the GridSearchCV object
                grid_search.fit(X_train, y_train)

                # Extract the best estimator
                best_model = grid_search.best_estimator_
                model = best_model

            # Train the model using the training set
            model.fit(X_train, y_train)
            

            # save chosen configuration
            if prediction_model == 'elastic_net':
                alpha[i,fold] = model.alpha_
                l1_ratio[i,fold] = model.l1_ratio_
                beta[i,fold,0:n_features] = model.coef_

            #return correlation
            y_pred_train, correlation_train = HMMClass.evaluate_model(model, X_train, y_train, my)
            y_pred, correlation = HMMClass.evaluate_model(model, X_test, y_test, my)
            predict_y_all[i,test_index] = y_pred
            corr_y[i,fold] = correlation
            corr_y_train[i,fold] = correlation_train
            print(correlation), print(correlation_train)
            logger.info(f"Training accuracy: {correlation_train}")
            logger.info(f"Test accuracy: {correlation}")

            logger.info(f"Pausing")
            time.sleep(10)
            
            print(corr_y[i,fold])

        prediction_accuracy_chunk[i] = pearsonr(np.squeeze(predict_y_all[i,:]), np.squeeze(y))[0]

        print("--- %s seconds ---" % (time.time() - start_time))
        chunk_file_name = f"{chunk_save_dir}/edge_prediction_{edge}_nm_{network_matrix}_pm_{prediction_matrix}_chunks_{n_chunk}_features_used_{features_to_use}_states_{n_states}_model_mean_{model_mean}_chunk{i}.npz"
        #np.savez(chunk_file_name, alpha=alpha, l1_ratio=l1_ratio, corr_y=corr_y, corr_y_train=corr_y_train, predict_y=predict_y_all, chunk=i, beta=beta, prediction_accuracy_chunk=prediction_accuracy_chunk, pca=pca, pca_model=pca_model)
        #previous_chunk_file_name = f"{chunk_save_dir}/edge_prediction_{edge}_nm_{network_matrix}_pm_{prediction_matrix}_chunks_{n_chunk}_features_used_{features_to_use}_states_{n_states}_model_mean_{model_mean}_chunk{i-1}.npz"
        #os.remove(previous_chunk_file_name)
    
    # save the proper edges here
    print(corr_y) 
    logger.info(f"Saving edge pred to: {chunk_save_dir}")

    if apply_filter==1:
        np.savez(f"{chunk_save_dir}/edge_prediction_{edge}_nm_{network_matrix}_pm_{prediction_matrix}_chunks_{n_chunk}_features_used_{features_to_use}_states_{n_states}_model_mean_{model_mean}_low_freq_{low_freq}".replace('.', '_')+f"_model_{prediction_model}.npz", alpha=alpha, l1_ratio=l1_ratio, corr_y=corr_y, corr_y_train=corr_y_train, predict_y=predict_y_all, chunk=i, beta=beta, prediction_accuracy_chunk=prediction_accuracy_chunk, pca=pca, pca_model=pca_model)
    elif apply_filter==0:
        np.savez(f"{chunk_save_dir}/edge_prediction_{edge}_nm_{network_matrix}_pm_{prediction_matrix}_chunks_{n_chunk}_features_used_{features_to_use}_states_{n_states}_model_mean_{model_mean}_model_{prediction_model}.npz", alpha=alpha, l1_ratio=l1_ratio, corr_y=corr_y, corr_y_train=corr_y_train, predict_y=predict_y_all, chunk=i, beta=beta, prediction_accuracy_chunk=prediction_accuracy_chunk, pca=pca, pca_model=pca_model)
