"""Train an HMM (in chunks) but at the group level first.

    We take the time series for each subject, one at a time, and divide it into {n_chunks} chunks (as defined by the user).
    By doing so, we create time_series_chunk (n_sub x n_chunk x n_timepoints_per_chunk x n_ICs/brain regions).
    We then select the time series data for all subjects for each chunk, one at a time, and fit an HMM to the concatenatenated time series.

"""

#%% Import packages
import os
import argparse
from osl_dynamics.data import Data
from osl_dynamics.models.hmm import Config, Model
from osl_dynamics.models import load
from nets_predict.classes.partial_correlation import PartialCorrelationClass
from nets_predict.classes.hmm import HiddenMarkovModelClass
import logging
from tqdm import trange
import pickle

PartialCorrClass = PartialCorrelationClass()
HMMClass = HiddenMarkovModelClass()
_logger = logging.getLogger("Chunk_project")

#%% Parse command line arguments and intialise classes
parser = argparse.ArgumentParser()
parser.add_argument("n_ICs", type=int, help='No. IC components for brain parcellation', choices = [25, 50])
parser.add_argument("n_states", type=int, help='No. HMM states', choices = [3, 6, 8, 9, 10, 12, 15])
parser.add_argument("run", type=int, help='HMM run')
parser.add_argument("trans_prob_diag", type=int, help='Prior on transition probability matrix')
parser.add_argument("n_chunks", type=int, help='Number of chunks to divide time series in to')
parser.add_argument('--model_mean', default=True, action=argparse.BooleanOptionalAction, help='add flag --model_mean to model the mean, and add flag --no-model_mean to not model the mean') 
parser.add_argument('--use_group_model', default=False, action=argparse.BooleanOptionalAction, help='add flag --use_group_model to use full time series to perform DE, and add flag --no-use_group_model to use chunked time series for DE') 


args = parser.parse_args()
n_ICs = args.n_ICs
n_states = args.n_states
run = args.run
trans_prob_diag = args.trans_prob_diag
n_chunks = args.n_chunks
model_mean = args.model_mean
use_group_model = args.use_group_model

#%% Load data

# Load the list of file names created by 1_find_data_files.py
proj_dir = "/well/win-fmrib-analysis/users/psz102/nets-predict/nets_predict"
with open(os.path.join(proj_dir, f"data/data_files_ICA{n_ICs}.txt"), "r") as file: # "r" means open in 'read' mode
    inputs = file.read().split("\n")

# Create Data object for training
data = Data(inputs, load_memmaps=False, n_jobs=8)

# Create directory for results
results_dir = f"{proj_dir}/results/ICA_{n_ICs}/dynamic/run{run:02d}_states{n_states:02d}_DD{trans_prob_diag:06d}_model_mean_{model_mean}/{n_chunks}_chunks"
results_dir_full = f"{proj_dir}/results/ICA_{n_ICs}/dynamic/run{run:02d}_states{n_states:02d}_DD{trans_prob_diag:06d}_model_mean_{model_mean}/1_chunks"
os.makedirs(results_dir, exist_ok=True)

if use_group_model:
    if os.path.isfile(f"{results_dir_full}/hmm_features_{n_chunks}_chunks.pickle"):
        print('exiting programme since HMM already developed')
        exit()
else:
    if os.path.isfile(f"{results_dir}/hmm_features_{n_chunks}_chunks.pickle"):
        print('exiting programme since HMM already developed')
        exit()

# Prepare data
data.standardize()

#%% Build model
initial_trans_prob = HMMClass.intialise_trans_prob(trans_prob_diag, n_states)

config = Config(
    n_states=n_states,
    n_channels=data.n_channels, # no. IC components (channels is a MEG term)
    sequence_length=50,
    learn_means=model_mean,
    learn_covariances=True,
    initial_trans_prob=initial_trans_prob,
    batch_size=512,
    learning_rate=0.01,
    n_epochs=20,
)

model = Model(config)
model.summary()

#%%  Divide the time series up into chunks
time_series_chunk =  PartialCorrClass.split_time_series(data, n_chunks)

# initialise list where we store HMM features per chunk (as a dictionary)
hmm_features_per_chunk = []

#%% Train model
_logger.info("Training model for each chunk of time series")
for chunk in trange(n_chunks, desc='Chunks'):
    # select time series of chunks
    time_series = time_series_chunk[:,chunk,:,:]
    
    if use_group_model:
        model_dir = f"{results_dir_full}/model_chunk_0"
    else:
        model_dir = f"{results_dir}/model_chunk_{chunk}"
    print(model_dir)
    if not os.path.isdir(f"{model_dir}"):
        _logger.info("Running full HMM")

        # might want to add multiple random initalisations here (and then choose from best)
        data = Data(time_series)
        init_history = model.random_state_time_course_initialization(data, n_init=3, n_epochs=1)
        
        # Full training
        history = model.fit(time_series)

        # Save model
        model.save(model_dir)
    else:
        _logger.info("Loading pre-trained HMM")
        if use_group_model:
            model = load(f"{results_dir_full}/model_chunk_0")
        else: 
            model = load(f"{results_dir}/model_chunk_{chunk}")

    HMM_params_dictionary = HMMClass.get_inferred_parameters(model, time_series)

    hmm_features_per_chunk.append(HMM_params_dictionary)

if use_group_model:    
    with open(os.path.join(results_dir_full, f"hmm_features_{n_chunks}_chunks.pickle"), 'wb') as file:
        pickle.dump(hmm_features_per_chunk, file)
else:
    with open(os.path.join(results_dir, f"hmm_features_{n_chunks}_chunks.pickle"), 'wb') as file:
        pickle.dump(hmm_features_per_chunk, file)

#%% Delete temporary directory
data.delete_dir()
