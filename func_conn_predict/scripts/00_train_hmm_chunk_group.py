"""
Train an HMM (in chunks/segments) at the group level.

We take the time series for each subject and divide it into a user-defined number of chunks, `n_chunks`. 
This creates a chunked time series array of shape (n_sub x n_chunk x n_timepoints_per_chunk x n_ICs), 
where `n_ICs` is the number of independent components (i.e., brain regions).

For each chunk, we concatenate the time series data across all subjects and fit an HMM to the concatenated series.
This process is repeated for each chunk.
"""

#%% Import packages
import os
import argparse
import logging
import pickle
import sys
from tqdm import trange

# OSLD imports
from osl_dynamics.data import Data
from osl_dynamics.models.hmm import Config, Model
from osl_dynamics.models import load

# Set directories
base_dir = "/gpfs3/well/win-fmrib-analysis/users/psz102/nets-predict/"
proj_dir = f"{base_dir}/nets_predict"
results_base_dir = f"{proj_dir}/results"
data_dir = f"{proj_dir}/data"

# my imports
sys.path.append(base_dir)
from nets_predict.classes.hmm import HMMInference, FeatureEngineering, TimeSeriesProcessing

#%% Initialise classes
time_series_processing = TimeSeriesProcessing()
feature_engineering = FeatureEngineering()

#%% Parse command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument("n_ICs", type=int, help='No. IC components for brain parcellation', choices = [25, 50])
parser.add_argument("n_states", type=int, help='No. HMM states', choices = [3, 6, 8, 9, 10, 12, 15])
parser.add_argument("run", type=int, help='HMM run')
parser.add_argument("trans_prob_diag", type=int, help='Prior on transition probability matrix')
parser.add_argument("n_subjects", type=int, help='Number of subjects to run model on', choices=range(1, 1004))
parser.add_argument("n_chunks", type=int, help='Number of chunks to divide time series in to')
parser.add_argument('--model_mean', default=True, action=argparse.BooleanOptionalAction, help='add flag --model_mean to model the mean, and add flag --no-model_mean to not model the mean') 
parser.add_argument('--use_group_model', default=False, action=argparse.BooleanOptionalAction, help='add flag --use_group_model to use full time series to perform DE, and add flag --no-use_group_model to use chunked time series for DE') 
parser.add_argument('--log_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Set logging level (default: INFO)')


args = parser.parse_args()
n_ICs = args.n_ICs
n_states = args.n_states
run = args.run
trans_prob_diag = args.trans_prob_diag
n_chunks = args.n_chunks
n_subjects = args.n_subjects
model_mean = args.model_mean
use_group_model = args.use_group_model

# Set up logging
logging.basicConfig(level=args.log_level.upper(),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
_logger = logging.getLogger(__name__)

#%% Load data
try:
    with open(os.path.join(data_dir, f"data_files_ICA{n_ICs}.txt"), "r") as file:
        inputs = [line.strip() for line in file if line.strip()]
except FileNotFoundError:
    _logger.error(f"Data file not found at {proj_dir}/data/data_files_ICA{n_ICs}.txt")
    sys.exit(1)

# Validate subject number
if len(inputs) < n_subjects:
    _logger.error(f"Requested {n_subjects} subjects, but only {len(inputs)} available.")
    sys.exit(1)

# Create Data object for training
_logger.debug(f"Loading data from {proj_dir}")
data = Data(inputs, load_memmaps=False, n_jobs=8)

# Create directory for results and check if HMMs have already been made
if use_group_model:
    results_dir = f"{results_base_dir}/1_chunks"
else:
    results_dir = f"{results_base_dir}/ICA_{n_ICs}/dynamic/run{run:02d}_states{n_states:02d}_DD{trans_prob_diag:06d}_model_mean_{model_mean}/{n_chunks}_chunks"

if os.path.isfile(f"{results_dir}/hmm_features_{n_chunks}_chunks.pickle"):
    _logger.info('exiting programme since HMM already developed')
    exit()

os.makedirs(results_dir, exist_ok=True)

# Prepare data
data.standardize()
_logger.info("Data standardized.")

#%% Initialise model
initial_trans_prob = feature_engineering.intialise_trans_prob(trans_prob_diag, n_states)

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

# Divide the time series into chunks
time_series_chunk =  time_series_processing.split_time_series(data, n_chunks)

# Initialize list for HMM features
hmm_features_per_chunk = []

#%% Train model
_logger.info("Training or loading model for each chunk of time series")
for chunk in trange(n_chunks, desc='Chunks'):
    # select time series of chunks
    time_series = time_series_chunk[1:n_subjects, chunk, :, :]
    
    # select chunk directory
    model_dir = f"{results_dir}/model_chunk_{chunk}"

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
        _logger.debug(f"Loading pre-trained HMM from {model_dir}")
        if use_group_model:
            model = load(f"{results_dir_full}/model_chunk_0")
        else: 
            model = load(model_dir)

    # organise the model parameters into a dictionary for selected chunk
    hmm_inference = HMMInference(model, time_series)
    HMM_params_dictionary = hmm_inference.get_inferred_parameters()
    hmm_features_per_chunk.append(HMM_params_dictionary)

# save model
_logger.info(f"Saving HMM features to {results_dir}")
output_file = os.path.join(results_dir, f"hmm_features_{n_chunks}_chunks.pickle")
with open(output_file, 'wb') as file:
    pickle.dump(hmm_features_per_chunk, file)

_logger.info(f"HMM features for {n_chunks} chunks saved at {output_file}")