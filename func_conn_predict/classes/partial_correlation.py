import numpy as np
import sys
import os
import logging
from tqdm import trange
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

_logger = logging.getLogger("Chunk_project")

base_dir = "/gpfs3/well/win-fmrib-analysis/users/psz102/nets-predict"
sys.path.append(base_dir)

class CovarianceUtils:
    """Utility class for covariance and precision calculations."""

    @staticmethod
    def covariance_to_precision(Sigma, rho=0.1):
        """Calculate precision matrix from covariance matrix using Cholesky decomposition."""
        np.fill_diagonal(Sigma, Sigma.diagonal() + rho)
        R = np.linalg.cholesky(Sigma)
        R_inv = np.linalg.inv(R)
        return np.transpose(R_inv) @ R_inv

    # @staticmethod
    # def fisher_transform(partial_corr, scaling_factor=-18.8310):
    #     """Apply Fisher transformation and scale z-scores."""
    #     partial_corr_r2z = 0.5 * np.log((1 + partial_corr) / (1 - partial_corr))
    #     return partial_corr_r2z * (-scaling_factor)

    @staticmethod
    def fisher_transform(partial_corr, scaling_factor=-18.8310):
        """Apply Fisher transformation and scale z-scores."""
        partial_corr_r2z = 0.5 * np.log((1 + partial_corr) / (1 - partial_corr))
        np.fill_diagonal(partial_corr_r2z, 0)  # Set diagonal to 0
        return partial_corr_r2z * (-scaling_factor)

class PartialCorrelation:
    """Class for running partial correlation analysis."""

    def __init__(self, fc_matrix, rho=0.1, scaling_factor=-18.8310):
        self.fc_matrix = fc_matrix
        self.rho = rho
        self.scaling_factor = scaling_factor
        self.utils = CovarianceUtils()

    def partial_corr(self):
        """Calculate partial correlation matrix from covariance matrix."""
        fc_matrix_norm = self.fc_matrix / np.sqrt(np.mean(np.diag(self.fc_matrix) ** 2))  # Normalize
        precision = self.utils.covariance_to_precision(fc_matrix_norm, self.rho)

        pii = np.sqrt(np.abs(np.diag(precision)))
        pii_rep = np.stack([pii] * len(self.fc_matrix), axis=1)
        pjj_rep = np.stack([pii] * len(self.fc_matrix), axis=0)

        partial_corr = (-precision / pii_rep) / pjj_rep
        np.fill_diagonal(partial_corr, 0)

        partial_corr_r2z = self.utils.fisher_transform(partial_corr, self.scaling_factor)
        return partial_corr, partial_corr_r2z, precision
   
    def extract_upper_off_main_diag(self):
        self.fc_matrix = np.squeeze(self.fc_matrix)
        m, n = np.triu_indices(self.fc_matrix.shape[-1], k=1)  # Use k=1 for upper diagonal
        return self.fc_matrix[..., m, n]

    def find_original_indices(self, edge_index):
        row, col = np.triu_indices(self.fc_matrix.shape[-1], 1)
        return row[edge_index], col[edge_index]


class PartialCorrelationAnalysis:
    """Class for analyzing partial correlations from time series data."""

    def __init__(self, time_series):
        self.time_series = time_series

    def get_ground_truth_matrix(self, n_ICs, n_session):
        n_sub = len(self.time_series)
        
        # Initialize arrays
        partial_correlation_session = np.zeros((n_sub, n_session, n_ICs, n_ICs))
        full_covariances_session = np.zeros((n_sub, n_session, n_ICs, n_ICs))

        _logger.info("Generating netmats")
        for sub in trange(n_sub, desc='Subjects'):
            time_series_session = np.split(self.time_series[sub], n_session)

            for sess in range(n_session):
                fc_matrix = np.cov(time_series_session[sess], rowvar=False)
                full_covariances_session[sub, sess] = fc_matrix
                partial_correlation = PartialCorrelation(fc_matrix)
                partial_correlation_session[sub, sess] = partial_correlation.partial_corr()[1]

        return (np.mean(partial_correlation_session, axis=1).squeeze(), 
                np.mean(full_covariances_session, axis=1).squeeze())

    def get_partial_correlation_chunk(self, time_series_chunk, n_chunks):
        n_sub = time_series_chunk.shape[0]
        n_ICs = time_series_chunk.shape[3]
        
        partial_correlations_chunk = np.zeros((n_sub, n_chunks, n_ICs, n_ICs))
        full_covariances_chunk = np.zeros((n_sub, n_chunks, n_ICs, n_ICs))

        _logger.info("Generating partial correlations per subject")
        for sub in trange(n_sub, desc='Subjects'):
            for chunk in range(n_chunks):
                fc_matrix = np.cov(time_series_chunk[sub, chunk], rowvar=False)
                full_covariances_chunk[sub, chunk] = fc_matrix
                partial_correlation = PartialCorrelation(fc_matrix)
                partial_correlations_chunk[sub, chunk] = partial_correlation.partial_corr()[1]

        return partial_correlations_chunk, full_covariances_chunk

class ProjectSetup:
    """Handles the initialization and directory setup for the project."""

    @staticmethod
    def initialize_parameters(args):
        """Initialize key variables and configurations."""
        n_ICs = args.n_ICs
        n_chunks = args.n_chunks
        n_session = args.n_session
        n_edge = int((n_ICs * (n_ICs - 1)) / 2)  # Calculate number of edges
        _logger.info(f"Initialized with n_ICs={n_ICs}, n_chunks={n_chunks}, n_sessions={n_session}")
        return n_ICs, n_chunks, n_session, n_edge

    @staticmethod
    def setup_directories(proj_dir, n_ICs):
        """Set up directories."""
        static_dir = f"{proj_dir}/results/ICA_{n_ICs}/static"
        ground_truth_dir = f"{proj_dir}/results/ICA_{n_ICs}/ground_truth"
        os.makedirs(static_dir, exist_ok=True)
        os.makedirs(ground_truth_dir, exist_ok=True)
        return static_dir, ground_truth_dir


class GroundTruthPreparation:
    """Handles preparation and flattening of ground truth matrices."""

    def prepare_ground_truth_matrices(self, data, n_session, ground_truth_dir):
        """Prepares ground truth matrices."""
        partial_correlation_analysis = PartialCorrelationAnalysis(data.time_series())
        ground_truth_matrix_partial, ground_truth_matrix_full = partial_correlation_analysis.get_ground_truth_matrix(data.n_channels, n_session)
        np.save(f"{ground_truth_dir}/ground_truth_partial_mean_{n_session}_sessions.npy", ground_truth_matrix_partial)
        np.save(f"{ground_truth_dir}/ground_truth_full_mean_{n_session}_sessions.npy", ground_truth_matrix_full)
        return ground_truth_matrix_partial, ground_truth_matrix_full

    def flatten_matrices(self, ground_truth_matrix_partial, ground_truth_matrix_full):
        """Flattens the matrices to the upper triangular elements."""
        partial_correlation = PartialCorrelation(ground_truth_matrix_partial)
        ground_truth_matrix_partial_flatten = partial_correlation.extract_upper_off_main_diag()

        partial_correlation = PartialCorrelation(ground_truth_matrix_full)
        ground_truth_matrix_full_flatten = partial_correlation.extract_upper_off_main_diag()

        return ground_truth_matrix_partial_flatten, ground_truth_matrix_full_flatten


class PartialCorrelationCalculation:
    """Handles chunk-based partial correlation calculation and metrics computation."""

    def calculate_partial_correlations_chunks(self, data, n_chunks):
        """Calculates partial correlations for data chunks."""
        from func_conn_predict.classes.hmm import TimeSeriesProcessing
        
        time_series_processing = TimeSeriesProcessing()
        time_series_chunk = time_series_processing.split_time_series(data, n_chunks)

        partial_correlation_analysis = PartialCorrelationAnalysis(data.time_series())
        partial_correlations_chunk, full_covariances_chunk = partial_correlation_analysis.get_partial_correlation_chunk(time_series_chunk, n_chunks)

        partial_correlation = PartialCorrelation(partial_correlations_chunk)
        partial_correlations_chunk_flatten = partial_correlation.extract_upper_off_main_diag()

        partial_correlation = PartialCorrelation(full_covariances_chunk)
        full_covariances_chunk_flatten = partial_correlation.extract_upper_off_main_diag()

        return partial_correlations_chunk, full_covariances_chunk, partial_correlations_chunk_flatten, full_covariances_chunk_flatten


    def compute_metrics(self, n_edge, n_chunks, ground_truth_matrix_partial_flatten, ground_truth_matrix_full_flatten, partial_correlations_chunk_flatten, full_covariances_chunk_flatten):
        """Computes accuracy and R2 metrics for each edge."""

        # Define the pairings of chunks and ground truth matrices
        chunk_matrices = {
            'nm_icov_pm_icov': ('partial_correlations_chunk_flatten', 'ground_truth_matrix_partial_flatten'),
            'nm_cov_pm_icov': ('full_covariances_chunk_flatten', 'ground_truth_matrix_partial_flatten'),
            'nm_icov_pm_cov': ('partial_correlations_chunk_flatten', 'ground_truth_matrix_full_flatten'),
            'nm_cov_pm_cov': ('full_covariances_chunk_flatten', 'ground_truth_matrix_full_flatten')
        }

        metrics = {
            'accuracy': {key: np.zeros((n_chunks, n_edge)) for key in chunk_matrices},
            'r2': {key: np.zeros((n_chunks, n_edge)) for key in chunk_matrices}
        }

        for edge in trange(n_edge, desc='Getting accuracy per edge:'):
            for chunk in range(n_chunks):
                for key, (chunk_matrix, ground_truth_matrix) in chunk_matrices.items():
                    chunk_data = eval(f"{chunk_matrix}[:, chunk, edge]")
                    ground_truth_data = eval(f"{ground_truth_matrix}[:, edge]")

                    metrics['accuracy'][key][chunk, edge] = pearsonr(chunk_data, ground_truth_data)[0]
                    metrics['r2'][key][chunk, edge] = r2_score(ground_truth_data, chunk_data)

        return metrics, chunk_matrices

    def save_results(self, static_dir, save_dir, n_chunks, partial_correlations_chunk, full_covariances_chunk, metrics, chunk_matrices):
        """Saves results to disk."""
        os.makedirs(save_dir, exist_ok=True)
    
        np.save(f"{static_dir}/partial_correlations_{n_chunks}_chunks.npy", partial_correlations_chunk)
        np.save(f"{static_dir}/full_covariances_{n_chunks}_chunks.npy", full_covariances_chunk)

        for key, _ in chunk_matrices.items():
            np.savez(f"{save_dir}/edge_prediction_all_{key}_chunks_{n_chunks}_features_used_actual_with_r2.npz", 
                    r2_accuracy_per_edge=metrics['r2'][key], 
                    accuracy_per_edge=metrics['accuracy'][key], 
                    partial_correlations_chunk=partial_correlations_chunk if 'icov' in key else full_covariances_chunk)