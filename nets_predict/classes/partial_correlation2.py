import numpy as np
from tqdm import trange
import logging

_logger = logging.getLogger("Chunk_project")

class CovarianceUtils:
    """Utility class for covariance and precision calculations."""

    @staticmethod
    def covariance_to_precision(Sigma, rho=0.1):
        """Calculate precision matrix from covariance matrix using Cholesky decomposition."""
        np.fill_diagonal(Sigma, Sigma.diagonal() + rho)
        R = np.linalg.cholesky(Sigma)
        R_inv = np.linalg.inv(R)
        return np.transpose(R_inv) @ R_inv

    @staticmethod
    def fisher_transform(partial_corr, scaling_factor=-18.8310):
        """Apply Fisher transformation and scale z-scores."""
        partial_corr_r2z = 0.5 * np.log((1 + partial_corr) / (1 - partial_corr))
        return partial_corr_r2z * (-scaling_factor)

class PartialCorrelation:
    """Class for running partial correlation analysis."""

    def __init__(self, rho=0.1, scaling_factor=-18.8310):
        self.rho = rho
        self.scaling_factor = scaling_factor
        self.utils = CovarianceUtils()

    def partial_corr(self, Sigma):
        """Calculate partial correlation matrix from covariance matrix."""
        Sigma = Sigma / np.sqrt(np.mean(np.diag(Sigma) ** 2))  # Normalize
        precision = self.utils.covariance_to_precision(Sigma, self.rho)

        pii = np.sqrt(np.abs(np.diag(precision)))
        pii_rep = np.stack([pii] * len(Sigma), axis=1)
        pjj_rep = np.stack([pii] * len(Sigma), axis=0)

        partial_corr = (-precision / pii_rep) / pjj_rep
        np.fill_diagonal(partial_corr, 0)

        partial_corr_r2z = self.utils.fisher_transform(partial_corr, self.scaling_factor)
        return partial_corr, partial_corr_r2z, precision
   



class PartialCorrelationAnalysis:
    """Class for analyzing partial correlations from time series data."""

    def __init__(self, partial_correlation_calculator):
        self.partial_correlation_calculator = partial_correlation_calculator

    def extract_upper_off_main_diag(self, hmm_covariances):
        hmm_covariances = np.squeeze(hmm_covariances)
        m, n = np.triu_indices(hmm_covariances.shape[-1], k=1)  # Use k=1 for upper diagonal
        return hmm_covariances[..., m, n]

    def get_ground_truth_matrix(self, data, n_session):
        n_sub = len(data.time_series())
        n_ICs = data.n_channels
        
        # Initialize arrays
        partial_correlation_session = np.zeros((n_sub, n_session, n_ICs, n_ICs))
        full_covariances_session = np.zeros((n_sub, n_session, n_ICs, n_ICs))

        _logger.info("Generating netmats")
        for sub in trange(n_sub, desc='Subjects'):
            time_series_session = np.split(data.time_series()[sub], n_session)

            for sess in range(n_session):
                full_covariances_session[sub, sess] = np.cov(time_series_session[sess], rowvar=False)
                partial_correlation_session[sub, sess] = self.partial_correlation_calculator.partial_corr(full_covariances_session[sub, sess])[1]

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
                full_covariances_chunk[sub, chunk] = np.cov(time_series_chunk[sub, chunk], rowvar=False)
                partial_correlations_chunk[sub, chunk] = self.partial_correlation_calculator.partial_corr(full_covariances_chunk[sub, chunk])[1]

        return partial_correlations_chunk, full_covariances_chunk

    def split_time_series(self, data, n_chunks, n_sub=None, n_ICs=None):
        n_sub = n_sub or len(data.time_series())
        if n_sub != len(data.time_series()):
            raise ValueError("Incorrect number of subjects")

        n_ICs = n_ICs or data.n_channels
        if n_ICs != data.n_channels:
            raise ValueError("Incorrect number of channels selected")

        n_timepoints = data.time_series()[0].shape[0]
        time_series_chunk = np.zeros((n_sub, n_chunks, n_timepoints // n_chunks, n_ICs))

        for sub in range(n_sub):
            time_series_sub = np.array_split(data.time_series()[sub], n_chunks, axis=0)
            time_series_chunk[sub] = time_series_sub

        return time_series_chunk

    def find_original_indices(self, edge_index, netmats):
        row, col = np.triu_indices(netmats.shape[-1], 1)
        return row[edge_index], col[edge_index]

    def remove_bad_components(self, data):
        n_sub = len(data.time_series())
        n_ICs = data.time_series()[0].shape[1]

        if n_ICs not in {25, 100}:
            raise ValueError(f"Unexpected number of independent components: {n_ICs}")

        good_components_indices = list(range(n_ICs))
        session_length = 1200
        num_sessions = 4
        time_point_indices = np.concatenate(
            [np.arange(i * session_length + 8, (i + 1) * session_length) for i in range(num_sessions)]
        )

        for i in range(n_sub):
            data.time_series()[i] = data.time_series()[i][time_point_indices, good_components_indices]

        return data