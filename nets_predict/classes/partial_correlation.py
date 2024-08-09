import numpy as np
from tqdm import trange
import logging

_logger = logging.getLogger("Chunk_project")

class PartialCorrelationClass:
    """Class for running analysis function for project 2.

    Parameters
    ----------
    Xxx : Xxx
        Xxx.
    """

    def partial_corr(self, Sigma, rho=0.1, do_rtoz=-18.8310):
        """Calculate partial correlation matrix (using formula based on precision matrix from Wikipedia.

        Parameters
        ----------
        Sigma : np.ndarray
            Covariance matrix. Shape is (n_ICs, n_ICs).
        rho : float, optional
            Regularization. 0.1 used as default.
        do_rtoz : float, optional.
            Scaling of Fisher transformation

        Returns
        -------
        partial_corr : np.ndarray
            Partial correlation matrix. Shape is (n_ICs, n_ICs). 
        partial_corr_r2z : np.ndarray
            Parital correlation matrix after applying Fisher transformation (i.e. inverse of covariance matrix). Shape is (n_ICs, n_ICs).       
        precision : np.ndarray
            Precision matrix (i.e. inverse of covariance matrix). Shape is (n_ICs, n_ICs).
        """

        # Normalize matrix in case this hasn't been done         
        Sigma = Sigma / np.sqrt(np.mean(np.diag(Sigma) ** 2))
        # Note: this only makes a difference if you want the precision matrix
        # if you want partial correlations, you normalize at the end anyway

        # get precision matrix from covariance matrix
        precision = self.covariance_to_precision(Sigma, rho)

        # calculate normalization (p_XX and p_YY) i.e., denominator of Wiki formula
        pii = np.sqrt(np.abs(np.diag(precision)))
        pii_rep = np.stack([pii for _ in range(len(Sigma))], axis=1) # p_XX
        pjj_rep = np.stack([pii for _ in range(len(Sigma))], axis=0) # p_YY

        # now apply Wikipedia formula
        partial_corr = ((- precision / pii_rep) / pjj_rep)

        # diagonal elements would all have -1 (1 because it's pii / sqrt(pii*pjj) and negative because the formula is commonly given with negative)
        # so let's set them to 0 
        np.fill_diagonal(partial_corr, 0) # set diagonal elements to 0 
        
        partial_corr_r2z = self.fisher_transform_scaled(partial_corr, do_rtoz)

        return partial_corr, partial_corr_r2z, precision


    def covariance_to_precision(self, Sigma, rho=0.1):
        """Calculate precision matrix from covariance matrix using Cholesky decomposition.

        Parameters
        ----------
        Sigma : np.ndarray
            Covariance matrix. Shape is (n_ICs, n_ICs).
        rho : float, optional
            Regularization. 0.1 used as default.

        Returns
        -------
        precision : np.ndarray
            Precision matrix (i.e. inverse of covariance matrix). Shape is (n_ICs, n_ICs).
        """
        
        # Add regularization (for inversion)
        np.fill_diagonal(Sigma, Sigma.diagonal() + rho)
        
        # use cholesky decomposition to find inverse covariance matrix (i.e., precision matrix)
        R = np.linalg.cholesky(Sigma)
        R_inv = np.linalg.inv(R)
        precision = np.transpose(R_inv) @ R_inv

        return precision


    def fisher_transform_scaled(self, partial_corr, do_rtoz=-18.8310):
        """Calculate Fisher transformation and scale z-scores.

        Parameters
        ----------
        partial_corr : np.ndarray
            Partial correlation matrix. Shape is (n_ICs, n_ICs).
        do_rtoz : float, optional.
            Scaling of Fisher transformation.

        Returns
        -------
        partial_corr_r2z_sclaed : np.ndarray
            Partial correlation matrix after applying Fisher transformation (i.e. inverse of covariance matrix) and being sclaed. Shape is (n_ICs, n_ICs).         
        """
        # Apply Fisher transformation (formula for atanh)
        partial_corr_r2z = 0.5 * np.log((1 + partial_corr) / (1 - partial_corr)) 
        
        # Scale z-sores
        partial_corr_r2z_scaled = partial_corr_r2z * (-do_rtoz)

        return partial_corr_r2z_scaled
    

    #def form_feat_mat(hmm_covariances):
    def extract_upper_off_main_diag(self, hmm_covariances):
        """Extract upper diagonal from HMM covariances.

        Parameters
        ----------
        hmm_covariances : np.ndarray
            Covariance matrices from HMM states. Shape is (n_states, n_ICs, n_ICs).

        Returns
        -------
        hmm_covariances_diag : np.ndarray
            Upper diagonal parts of covariances matrix. Shape is (n_states, (n_ICs * (n_ICs + 1))/2)
        """  

        hmm_covariances = np.squeeze(hmm_covariances)
        m, n = np.triu_indices(hmm_covariances.shape[-1],1)
        hmm_covariances_diag = hmm_covariances[..., m, n]

        return hmm_covariances_diag
    

    def get_ground_truth_matrix(self, data, n_session):
         
        # Initialise arrays
        n_sub = len(data.time_series())
        n_ICs = data.n_channels
        partial_correlation_session  = np.zeros([n_sub, n_session, n_ICs, n_ICs])
        full_covariances_session  = np.zeros([n_sub, n_session, n_ICs, n_ICs])

        

        # Determine netmats per subject per session
        _logger.info("Generating netmats")
        for sub in trange(n_sub, desc='Subjects'):
            # For given subject, split time series of length 4800 into 4 sessions of 1200 time points (produces a list of length 4 with an array of 1200 x 25 in each)
            time_series_session = np.split(data.time_series()[sub], n_session)

            # Calculate FC matrix then the corresponding PC matrix per subject per session
            for sess in range(n_session):
                full_covariances_session[sub,sess,:,:] = np.cov(time_series_session[sess], rowvar=False) # rowvar=False to get n_IC x n_IC
                partial_correlation_session[sub,sess,:,:] = self.partial_corr(full_covariances_session[sub,sess,:,:])[1] # get second output so we have applied r2z transformation

        ground_truth_matrix_partial = np.squeeze(np.mean(partial_correlation_session, axis=1))
        ground_truth_matrix_full = np.squeeze(np.mean(full_covariances_session, axis=1))


        return ground_truth_matrix_partial, ground_truth_matrix_full 
    
    def get_partial_correlation_chunk(self, time_series_chunk, n_chunks):
         
        # Initialise array
        n_sub = time_series_chunk.shape[0]
        n_ICs = time_series_chunk.shape[3]
        partial_correlations_chunk = np.zeros((n_sub, n_chunks, n_ICs, n_ICs))   
        full_covariances_chunk = np.zeros((n_sub, n_chunks, n_ICs, n_ICs))   

        # Determine netmats per subject per session
        _logger.info("Generating partial correlations per subject")
        for sub in trange(n_sub, desc='Subjects'):
            for chunk in range(n_chunks):
                # determine full covariances (for each chunk)
                full_covariances_chunk[sub,chunk,:,:] = np.cov(time_series_chunk[sub,chunk,:,:], rowvar=False) # rowvar=False to get n_IC x n_IC

                # calculate partial correlations (for each chunk)
                partial_correlations_chunk[sub,chunk,:,:] = self.partial_corr(full_covariances_chunk[sub,chunk,:,:])[1] # get second output so we have applied R2Z transformation
        
        return partial_correlations_chunk, full_covariances_chunk

    def split_time_series(self, data, n_chunks, n_sub=None, n_ICs=None):

        if n_sub is None:
            n_sub = len(data.time_series())
        else:
            if n_sub != len(data.time_series()):
                raise ValueError("Incorrect number of subjects")

        if n_ICs is None:
            n_ICs = data.n_channels
        else:
            if n_ICs != data.n_channels:
                raise ValueError("Incorrect number of channels selected")

        n_timepoints = data.time_series()[0].shape[0]

        time_series_chunk = np.zeros((n_sub, n_chunks, int(n_timepoints/n_chunks),n_ICs))

        # split data into chunks
        for sub in range(n_sub):
            # get data for 1 subject and split it into 'n' chunks
            time_series_sub = np.split(data.time_series()[sub],n_chunks,axis=0)

            # for given subject, store the chunked up data
            for chunk in range(n_chunks):
                time_series_chunk[sub,chunk,:,:] = time_series_sub[chunk]

        return time_series_chunk
    
    def find_original_indices(self, edge_index, netmats):
        # Get upper off-diagonal indices
        upper_off_diag_indices = np.triu_indices(netmats.shape[-1], 1)
        row = upper_off_diag_indices[0][edge_index]
        col = upper_off_diag_indices[1][edge_index]
        return row, col
    

    def remove_bad_components(self, data):

        n_sub = len(data.time_series())
        n_ICs = data.time_series()[0].shape[1]

        # I only have the bad components for UKB but this project is for HCP so I've removed this for now
        if n_ICs == 25:
            good_components = list(range(1, 26))# keep all, the below is for UKB
            #good_components = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        elif n_ICs == 100:
            good_components = list(range(1, 101))
            #good_components = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 48, 49, 50, 52, 53, 57, 58, 60, 63, 64, 93]
            

        # Adjust good_components for 0-based indexing
        good_components_indices = [i - 1 for i in good_components]

        # Note that we also remove the first 8 time points since this is often where [errors?] occur (Ask Janus)
        session_length = 1200
        num_sessions = 4
        time_point_indices = np.concatenate([np.arange(i * session_length + 8, (i + 1) * session_length) for i in range(num_sessions)])
        
        # Iterate over each subject's time series and select the good components and good time points
        for i in range(n_sub):
            data.time_series()[i] = data.time_series()[i][time_point_indices, good_components_indices]

        return data