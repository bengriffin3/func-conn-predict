import numpy as np
import unittest
from osl_dynamics.data import Data
from nets_predict.classes.partial_correlation import PartialCorrelationClass

proj_dir = '/gpfs3/well/win-fmrib-analysis/users/psz102/osld_scripts/hcp_scripts/'

class TestPartialCorrelation(unittest.TestCase):

    @classmethod
    def setUp(self):
        self.partial_corr = PartialCorrelationClass()
        self.n_ICs = 50
        self.n_chunks = 4
        self.n_sub = 10
        self.n_timepoints = 4800


        # find test data files
        with open(f"{proj_dir}/data_files_ICA{self.n_ICs}_TEST.txt", "r") as file:
            inputs = file.read().split("\n")

        # Load test data
        data = Data(inputs, load_memmaps=False, n_jobs=8)
        # Standardize data
        data.standardize() 

        self.test_data = data

    def test_load_data(self): 
        self.assertEqual(len(self.test_data.time_series()), self.n_sub)
        self.assertEqual(self.test_data.time_series()[0].shape[0], self.n_timepoints)
        self.assertEqual(self.test_data.time_series()[0].shape[1], self.n_ICs)
        #self.assertEqual(np.round(np.sum(self.test_data.time_series()[0]), 7), 0.0005531)

    def test_get_ground_truth_matrix(self):

        n_session = 4

        # develop ground truth matrix using function
        ground_truth_matrix = self.partial_corr.get_ground_truth_matrix(self.test_data, n_session)[0]

        # this should give us the answer
        ground_truth_matrix_target_sum = 2125.58994885 # 8 DP

        self.assertAlmostEqual(np.sum(ground_truth_matrix),ground_truth_matrix_target_sum)

    def test_split_time_series(self):

        # split time series using function
        time_series_split = self.partial_corr.split_time_series(self.test_data, self.n_chunks, n_sub=None, n_ICs=None)

        # check time series is split correctly
        time_series_target = 0.00011231
         
        self.assertEqual(np.round(np.sum(time_series_split), 8), time_series_target)
        self.assertEqual(time_series_split.shape, (self.n_sub, self.n_chunks, self.n_timepoints/self.n_chunks, self.n_ICs))

    def test_partial_correlation_chunk(self):


        time_series_chunk =  self.partial_corr.split_time_series(self.test_data, self.n_chunks, self.n_sub, self.n_ICs) # creates array of size (n_sub x n_chunk x n_IC x n_IC)

        partial_correlations_chunk = self.partial_corr.get_partial_correlation_chunk(time_series_chunk, self.n_chunks)[0]

        partial_correlations_chunk_target = 8502.35979540
    
        self.assertEqual(np.round(np.sum(partial_correlations_chunk), 8), partial_correlations_chunk_target)
        self.assertEqual(partial_correlations_chunk.shape, (self.n_sub, self.n_chunks, self.n_ICs, self.n_ICs))
 

if __name__ == '__main__':
    unittest.main()