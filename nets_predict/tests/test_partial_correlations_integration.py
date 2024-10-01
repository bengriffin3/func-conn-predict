import numpy as np
import unittest
from osl_dynamics.data import Data
from nets_predict.classes.partial_correlation import PartialCorrelation, PartialCorrelationAnalysis

proj_dir = '/gpfs3/well/win-fmrib-analysis/users/psz102/osld_scripts/hcp_scripts/'

class TestPartialCorrelation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fc_matrix = np.random.rand(10, 10)
        cls.partial_corr = PartialCorrelation(cls.fc_matrix)
        # cls.partial_corr = PartialCorrelation()
        cls.partial_corr_analysis = PartialCorrelationAnalysis(cls.partial_corr)  # Pass the calculator instance
        cls.n_ICs = 50
        cls.n_chunks = 4
        cls.n_sub = 10
        cls.n_timepoints = 4800

        # find test data files
        with open(f"{proj_dir}/data_files_ICA{cls.n_ICs}_TEST.txt", "r") as file:
            inputs = file.read().split("\n")

        # Load test data
        cls.test_data = Data(inputs, load_memmaps=False, n_jobs=8)
        # Standardize data
        cls.test_data.standardize() 

    def test_load_data(self): 
        self.assertEqual(len(self.test_data.time_series()), self.n_sub)
        self.assertEqual(self.test_data.time_series()[0].shape[0], self.n_timepoints)
        self.assertEqual(self.test_data.time_series()[0].shape[1], self.n_ICs)
        # self.assertEqual(np.round(np.sum(self.test_data.time_series()[0]), 7), 0.0005531)

    # def test_get_ground_truth_matrix(self):
    #     n_session = 4

    #     # develop ground truth matrix using function
    #     ground_truth_matrix = self.partial_corr_analysis.get_ground_truth_matrix(self.test_data, n_session)[0]

    #     # this should give us the answer
    #     ground_truth_matrix_target_sum = 2125.58994885  # 8 DP

    #     self.assertAlmostEqual(np.sum(ground_truth_matrix), ground_truth_matrix_target_sum)

    # def test_split_time_series(self):
    #     # split time series using function
    #     time_series_split = self.partial_corr_analysis.split_time_series(self.test_data, self.n_chunks, n_sub=self.n_sub, n_ICs=self.n_ICs)

    #     # check time series is split correctly
    #     time_series_target = 0.00011231
         
    #     self.assertEqual(np.round(np.sum(time_series_split), 8), time_series_target)
    #     self.assertEqual(time_series_split.shape, (self.n_sub, self.n_chunks, self.n_timepoints // self.n_chunks, self.n_ICs))

    # def test_partial_correlation_chunk(self):
    #     time_series_chunk = self.partial_corr_analysis.split_time_series(self.test_data, self.n_chunks, n_sub=self.n_sub, n_ICs=self.n_ICs)  # creates array of size (n_sub x n_chunk x n_IC x n_IC)

    #     partial_correlations_chunk = self.partial_corr_analysis.get_partial_correlation_chunk(time_series_chunk, self.n_chunks)[0]

    #     partial_correlations_chunk_target = 8502.35979540
    
    #     self.assertEqual(np.round(np.sum(partial_correlations_chunk), 8), partial_correlations_chunk_target)
    #     self.assertEqual(partial_correlations_chunk.shape, (self.n_sub, self.n_chunks, self.n_ICs, self.n_ICs))

if __name__ == '__main__':
    unittest.main()

# import numpy as np
# import unittest
# from osl_dynamics.data import Data
# from nets_predict.classes.partial_correlation import PartialCorrelation, PartialCorrelationAnalysis, CovarianceUtils


# proj_dir = '/gpfs3/well/win-fmrib-analysis/users/psz102/osld_scripts/hcp_scripts/'

# class TestPartialCorrelation(unittest.TestCase):

#     @classmethod
#     def setUpClass(cls):
#         cls.partial_corr = PartialCorrelation()
#         cls.partial_corr_analysis = PartialCorrelationAnalysis()
#         cls.n_ICs = 50
#         cls.n_chunks = 4
#         cls.n_sub = 10
#         cls.n_timepoints = 4800

        
#         # find test data files
#         with open(f"{proj_dir}/data_files_ICA{cls.n_ICs}_TEST.txt", "r") as file:
#             inputs = file.read().split("\n")

#         # Load test data
#         cls.test_data = Data(inputs, load_memmaps=False, n_jobs=8)
#         # Standardize data
#         cls.test_data.standardize() 

#     def test_load_data(self): 
#         self.assertEqual(len(self.test_data.time_series()), self.n_sub)
#         self.assertEqual(self.test_data.time_series()[0].shape[0], self.n_timepoints)
#         self.assertEqual(self.test_data.time_series()[0].shape[1], self.n_ICs)
#         # self.assertEqual(np.round(np.sum(self.test_data.time_series()[0]), 7), 0.0005531)

#     def test_get_ground_truth_matrix(self):
#         n_session = 4

#         # develop ground truth matrix using function
#         ground_truth_matrix = self.partial_corr.get_ground_truth_matrix(self.test_data, n_session)[0]

#         # this should give us the answer
#         ground_truth_matrix_target_sum = 2125.58994885  # 8 DP

#         self.assertAlmostEqual(np.sum(ground_truth_matrix), ground_truth_matrix_target_sum)

#     def test_split_time_series(self):
#         # split time series using function
#         time_series_split = self.partial_corr_analysis.split_time_series(self.test_data, self.n_chunks, n_sub=self.n_sub, n_ICs=self.n_ICs)

#         # check time series is split correctly
#         time_series_target = 0.00011231
         
#         self.assertEqual(np.round(np.sum(time_series_split), 8), time_series_target)
#         self.assertEqual(time_series_split.shape, (self.n_sub, self.n_chunks, self.n_timepoints // self.n_chunks, self.n_ICs))

#     def test_partial_correlation_chunk(self):
#         time_series_chunk = self.partial_corr_analysis.split_time_series(self.test_data, self.n_chunks, n_sub=self.n_sub, n_ICs=self.n_ICs)  # creates array of size (n_sub x n_chunk x n_IC x n_IC)

#         partial_correlations_chunk = self.partial_corr.get_partial_correlation_chunk(time_series_chunk, self.n_chunks)[0]

#         partial_correlations_chunk_target = 8502.35979540
    
#         self.assertEqual(np.round(np.sum(partial_correlations_chunk), 8), partial_correlations_chunk_target)
#         self.assertEqual(partial_correlations_chunk.shape, (self.n_sub, self.n_chunks, self.n_ICs, self.n_ICs))


# if __name__ == '__main__':
#     unittest.main()