import numpy as np
import unittest
import tempfile
import os
from unittest.mock import patch
from func_conn_predict.classes.partial_correlation import PartialCorrelation, PartialCorrelationAnalysis, CovarianceUtils, GroundTruthPreparation, PartialCorrelationCalculation

class TestCovarianceUtils(unittest.TestCase):

    def test_fisher_transform(self):
        # Create a sample partial correlation matrix
        partial_corr = np.array([[1.0, 0.2], [0.2, 1.0]])

        # Run the Fisher transformation
        fisher_transformed = CovarianceUtils.fisher_transform(partial_corr)

        # Check that the diagonal is still 0 after transformation (for z-scores)
        self.assertTrue(np.allclose(np.diag(fisher_transformed), [0, 0]))

        # Check if transformation of off-diagonal elements is applied correctly
        # Calculate expected values
        expected_fisher = 0.5 * np.log((1 + partial_corr[0, 1]) / (1 - partial_corr[0, 1])) * -18.8310
        self.assertAlmostEqual(fisher_transformed[0, 1], -expected_fisher, places=5)

class TestPartialCorrelation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fc_matrix = np.random.rand(10, 10) 
        cls.partial_corr = PartialCorrelation(cls.fc_matrix)
        cls.partial_corr_analysis = PartialCorrelationAnalysis(cls.partial_corr) 
        cls.covariance_utils = CovarianceUtils()

    def test_matrix_inversion(self):
        # matrix to invert
        m1 = np.array([[0.5, 1.5],[1.5, 4.5]])

        # invert using function
        m1_inv = np.round(self.covariance_utils.covariance_to_precision(m1, rho=0.1), 8)

        # ideal inversion
        m2 = np.array([[-9.01960784, 2.94117647], [2.94117647, -1.17647059]]) 

        np.testing.assert_array_almost_equal(m1_inv, m2, decimal=8)

    def test_fisher_transform_calc(self):
        Sigma_partial_corr = np.array([[0, 0.75441865], [0.75441865, 0]])
        Sigma_partial_corr_r2z = np.array([[0, 18.51336406], [18.51336406, 0]])

        Fisher_transform_pc = np.round(self.partial_corr.utils.fisher_transform(Sigma_partial_corr), 8)

        np.testing.assert_array_almost_equal(Fisher_transform_pc, Sigma_partial_corr_r2z, decimal=8)


    def test_matrix_inversion(self):

        # matrix to invert
        m1 = np.array([[0.5, 1.5],[1.5, 4.5]])

        # invert using function
        m1_inv = np.round(self.covariance_utils.covariance_to_precision(m1, rho=0.1), 8)

        # ideal inversion
        m2 = np.array([[ 9.01960784, -2.94117647], [-2.94117647,  1.17647059]]) 

        self.assertAlmostEqual(m1_inv.tolist(), m2.tolist())


class TestPartialCorrelation(unittest.TestCase):

    def setUp(self):
        # Create a sample functional connectivity (fc) matrix
        self.fc_matrix = np.array([[4, 2, 0.6], [2, 5, 0.9], [0.6, 0.9, 3]])

        # Instantiate the PartialCorrelation object
        self.partial_corr_obj = PartialCorrelation(self.fc_matrix, rho=0.1, scaling_factor=-18.8310)

    def test_partial_corr(self):
        # Run the partial correlation method
        partial_corr, partial_corr_r2z, precision = self.partial_corr_obj.partial_corr()

        # Check that the precision matrix is symmetric
        self.assertTrue(np.allclose(precision, precision.T))

        # Check that the diagonal of partial_corr is 0
        self.assertTrue(np.allclose(np.diag(partial_corr), [0, 0, 0]))

        # Check the transformation on off-diagonal elements
        expected_fisher = 0.5 * np.log((1 + partial_corr[0, 1]) / (1 - partial_corr[0, 1])) * -18.8310
        self.assertAlmostEqual(abs(partial_corr_r2z[0, 1]), abs(expected_fisher), places=5)
    
    def test_extract_upper_off_main_diag(self):
        # Run the extraction of the upper diagonal
        upper_off_diag = self.partial_corr_obj.extract_upper_off_main_diag()

        # Check the length of the result (for 3x3 matrix, upper diag has 3 elements)
        self.assertEqual(len(upper_off_diag), 3)

        # Ensure the result matches expected upper off-diagonal values
        expected_upper_diag = np.array([2, 0.6, 0.9])
        np.testing.assert_array_equal(upper_off_diag, expected_upper_diag)

    def test_find_original_indices(self):
        # Run the method to find original indices for an edge index
        edge_index = 1  # Second element in upper diagonal
        row, col = self.partial_corr_obj.find_original_indices(edge_index)

        # Check that the row, col corresponds to the second element in the upper triangle
        self.assertEqual(row, 0)
        self.assertEqual(col, 2)

class TestPartialCorrelationAnalysis(unittest.TestCase):

    def setUp(self):
        # Create dummy time series data
        self.n_sub = 5  # Number of subjects
        self.n_session = 3  # Number of sessions
        self.n_ICs = 4  # Number of Independent Components
        self.time_series = np.random.rand(self.n_sub, self.n_session * 10, self.n_ICs)  # Simulate random time series data

        self.analysis = PartialCorrelationAnalysis(self.time_series)

    def test_get_partial_correlation_chunk(self):
        time_series_chunk = np.random.rand(self.n_sub, self.n_session, 10, self.n_ICs)  # Random chunks
        partial_corr_chunk, full_cov_chunk = self.analysis.get_partial_correlation_chunk(time_series_chunk, self.n_session)

        # Test shapes
        self.assertEqual(partial_corr_chunk.shape, (self.n_sub, self.n_session, self.n_ICs, self.n_ICs))
        self.assertEqual(full_cov_chunk.shape, (self.n_sub, self.n_session, self.n_ICs, self.n_ICs))

        # Test for non-negativity of the full covariance matrix
        self.assertTrue(np.all(np.linalg.eigvals(full_cov_chunk) >= 0))

class TestGroundTruthPreparation(unittest.TestCase):

    def test_flatten_matrices(self):
        # Prepare mock matrices
        ground_truth_matrix_partial = np.random.rand(5, 5)
        ground_truth_matrix_full = np.random.rand(5, 5)

        # Create an instance of the class
        gtp = GroundTruthPreparation()

        # Call the method
        flattened_partial, flattened_full = gtp.flatten_matrices(ground_truth_matrix_partial, ground_truth_matrix_full)

        # Check shapes of the flattened matrices
        self.assertEqual(flattened_partial.shape, (10,))  # Example expected shape for upper triangular flattening
        self.assertEqual(flattened_full.shape, (10,))  # Same as above
