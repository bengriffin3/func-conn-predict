import numpy as np
import unittest
from unittest.mock import patch, MagicMock
from nets_predict.classes.hmm import HMMInference, TimeSeriesProcessing, FeatureEngineering, HMMFeatures, Prediction

class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        self.fe = FeatureEngineering()

    # Test for intialise_trans_prob
    def test_intialise_trans_prob(self):
        trans_prob_diag = 10
        n_states = 3
        expected_trans_prob = np.array([
            [0.8333333, 0.0833333, 0.0833333],
            [0.0833333, 0.8333333, 0.0833333],
            [0.0833333, 0.0833333, 0.8333333]
        ])
        result = self.fe.initialise_trans_prob(trans_prob_diag, n_states)
        np.testing.assert_almost_equal(result, expected_trans_prob)

    # Test for reshape_summary_stats
    def test_reshape_summary_stats(self):
        n_sub = 5
        fo = np.random.rand(n_sub, 3)
        intv = np.random.rand(n_sub, 3)
        lt = np.random.rand(n_sub, 3)
        sr = np.random.rand(n_sub, 3)
        
        result = self.fe.reshape_summary_stats(fo, intv, lt, sr)
        expected_shape = (n_sub, fo.size // n_sub * 4)  # Concatenating 4 reshaped stats
        self.assertEqual(result.shape, expected_shape)

    # Test for reshape_dynamic_features
    @patch('nets_predict.classes.partial_correlation.PartialCorrelation')
    def test_reshape_dynamic_features_fc(self, mock_partial_correlation):
        hmm_features_dict = {
            'icovs_chunk': np.random.rand(10, 25, 25),
            'covs_chunk': np.random.rand(10, 25, 25),
            'means_chunk': np.random.rand(10, 25),
            'trans_prob_chunk': np.random.rand(10, 25, 25),
            'fo_chunk': np.random.rand(10, 3),
            'intv_chunk': np.random.rand(10, 3),
            'lt_chunk': np.random.rand(10, 3),
            'sr_chunk': np.random.rand(10, 3),
            'weighted_covs_chunk': np.random.rand(10, 25, 25),
            'weighted_icovs_chunk': np.random.rand(10, 25, 25)
        }
        dynamic_add = 'fc'
        result = self.fe.reshape_dynamic_features(hmm_features_dict, dynamic_add)
        expected_shape = (10, 325)  # Upper triangle from a 25x25 matrix
        self.assertEqual(result.shape, expected_shape)

    # Test for reshape_icovs
    @patch('nets_predict.classes.partial_correlation.PartialCorrelation')
    def test_reshape_icovs(self, mock_partial_correlation):
        icovs = np.random.rand(10, 25, 25)
        mock_partial_correlation.return_value.extract_upper_off_main_diag.return_value = np.random.rand(10, 300)

        result = self.fe.reshape_icovs(icovs)
        self.assertEqual(result.shape, (10, 300))

    # Test for concatenate_all
    @patch('nets_predict.classes.partial_correlation.PartialCorrelation')
    def test_concatenate_all(self, mock_partial_correlation):
        icovs = np.random.rand(10, 25, 25)
        covs = np.random.rand(10, 25, 25)
        means = np.random.rand(10, 25)
        trans_prob = np.random.rand(10, 5, 5)
        fo = np.random.rand(10, 3)
        intv = np.random.rand(10, 3)
        lt = np.random.rand(10, 3)
        sr = np.random.rand(10, 3)

        mock_partial_correlation.return_value.extract_upper_off_main_diag.return_value = np.random.rand(10, 300)

        result = self.fe.concatenate_all(icovs, covs, means, trans_prob, fo, intv, lt, sr)
        expected_shape = (10, 300 + 325 + 25 + 25 + fo.size // 10 * 4)
        self.assertEqual(result.shape, expected_shape)

    # Test for reshape_cov_features
    def test_reshape_cov_features(self):
        netmats = np.random.rand(10, 25, 25)
        result = self.fe.reshape_cov_features(netmats)
        expected_shape = (10, 325)  # Upper triangle of 25x25 matrix
        self.assertEqual(result.shape, expected_shape)


    def test_reshape_cov_features(self):
        # Create a random netmats input (e.g., 10 subjects, 4x4 matrices)
        netmats = np.random.rand(10, 4, 4)

        # Run the function
        result = self.fe.reshape_cov_features(netmats)

        # Expected shape: 10 subjects, and each 4x4 matrix is flattened into its upper triangular part
        # For a 4x4 matrix, there are 10 upper triangular elements (including diagonal)
        expected_shape = (10, 10)  # 10 subjects, 10 features (upper triangular elements)

        # Check the shape
        self.assertEqual(result.shape, expected_shape)

        # Check that the correct elements are extracted (upper triangular)
        m, n = np.triu_indices(4)
        expected_elements = netmats[0, m, n]
        self.assertTrue(np.allclose(result[0], expected_elements))


    def test_determine_n_features_fc(self):
        n_ICs = 4
        n_states = 3
        features_to_use = 'fc'

        # Run the function
        result = self.fe.determine_n_features(features_to_use, n_ICs, n_states)

        # For 'fc', we calculate the number of upper diagonal elements including the diagonal:
        # n_upper_diag = (n_ICs * (n_ICs + 1)) // 2 = (4 * (4 + 1)) // 2 = 10
        # Total features = n_states * n_upper_diag = 3 * 10 = 30
        expected_result = 30

        self.assertEqual(result, expected_result)

    def test_determine_n_features_means(self):
        n_ICs = 4
        n_states = 3
        features_to_use = 'means'

        # Run the function
        result = self.fe.determine_n_features(features_to_use, n_ICs, n_states)

        # For 'means', the result should be n_states * n_ICs = 3 * 4 = 12
        expected_result = 12

        self.assertEqual(result, expected_result)

    def test_determine_n_features_tpms_ss(self):
        n_ICs = 4
        n_states = 3
        features_to_use = 'tpms_ss'

        # Run the function
        result = self.fe.determine_n_features(features_to_use, n_ICs, n_states)

        # For 'tpms_ss', we calculate the size:
        # n_summary_stats = n_states * 4 = 3 * 4 = 12
        # n_tpms = n_states * n_states = 3 * 3 = 9
        # Total features = n_summary_stats + n_tpms = 12 + 9 = 21
        expected_result = 21

        self.assertEqual(result, expected_result)

    def test_determine_n_features_static(self):
        n_ICs = 4
        features_to_use = 'static'

        # Run the function
        result = self.fe.determine_n_features(features_to_use, n_ICs)

        # For 'static', the result should be the upper off-diagonal elements:
        # n_upper_diag = (n_ICs * (n_ICs - 1)) // 2 = (4 * (4 - 1)) // 2 = 6
        expected_result = 6

        self.assertEqual(result, expected_result)

    def test_determine_n_features_error(self):
        n_ICs = 4
        features_to_use = 'fc'

        # Test case when n_states is 0 and 'static' is not in features_to_use (should raise ValueError)
        with self.assertRaises(ValueError):
            self.fe.determine_n_features(features_to_use, n_ICs, n_states=0)

    def test_calculate_feature_size_with_diag(self):
        n_ICs = 4
        n_states = 3
        include_diag = True

        # Run the function
        result = self.fe._calculate_feature_size(n_ICs, n_states, include_diag)

        # n_upper_diag = (n_ICs * (n_ICs + 1)) // 2 = (4 * (4 + 1)) // 2 = 10
        # Total features = n_states * n_upper_diag = 3 * 10 = 30
        expected_result = 30

        self.assertEqual(result, expected_result)

    def test_calculate_feature_size_without_diag(self):
        n_ICs = 4
        n_states = 3
        include_diag = False

        # Run the function
        result = self.fe._calculate_feature_size(n_ICs, n_states, include_diag)

        # n_upper_diag = (n_ICs * (n_ICs - 1)) // 2 = (4 * (4 - 1)) // 2 = 6
        # Total features = n_states * n_upper_diag = 3 * 6 = 18
        expected_result = 18

        self.assertEqual(result, expected_result)


    def test_tpms_summary_stats_size(self):
        n_states = 3

        # Run the function
        result = self.fe._tpms_summary_stats_size(n_ICs=4, n_states=n_states)

        # n_summary_stats = n_states * 4 = 3 * 4 = 12
        # n_tpms = n_states * n_states = 3 * 3 = 9
        # Total features = 12 + 9 = 21
        expected_result = 21

        self.assertEqual(result, expected_result)

    def test_static_feature_size(self):
        n_ICs = 4

        # Run the function
        result = self.fe._static_feature_size(n_ICs)

        # n_upper_diag = (n_ICs * (n_ICs - 1)) // 2 = (4 * (4 - 1)) // 2 = 6
        expected_result = 6

        self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()