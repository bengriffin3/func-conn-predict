import numpy as np
import unittest
from unittest.mock import patch, MagicMock
from nets_predict.classes.hmm import HMMInference, TimeSeriesProcessing, FeatureEngineering, HMMFeatures, Prediction
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr


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

class TestPredictionClass(unittest.TestCase):

    def setUp(self):
        self.prediction = Prediction()

    def test_standardise_train_apply_to_test(self):
        X_train = np.random.rand(100, 10)
        X_test = np.random.rand(20, 10)
        X_train_std, X_test_std = self.prediction.standardise_train_apply_to_test(X_train, X_test)
        
        # Check if the result has the same shape
        self.assertEqual(X_train_std.shape, X_train.shape)
        self.assertEqual(X_test_std.shape, X_test.shape)
        
        # Check if the mean of the training data is 0 and std deviation is 1
        np.testing.assert_almost_equal(np.mean(X_train_std, axis=0), np.zeros(10), decimal=5)
        np.testing.assert_almost_equal(np.std(X_train_std, axis=0), np.ones(10), decimal=5)

    def test_evaluate_model(self):
        model = MagicMock()
        X_test = np.random.rand(50, 10)
        y_test = np.random.rand(50)
        model.predict.return_value = np.random.rand(50)
        
        y_pred, corr = self.prediction.evaluate_model(model, X_test, y_test, 0)
        
        # Check if the predicted values match the model's predictions
        np.testing.assert_array_equal(y_pred, model.predict(X_test))
        
        # Check if the correlation is calculated
        expected_corr = pearsonr(y_test.squeeze(), y_pred.squeeze())[0]
        self.assertAlmostEqual(corr, expected_corr)

    def test_pca_dynamic_only(self):
        X_train = np.random.rand(100, 50)
        X_test = np.random.rand(20, 50)
        n_ICs = 5

        # Use the PCA transformation
        X_train_combined, X_test_combined, pca_model = self.prediction.pca_dynamic_only(X_train, X_test, n_ICs)
        
        # Check the combined shape
        self.assertEqual(X_train_combined.shape[0], X_train.shape[0])
        self.assertEqual(X_test_combined.shape[0], X_test.shape[0])

        # Check if PCA model was fitted correctly
        self.assertTrue(hasattr(pca_model, 'components_'))

    def test_self_predict_plus_pca(self):
        X = np.random.rand(100, 10)
        edge_index = 3
        
        # Run PCA and preserve the self edge
        X_combined = self.prediction.self_predict_plus_pca(X, edge_index)
        
        # Check if the self edge is still preserved
        self.assertTrue(np.all(X_combined[:, 0] == X[:, edge_index]))

        # Check if the number of features matches after PCA transformation
        self.assertEqual(X_combined.shape[0], X.shape[0])

    def test_get_predictor_features_static(self):
        netmats = np.random.rand(100, 10, 10)
        hmm_features_dict = {}
        edge_index = 0
        
        partial_corr_mock = MagicMock()
        PartialCorrelation = MagicMock(return_value=partial_corr_mock)
        partial_corr_mock.extract_upper_off_main_diag.return_value = np.random.rand(100, 45)

        X = self.prediction.get_predictor_features(netmats, hmm_features_dict, 'static', edge_index)
        
        # Check if static features were extracted correctly
        self.assertEqual(X.shape[0], 100)

    def test_get_predictor_features_static_self_edge(self):
        netmats = np.random.rand(100, 10, 10)
        hmm_features_dict = {}
        edge_index = 3
        
        partial_corr_mock = MagicMock()
        PartialCorrelation = MagicMock(return_value=partial_corr_mock)
        partial_corr_mock.extract_upper_off_main_diag.return_value = np.random.rand(100, 45)

        X = self.prediction.get_predictor_features(netmats, hmm_features_dict, 'static_self_edge_only', edge_index)
        
        # Check if the correct shape of predictor matrix was returned
        self.assertEqual(X.shape[0], 100)
        self.assertEqual(X.shape[1], 45)

class TestHMMFeatures(unittest.TestCase):

    def setUp(self):
        """Set up the HMMFeatures instance for testing."""
        self.hmm_features = HMMFeatures()

    def test_organise_hmm_features_across_chunks(self):
        # Simulate hmm_features_list with 2 chunks, 3 subjects, 4 states, and 2 ICs
        hmm_features_list = [
            {
                'fo_chunk': np.random.rand(3, 4),
                'lt_chunk': np.random.rand(3, 4),
                'sr_chunk': np.random.rand(3, 4),
                'intv_chunk': np.random.rand(3, 4),
                'means_chunk': np.random.rand(3, 4, 2),
                'covs_chunk': np.random.rand(3, 4, 2, 2),
                'trans_prob_chunk': np.random.rand(3, 4, 4),
                'icovs_chunk': np.random.rand(3, 4, 2, 2)
            },
            {
                'fo_chunk': np.random.rand(3, 4),
                'lt_chunk': np.random.rand(3, 4),
                'sr_chunk': np.random.rand(3, 4),
                'intv_chunk': np.random.rand(3, 4),
                'means_chunk': np.random.rand(3, 4, 2),
                'covs_chunk': np.random.rand(3, 4, 2, 2),
                'trans_prob_chunk': np.random.rand(3, 4, 4),
                'icovs_chunk': np.random.rand(3, 4, 2, 2)
            }
        ]

        # Run the function
        result = self.hmm_features.organise_hmm_features_across_chunks(hmm_features_list)

        # Expected shapes for each feature
        expected_shapes = {
            'fo': (2, 3, 4),
            'lt': (2, 3, 4),
            'sr': (2, 3, 4),
            'intv': (2, 3, 4),
            'means': (2, 3, 4, 2),
            'covs': (2, 3, 4, 2, 2),
            'trans_prob': (2, 3, 4, 4),
            'icovs': (2, 3, 4, 2, 2)
        }

        # Unpack the result to match the feature names
        fo, lt, sr, intv, means, covs, trans_prob, icovs = result

        # Check the shapes of each feature
        self.assertEqual(fo.shape, expected_shapes['fo'])
        self.assertEqual(lt.shape, expected_shapes['lt'])
        self.assertEqual(sr.shape, expected_shapes['sr'])
        self.assertEqual(intv.shape, expected_shapes['intv'])
        self.assertEqual(means.shape, expected_shapes['means'])
        self.assertEqual(covs.shape, expected_shapes['covs'])
        self.assertEqual(trans_prob.shape, expected_shapes['trans_prob'])
        self.assertEqual(icovs.shape, expected_shapes['icovs'])

    def test_organise_hmm_features_across_chunks_empty(self):
        hmm_features_list = []

        with self.assertRaises(IndexError):  # or another appropriate error
            self.hmm_features.organise_hmm_features_across_chunks(hmm_features_list)


if __name__ == '__main__':
    unittest.main()