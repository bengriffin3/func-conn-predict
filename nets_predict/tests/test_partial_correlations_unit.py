import numpy as np
import unittest
from nets_predict.classes.partial_correlation import PartialCorrelation, PartialCorrelationAnalysis, CovarianceUtils

class TestPartialCorrelation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.partial_corr = PartialCorrelation()
        cls.partial_corr_analysis = PartialCorrelationAnalysis(cls.partial_corr) 
        cls.covariance_utils = CovarianceUtils()

    def test_matrix_inversion(self):
        # matrix to invert
        m1 = np.array([[0.5, 1.5],[1.5, 4.5]])

        # invert using function
        m1_inv = np.round(self.covariance_utils.covariance_to_precision(m1, rho=0.1), 8)

        # ideal inversion
        m2 = np.array([[9.01960784, -2.94117647], [-2.94117647, 1.17647059]]) 

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

    def test_partial_correlation_calc(self):

        #self, Sigma, rho=0.1, do_rtoz=-18.8310):
        Sigma = np.array([[0.5, 1.5],[1.5, 4.5]])
        Sigma_partial_corr = np.array([[0 , 0.75441865], [0.75441865, 0]])
        Sigma_partial_corr_r2z = np.array([[ 0, 18.51336385], [18.51336385,  0]])
        Sigma_precision_matrix = np.array([[ 9.06017797, -2.81946608], [-2.81946608,  1.54160175]])

        Sigma_partials = self.partial_corr.partial_corr(Sigma)

        partial_correlations = np.round(Sigma_partials[0], 8)
        partial_correlations_r2z = np.round(Sigma_partials[1], 8)
        precision_matrix = np.round(Sigma_partials[2], 8)

        self.assertEqual(Sigma_partial_corr.tolist(), partial_correlations.tolist())
        self.assertEqual(Sigma_partial_corr_r2z.tolist(), partial_correlations_r2z.tolist())
        self.assertEqual(Sigma_precision_matrix.tolist(), precision_matrix.tolist())


    def test_extract_upper_diag_array(self):
        # Using the instance from setUpClass
        hmm_covariances = np.array([[[1, 7, 0],[5, 2, 0],[7, 7, 1]],
                                     [[6, 0, 1],[4, 3, 0],[6, 5, 8]],
                                     [[6, 5, 4],[3, 5, 0],[0, 6, 6]],
                                     [[9, 1, 0],[5, 0, 2],[5, 1, 5]]])

        hmm_covariances_upper_diag_target = np.array([[7, 0, 0], [0, 1, 0], [5, 4, 0], [1, 0, 2]])
        hmm_covariances_upper_diag_fun = self.partial_corr_analysis.extract_upper_off_main_diag(hmm_covariances)
        np.testing.assert_array_equal(hmm_covariances_upper_diag_fun, hmm_covariances_upper_diag_target)

        hmm_covariances_2 = np.array([[0.5, 1.5],[1.5, 4.5]])
        hmm_covariances_upper_diag_target_2 = np.array([1.5])
        hmm_covariances_upper_diag_fun_2 = self.partial_corr_analysis.extract_upper_off_main_diag(hmm_covariances_2)
        np.testing.assert_array_equal(hmm_covariances_upper_diag_fun_2, hmm_covariances_upper_diag_target_2)


if __name__ == '__main__':
    unittest.main()