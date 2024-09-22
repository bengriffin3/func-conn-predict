import numpy as np
import unittest
from nets_predict.classes.hmm import HMMInference, TimeSeriesProcessing, FeatureEngineering, HMMFeatures, Prediction

class TestHiddenMarkovModel(unittest.TestCase):

    def test_initialize_trans_prob(self):
        # Create an instance of the FeatureEngineering class
        feature_engineering = FeatureEngineering()  # Adjust this if the constructor requires arguments

        # 2 test matrices
        initial_trans_prob_m1 = feature_engineering.intialise_trans_prob(10, 3)
        initial_trans_prob_m2 = feature_engineering.intialise_trans_prob(100, 4)

        # Target initial probabilities
        m1 = np.array([[0.83333333, 0.08333333, 0.08333333],
                       [0.08333333, 0.83333333, 0.08333333],
                       [0.08333333, 0.08333333, 0.83333333]])
        m2 = np.array([[0.97087379, 0.00970874, 0.00970874, 0.00970874],
                       [0.00970874, 0.97087379, 0.00970874, 0.00970874],
                       [0.00970874, 0.00970874, 0.97087379, 0.00970874],
                       [0.00970874, 0.00970874, 0.00970874, 0.97087379]])

        # Test 1 - test for shape correctness
        self.assertEqual(initial_trans_prob_m1.shape[0], initial_trans_prob_m1.shape[1])
        self.assertEqual(initial_trans_prob_m2.shape[0], initial_trans_prob_m2.shape[1])

        # Test 2 - check if the initialized probabilities match the expected values
        self.assertTrue(np.allclose(np.round(initial_trans_prob_m1, 8), m1))
        self.assertTrue(np.allclose(np.round(initial_trans_prob_m2, 8), m2))

        # Test 3 - check if rows sum to 1
        self.assertTrue(np.all(np.isclose(np.sum(initial_trans_prob_m1, axis=1), 1)))
        self.assertTrue(np.all(np.isclose(np.sum(initial_trans_prob_m2, axis=1), 1)))

        # Test 4 - check if columns sum to 1
        self.assertTrue(np.all(np.isclose(np.sum(initial_trans_prob_m1, axis=0), 1)))
        self.assertTrue(np.all(np.isclose(np.sum(initial_trans_prob_m2, axis=0), 1)))

if __name__ == '__main__':
    unittest.main()

# import numpy as np
# import unittest
# from nets_predict.classes.hmm import HMMInference, TimeSeriesProcessing, FeatureEngineering, HMMFeatures, Prediction

# class TestHiddenMarkovModel(unittest.TestCase):

#     def test_intialise_trans_prob(self):

#         HMMClass = HiddenMarkovModelClass()


#         # 2 test matrices
#         initial_trans_prob_m1 = HMMClass.intialise_trans_prob(10, 3)
#         initial_trans_prob_m2 = HMMClass.intialise_trans_prob(100, 4)

#         # target initial probs
#         m1 = np.array([[0.83333333, 0.08333333, 0.08333333], [0.08333333, 0.83333333, 0.08333333], [0.08333333, 0.08333333, 0.83333333]])
#         m2 = np.array([[0.97087379, 0.00970874, 0.00970874, 0.00970874], [0.00970874, 0.97087379, 0.00970874, 0.00970874], [0.00970874, 0.00970874, 0.97087379, 0.00970874], [0.00970874, 0.00970874, 0.00970874, 0.97087379]])


#         # Test 1 - test for shape correction
#         self.assertEqual(initial_trans_prob_m1.shape[0], initial_trans_prob_m1.shape[1])
#         self.assertEqual(initial_trans_prob_m2.shape[0], initial_trans_prob_m2.shape[1])

#         # Test 2 - now test what happens is what we expect it to
#         self.assertEqual(np.round(initial_trans_prob_m1,8).tolist(), m1.tolist())
#         self.assertEqual(np.round(initial_trans_prob_m2,8).tolist(), m2.tolist())

#         # Test 3 - rows sum to 1
#         self.assertEqual(np.sum(np.sum(initial_trans_prob_m1, axis=0)==np.ones(3)), 3)
#         self.assertEqual(np.sum(np.sum(initial_trans_prob_m2, axis=0)==np.ones(4)), 4)

#         # Test 3 - columns sum to 1
#         self.assertEqual(np.sum(np.sum(initial_trans_prob_m1, axis=1)==np.ones(3)), 3)
#         self.assertEqual(np.sum(np.sum(initial_trans_prob_m2, axis=1)==np.ones(4)), 4)



# if __name__ == '__main__':
#     unittest.main()