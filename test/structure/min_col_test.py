'''
Test Cases for Minicolumn

Oluwatobi Ajoku Dec 2021

'''

from context import cortexa


import unittest
import sensible_defaults as defaults
class TestMiniColumn(unittest.TestCase):

    def make_input(self,cnt):
        return list(range(cnt))

    def make_minicol(self,id ,bu_inputs):
        return cortexa.structure.min_col.MiniColumn(id,
                                                    defaults.small_p_pool_size,
                                                    bu_inputs,
                                                    defaults.sigmoid_p1,
                                                    defaults.sigmoid_p2)

    def test_minicol(self):
        bu_inputs = self.make_input(12)
        
        test_col = self.make_minicol(0,bu_inputs)
        bu_signal = [0, 1, 3, 6, 8]
        self.assertIsNone(test_col.first_round_activation)
        test_col.cm_first_round(bu_signal)
        self.assertEqual(0, test_col.first_round_activation)
        test_col.cm_second_round(bu_signal, expansivity=0)
        
        #relative win chance for first round for each pyramidal is 1
        # win probability for first round for each is 1/count of pyramidals
        self.assertAlmostEqual(1/defaults.small_p_pool_size, test_col.sec_rnd_win_prob[0])
        winner = test_col.cm_winner
        self.assertIsNotNone(winner)
        self.assertEqual(1, winner.bu_inputs[0])
        self.assertIsNone(test_col.first_round_activation)

        # recieving same signal means the minicolumns 
        # next first round activation high
        test_col.cm_first_round(bu_signal)
        self.assertEqual(1, test_col.first_round_activation)

        # Recieving a similar signal should 
        # result in lesser activation
        test_col.reset_column()
        close_bu_signal = [0, 1, 3, 6, 10] 
            # instead of  [0, 1, 3, 6, 8]
        test_col.cm_first_round(close_bu_signal)
        self.assertEqual(0.8, test_col.first_round_activation)
        


if __name__ == '__main__':
    unittest.main()