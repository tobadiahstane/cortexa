'''

Test cases for MacroColumn

Oluwatobi Ajoku Dec 2021

'''

from context import cortexa
import unittest
import sensible_defaults as defaults

class TestMacroColumn(unittest.TestCase):

    def make_inputs(self, cnt):
        return list(range(cnt))

    def make_macro(self, id, bu_conns):
        return cortexa.structure.mac_col.MacroColumn(id,
                                                    defaults.small_minpool_size,
                                                    defaults.small_p_pool_size,
                                                    bu_conns,
                                                    defaults.sigmoid_p1,
                                                    defaults.sigmoid_p2)

    def test_macro(self):
        bu_inputs = self.make_inputs(12)
        test_col = self.make_macro(0,bu_inputs)
        for min in test_col.minicolumn_pool:
            self.assertEqual(defaults.small_p_pool_size, len(min.pyramidal_pool))
            self.assertEqual(defaults.sigmoid_p1, min.pyramidal_pool[0].sigmoid_p1)
            self.assertEqual(defaults.sigmoid_p2, min.pyramidal_pool[2].sigmoid_p2)
        bu_signal = [0, 1, 2, 7, 10]
        test_col.code_selection_algorithm(bu_signal)
        winners = test_col.encoded_winners
        self.assertIsNotNone(winners)
        self.assertEqual(defaults.small_minpool_size, 
                        len(winners))

        input_str = str(bu_signal)[1:-1]
        print("")
        print("This is the input: " + input_str)
        output = list(map(lambda x: x.cell_id , winners))
        print("For input " + input_str + " This is the output: " + str(output)[1:-1])
        print("The current familiarity with the input is: " + str(test_col.familiarity))


        test_col.reset_column()
        n_bu_signal = [0, 1, 2, 7, 10]
        test_col.code_selection_algorithm(n_bu_signal)
        n_winners = test_col.encoded_winners
        self.assertIsNotNone(n_winners)
        self.assertEqual(defaults.small_minpool_size, 
                        len(n_winners))

        n_input_str = str(n_bu_signal)[1:-1]
        print("")
        print("This is the next input: " + n_input_str)
        n_output = list(map(lambda x: x.cell_id , n_winners))
        print("For input " + n_input_str + " This is the next output: " + str(n_output)[1:-1])
        print("The current familiarity with the next input is: " + str(test_col.familiarity))

