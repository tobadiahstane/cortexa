'''

Test cases for PyramidalCell

Oluwatobi Ajoku Dec 2021

'''

from context import cortexa

import unittest
import sensible_defaults as defaults

class TestPyramidalCell(unittest.TestCase):
    def make_input(self,cnt):
        return list(range(cnt))

    def make_cell(self, id, bu_inputs):
        return cortexa.structure.pyramidal_cell.PyramidalCell(id, 
                                                            bu_inputs,
                                                            defaults.sigmoid_p1,
                                                            defaults.sigmoid_p2)


    def test_count_of_inputs(self):
        bu_inputs =self.make_input(12)
        cell = self.make_cell(0, bu_inputs)
        self.assertEqual(12, cell.count_of_bottom_up_inputs())
        self.assertIsNone(cell.postsynaptic_potential)
        self.assertIsNone(cell.rel_win_chance)

    def test_receive_signal(self):
        bu_inputs = self.make_input(12)
        cell = self.make_cell(0, bu_inputs)        
        test_bu_signal = [0, 3, 5, 1, 9]
        test_signal = {"bu": test_bu_signal}
        cell.recieve_signals(test_signal)
        self.assertEqual(0, cell.postsynaptic_potential)
    
    def test_second_round(self):
        bu_inputs = self.make_input(12)
        cell = self.make_cell(0, bu_inputs)        
        test_bu_signal = [0, 3, 5, 1, 9]
        test_signals = {"bu": test_bu_signal}
        cell.determine_win_chance(0, test_signals)
        self.assertEqual(1, cell.rel_win_chance)
        cell.reinforce_signal(test_signals)
        self.assertEqual(1, cell.bu_inputs[0])
        self.assertIsNone(cell.postsynaptic_potential)
        self.assertIsNone(cell.rel_win_chance)


if __name__ == '__main__':
    unittest.main()     