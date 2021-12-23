from cortexa.structure.pyramidal_cell import PyramidalCell

import logging
import logging.config
import numpy as np
class MiniColumn:
    '''
    Small, physically localized pools of 
    L2/3 pyramidals, e.g., ∼20 such cells,
    act as winner-take-all (WTA) 
    competitive modules (CMs).
    '''

    # rework pyramidal from individual obj to set of operations on arrays
    

    def __init__(self, id, pyramidal_pool_size, bottom_up_conn, 
                sigmoid_p1, sigmoid_p2):
        logging.config.fileConfig('logging.conf')
        self._logger = logging.getLogger(__name__)

        self.pyramidal_pool = []
        self.min_id = id
        self.first_round_activation = None
        self.cm_winner = None
        self.sec_rnd_win_prob = None
        for p_cell in range(pyramidal_pool_size):
            self.pyramidal_pool.append(PyramidalCell(p_cell,
                                                        bottom_up_conn,
                                                        sigmoid_p1,
                                                        sigmoid_p2))

    def cm_first_round(self, bottom_up_signal):
        signal = {"bu": bottom_up_signal}
        
        for p_cell in self.pyramidal_pool:
            p_cell.recieve_signals(signal)
        self.pyramidal_pool.sort(key=lambda x : x.postsynaptic_potential, reverse=True)
        
        self.first_round_activation = self.pyramidal_pool[0].postsynaptic_potential

    def cm_second_round(self, bottom_up_signal, expansivity):
        signals = {"bu": bottom_up_signal}
        #this could be better expressed as a set of functions operating on an array of data
        sum_win_likely = 0
        for p_cell in self.pyramidal_pool:
            p_cell.determine_win_chance(expansivity,signals)
            sum_win_likely += p_cell.rel_win_chance
        self.sec_rnd_win_prob = np.array([p_cell.rel_win_chance/sum_win_likely for p_cell in self.pyramidal_pool])
        self._logger.debug("For column: "+ str(self.min_id) +" round 2 this is the set of win probabilities: " + str(self.sec_rnd_win_prob)[1:-1])

        self.cm_winner = np.random.choice(self.pyramidal_pool,1,p = self.sec_rnd_win_prob)[0]
        for cell in self.pyramidal_pool:
            if(cell.cell_id == self.cm_winner.cell_id):
                cell.reinforce_signal(signals)
            else:
                cell.lose_cm()
        self.first_round_activation = None


    def reset_column(self):
        self.first_round_activation = None
        self.cm_winner = None
        self.sec_rnd_win_prob = None
        for p_cell in self.pyramidal_pool:
            p_cell.clear_cell_synapses()






