from cortexa.structure.min_col import MiniColumn

import logging
import logging.config
import numpy as np
class MacroColumn:
    
    '''
    The function of a macrocolumn is to store 
    sparse distributed representations of its 
    overall input patterns, and to act as a 
    recognizer of those patterns.
    '''
    
    

    def __init__(self, 
                mac_id,
                minicolumn_pool_size,
                pyramidal_count_per_minicol,
                bottom_up_connections,
                sigmoid_para_one,
                sigmoid_para_two):
        self.id = mac_id
        logging.config.fileConfig('logging.conf')
        self._logger = logging.getLogger(__name__)
        self.minicolumn_pool = []
        self.encoded_winners = None
        self.familiarity = None    
        self.sigmoid_para_one = sigmoid_para_one
        self.sigmoid_para_two = sigmoid_para_two
        for min_col in range(minicolumn_pool_size):
            col = MiniColumn(min_col,
                            pyramidal_count_per_minicol,
                            bottom_up_connections,
                            self.sigmoid_para_one,
                            self.sigmoid_para_two)
            self.minicolumn_pool.append(col)
          

    # Determine the familiarity of a
    #  macrocolumn’s input. To a first 
    #  approximation, an input’s familiarity
    #  is its maximum similarity to any input
    #  previously stored in the macrocolumn.
    

    def calculate_input_familiarity(self):
        input_summations = 0
        for mincol in self.minicolumn_pool:
            input_summations += mincol.first_round_activation
        self.familiarity = input_summations

    def calculate_expansivity(self):
        '''
        The expansivity is set as an increasing nonlinear
        function of G(macrocolumnal global familiarity).
        Currently expressed as cascading if/elif/else
        table(-- replace with better impl)
        
        The specific parameters of any instance of 
        the G-to-η mapping will determine the specifics 
        of the relation between input similarity 
        and code similarity, i.e., the expected 
        code intersection as a function of input similarity.
 
        The idea is to increase the range of 
        relative win likelihoods, W(i) over
        any given CM(minicolumn)’s units as G goes to 1.
        '''
        #def expansivityFunction(par1, par2):
        #  return par1e^(par2* ____)
                  #10
        #0.0035e^(10*x) 
        exp = None
        if self.familiarity == 0:
            exp = 0
        elif self.familiarity <= 0.2:
            exp = 0
        elif self.familiarity <= 0.4:
            exp = 0.2
        elif self.familiarity <= 0.6:
            exp = 5
        elif self.familiarity <= 0.8:
            exp = 12
        else:
            exp = 100
        self._logger.debug("For macrocolumn: " + str(self.id) +"The expansivity is: " + str(exp))
        return exp

    def csa_first_round(self, bu_input):
        for min_col in self.minicolumn_pool:
            min_col.cm_first_round(bu_input)

    def csa_second_round(self, bu_input, expansivity):
        for min_col in self.minicolumn_pool:
            min_col.cm_second_round(bu_input, 
                                    expansivity)
        self.encoded_winners = [min_col.cm_winner for min_col in self.minicolumn_pool]


    def code_selection_algorithm(self, bu_input):
        self.csa_first_round(bu_input)
        self.calculate_input_familiarity()
        cm_expansivity = self.calculate_expansivity()
        self.csa_second_round(bu_input, cm_expansivity)
       
    
    def reset_column(self):
        self.familiarity = None
        self.encoded_winners = None
