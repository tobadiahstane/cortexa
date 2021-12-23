import numpy as np

class PyramidalCell:
    ''' 
    Each pyramidal cell has j number of bottom up(bu) inputs with w weight.
    These BU input weights (synapses) are binary
    On initialization of pyramidal each bu input has w of 0.
    and are permanently set to a weight of 1 the first time the
    pre- and postsynaptic units are co-active
    '''


    def __init__(self, cell_id, bu_connections, sigmoid_p1, sigmoid_p2):
        self.bu_inputs = {}
        self.bu_input_summation = 0
        self.postsynaptic_potential = None
        self.rel_win_chance = None
        self.cell_id = cell_id
        self.sigmoid_p1 = sigmoid_p1
        self.sigmoid_p2 = sigmoid_p2
        self.add_new_bottom_up_inputs(bu_connections)

    def count_of_bottom_up_inputs(self):
        return len(self.bu_inputs)

    def add_new_bottom_up_input(self, inpt):
        #bu input has w of 0
        self.bu_inputs[inpt] = 0

    def add_new_bottom_up_inputs(self, inpts):
        for inpt in inpts:
            self.add_new_bottom_up_input(inpt)

    def calculate_input_summation(self, origin_inpt):
        # Because unit activations are binary, 
        # we can simply sum the weights which are also binary.
        self.bu_input_summation = (self.bu_input_summation +
                                   self.bu_inputs[origin_inpt])

    def recieve_bottom_up_signal(self, bu_signals):
        # Each PyramidalCell i computes its raw input summation, u(i).
        for bu_signal in bu_signals:
            self.calculate_input_summation(bu_signal)

    def normalize_input_summations(self, signals):
        #Normalize input summation to [0..1], yielding V(i).
        bu_signals = signals['bu']
        normalized_bu_summation = (self.bu_input_summation / len(bu_signals))
        return normalized_bu_summation

    def calculate_postsynaptic_potential(self, signals):
        #V(i) is a local measure of support, or likelihood, that this unit should be activated.
        self.postsynaptic_potential = self.normalize_input_summations(signals)

    def recieve_signals(self, signals):
        bu_signals = signals['bu']
        if(bu_signals is not None or len(bu_signals) == 0):
            self.recieve_bottom_up_signal(bu_signals)
        self.calculate_postsynaptic_potential(signals)
    
    def pass_activation_through_sigmoid(self, expansivity):
        initial= expansivity/np.exp(np.negative(self.sigmoid_p1*self.postsynaptic_potential+self.sigmoid_p2))
        return initial+1

    def determine_win_chance(self, expansivity, signals):
        self.recieve_signals(signals)
        self.rel_win_chance =  self.pass_activation_through_sigmoid(expansivity)
     
    def reinforce_bottom_up_inputs(self, bu_signals):
       for inpt in bu_signals:
           self.bu_inputs[inpt] = 1
       
    def clear_cell_synapses(self):
        self.postsynaptic_potential = None
        self.bu_input_summation = 0
        self.rel_win_chance = None

    def reinforce_signal(self, signals):
         bu_signals = signals['bu']
         self.reinforce_bottom_up_inputs(bu_signals)
         self.clear_cell_synapses()

    def lose_cm(self):
        self.clear_cell_synapses()