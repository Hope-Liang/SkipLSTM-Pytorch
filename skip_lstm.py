import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.nn.modules.rnn import RNNCellBase
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_
import math
import numpy as np


class BinaryLayer(Function):
    def forward(self, input):
        return input.round()
 
    def backward(self, grad_output):
        return grad_output


def SkipLSTMCell(input, hidden, num_layers, w_ih, w_hh, w_uh, b_ih=None, b_hh=None, b_uh=None,
                  activation=F.tanh, lst_layer_norm=None):
    if num_layers != 1:
        raise RuntimeError("wrong num_layers: got {}, expected {}".format(num_layers, 1))
    w_ih, w_hh = w_ih[0], w_hh[0]
    b_ih = b_ih[0] if b_ih is not None else None
    b_hh = b_hh[0] if b_hh is not None else None

    c_prev, h_prev, update_prob_prev, cum_update_prob_prev = hidden

    gates = F.linear(input, w_ih, b_ih) + F.linear(h_prev, w_hh, b_hh)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    if lst_layer_norm:
        ingate = lst_layer_norm[0][0](ingate.contiguous())
        forgetgate = lst_layer_norm[0][1](forgetgate.contiguous())
        cellgate = lst_layer_norm[0][2](cellgate.contiguous())
        outgate = lst_layer_norm[0][3](outgate.contiguous())

    ingate = F.sigmoid(ingate)
    forgetgate = F.sigmoid(forgetgate)
    cellgate = activation(cellgate)
    outgate = F.sigmoid(outgate)

    new_c_tilde = (forgetgate * c_prev) + (ingate * cellgate)
    new_h_tilde = outgate * activation(new_c_tilde)
    # Compute value for the update prob
    new_update_prob_tilde = F.sigmoid(F.linear(new_c_tilde, w_uh, b_uh))
    # Compute value for the update gate
    cum_update_prob = cum_update_prob_prev + torch.min(update_prob_prev, 1. - cum_update_prob_prev)
    # round
    bn = BinaryLayer()
    update_gate = bn(cum_update_prob)
    # Apply update gate
    new_c = update_gate * new_c_tilde + (1. - update_gate) * c_prev
    new_h = update_gate * new_h_tilde + (1. - update_gate) * h_prev
    new_update_prob = update_gate * new_update_prob_tilde + (1. - update_gate) * update_prob_prev
    new_cum_update_prob = update_gate * 0. + (1. - update_gate) * cum_update_prob

    new_state = (new_c, new_h, new_update_prob, new_cum_update_prob)
    new_output = (new_h, update_gate)

    return new_output, new_state


# Implementation from nn._functions.rnn.py
def BasicLSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None,
                  activation=F.tanh, lst_layer_norm=None):
    '''
    Parameters of a basic LSTM cell
    :param forget_bias: float, the bias added to the forget gates. The biases are for the forget gate in order to
    reduce the scale of forgetting in the beginning of the training. Default 1.0.
    :param activation: activation function of the inner states. Default: F.tanh()
    :param layer_norm: bool, whether to use layer normalization.
    :return: 
    '''
    hx, cx = hidden

    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    if lst_layer_norm:
        ingate = lst_layer_norm[0](ingate.contiguous())
        forgetgate = lst_layer_norm[1](forgetgate.contiguous())
        cellgate = lst_layer_norm[2](cellgate.contiguous())
        outgate = lst_layer_norm[3](outgate.contiguous())

    ingate = F.sigmoid(ingate)
    forgetgate = F.sigmoid(forgetgate)
    cellgate = activation(cellgate)
    outgate = F.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * activation(cy)

    return hy, cy


class CCellBase(RNNCellBase):

    def __init__(self, cell, learnable_elements, input_size, hidden_size, num_layers=1,
                    bias=True, batch_first=False, activation=F.tanh, layer_norm=False):
        super(CCellBase, self).__init__(input_size, hidden_size, bias, num_chunks=learnable_elements)

        assert num_layers == 1 # This implementation only works for single layer RNN
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.batch_first = batch_first
        self.cell = cell
        self.num_layers = num_layers
        self.weight_ih = Parameter(xavier_uniform_(torch.Tensor(learnable_elements * hidden_size, input_size)))
        self.weight_hh = Parameter(xavier_uniform_(torch.Tensor(learnable_elements * hidden_size, hidden_size)))
        if bias:
            self.bias_ih = Parameter(torch.zeros(learnable_elements * hidden_size))
            self.bias_hh = Parameter(torch.zeros(learnable_elements * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        self.activation = activation
        self.layer_norm = layer_norm
        self.lst_bnorm_rnn = None


class CCellBaseSkipLSTM(CCellBase):

    def __init__(self, cell, learnable_elements, input_size, hidden_size, num_layers=1,
                    bias=True, batch_first = False, activation=F.tanh, layer_norm=False):
        super(CCellBaseSkipLSTM, self).__init__(cell, learnable_elements, input_size, hidden_size, num_layers,
                                                bias, batch_first, activation, layer_norm)
        self.weight_uh = Parameter(xavier_uniform_(torch.Tensor(1, hidden_size)))
        if bias:
            self.bias_uh = Parameter(torch.ones(1))
        else:
            self.register_parameter('bias_uh', None)

    def forward(self, input, hx = None):

        if len(input.shape) == 3:
            if self.batch_first:
                input = input.transpose(0,1)
            sequence_length, batch_size, input_size = input.shape
        else:
            sequence_length = 1
            batch_size, input_size = input.shape

        if hx is None:
            hx = self.init_hidden(batch_size)
            if input.is_cuda:
                if self.num_layers == 1:
                    hx = tuple([x.cuda() for x in hx])
                else:
                    hx = [tuple([j.cuda() if j is not None else None for j in i]) for i in hx]

        if len(input.shape) == 3:
            self.check_forward_input(input[0])
            if self.num_layers > 1:
                self.check_forward_hidden(input[0], hx[0][0], '[0]')
                self.check_forward_hidden(input[0], hx[0][1], '[1]')
            else:
                self.check_forward_hidden(input[0], hx[0], '[0]')
                self.check_forward_hidden(input[0], hx[1], '[1]')
        else:
            self.check_forward_input(input)
            if self.num_layers > 1:
                self.check_forward_hidden(input, hx[0][0], '[0]')
                self.check_forward_hidden(input, hx[0][1], '[1]')
            else:
                self.check_forward_hidden(input, hx[0], '[0]')
                self.check_forward_hidden(input, hx[1], '[1]')

        # Initialize batchnorm layers
        if self.layer_norm and self.lst_bnorm_rnn is None:
            self.lst_bnorm_rnn = []
            for i in np.arange(self.num_layers):
                # Create gain and bias for input_gate, new_input, forget_gate, output_gate
                lst_bnorm_rnn_tmp = torch.nn.ModuleList([nn.BatchNorm1d(self.hidden_size) for i in np.arange(4)])
                if input.is_cuda:
                    lst_bnorm_rnn_tmp = lst_bnorm_rnn_tmp.cuda()
                self.lst_bnorm_rnn.append(lst_bnorm_rnn_tmp)
            self.lst_bnorm_rnn = torch.nn.ModuleList(self.lst_bnorm_rnn)

        lst_output = []
        lst_update_gate = []
        for t in np.arange(sequence_length):
            output, hx = self.cell(
                input[t], hx, self.num_layers,
                self.weight_ih, self.weight_hh, self.weight_uh,
                self.bias_ih, self.bias_hh, self.bias_uh,
                activation=self.activation,
                lst_layer_norm=self.lst_bnorm_rnn
            )
            new_h, update_gate = output
            lst_output.append(new_h)
            lst_update_gate.append(update_gate)
        output = torch.stack(lst_output)
        update_gate = torch.stack(lst_update_gate)
        if self.batch_first:
            output = output.transpose(0, 1)
            update_gate = update_gate.transpose(0, 1)
        return output, hx, update_gate


class CSkipLSTMCell(CCellBaseSkipLSTM):
    def __init__(self, *args, **kwargs):
        super(CSkipLSTMCell, self).__init__(cell=SkipLSTMCell, learnable_elements=4, num_layers=1,
                                            *args, **kwargs)

    def init_hidden(self, batch_size):
        return (Variable(torch.randn(batch_size, self.hidden_size)),
                Variable(torch.randn(batch_size, self.hidden_size)),
                Variable(torch.ones(batch_size, 1),requires_grad=False),
                Variable(torch.zeros(batch_size, 1),requires_grad=False))

