import torch
import torch.nn as nn
from pdb import set_trace

# num of dimensions per note
dims = 73

class HarmonyModel(nn.Module):
    '''
    Given three notes, this model will determine a fourth note that will fit
    within the context in terms of harmony.
    '''
    def __init__(self, input_size, output_size):
        super(HarmonyModel, self).__init__()
        self.aff_1 = nn.Linear(input_size, output_size)

    def forward(self, X):
        aff_1 = self.aff_1(X)
        return torch.softmax(aff_1, dim=0)

class CombineModel(nn.Module):
    '''
    Given three note suggestions, this model will determine which one to use.
    '''
    def __init__(self, input_size, output_size):
        super(CombineModel, self).__init__()
        self.aff_1 = nn.Linear(input_size, output_size)

    def forward(self, X):
        aff_1 = self.aff_1(X)
        return torch.softmax(aff_1, dim=0)

class ForwardPitchModel(nn.Module):
    '''
    Given a sequence of notes, this model will predict the next note.
    '''
    def __init__(self, input_size, hidden_size, output_size):
        super(ForwardPitchModel, self).__init__()
        self.hidden_dim = hidden_size

        self.lstm_1 = nn.LSTM(input_size, hidden_size)
        self.aff_1 = nn.Linear(hidden_size, output_size)

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))


def assert_harmony_model():
    global dims
    with torch.no_grad():
        harmony = HarmonyModel(3*dims, dims)
        input_tensor = torch.randn(10, 3, dims).view(10, 3*dims)
        output = harmony(input_tensor)
        assert output.shape == (10, dims), "Invalid output shape."

def assert_combine_model():
    global dims
    with torch.no_grad():
        harmony = CombineModel(3*dims, dims)
        input_tensor = torch.randn(10, 3, dims).view(10, 3*dims)
        output = harmony(input_tensor)
        assert output.shape == (10, dims), "Invalid output shape."

if __name__ == "__main__":
    assert_harmony_model()
    assert_combine_model()
