import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace

# num of dimensions for embedding
EMBED_DIM = 5
# size of the pitch vocabulary, 120, 0 to 119 for midi pitches
PITCH_VOCAB_SIZE = 120
# default batch size
BATCH_SIZE = 32

class EmbedModel(nn.Module):
    def __init__(self, vocab_size=120, embed_dim=EMBED_DIM):
        super(EmbedModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, X):
        return self.embed(X)

class HarmonyModel(nn.Module):
    '''
    Given three notes, this model will determine a fourth note that will fit
    within the context in terms of harmony.
    '''
    def __init__(self, input_dim, output_dim):
        super(HarmonyModel, self).__init__()
        self.aff_1 = nn.Linear(input_dim, output_dim)

    def forward(self, X):
        aff_1 = self.aff_1(X)
        return aff_1

class CombineModel(nn.Module):
    '''
    Given three note suggestions, this model will determine which one to use.
    '''
    def __init__(self, input_dim, output_dim):
        super(CombineModel, self).__init__()
        self.aff_1 = nn.Linear(input_dim, output_dim)

    def forward(self, X):
        aff_1 = self.aff_1(X)
        return aff_1

class NoteModel(nn.Module):
    '''
    Given a sequence of notes, this model will predict the next note.
    '''
    def __init__(self, input_dim, hidden_dim=100, **kwargs):
        super(NoteModel, self).__init__()
        self.num_layers = kwargs.get("num_layers", 1)
        self.batch_size = kwargs.get("batch_size", 32)
        self.hidden_dim = hidden_dim
        vocab_size = kwargs.get("vocab_size", PITCH_VOCAB_SIZE)

        self.lstm_1 = nn.LSTM(input_dim, hidden_dim)
        self.pitch = nn.Linear(hidden_dim, vocab_size)
        self.rhythm = nn.Linear(hidden_dim, 1)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, X):
        # lstm_1.shape == [seq_len, batch_size, hidden_size]
        lstm_1, self.hidden = self.lstm_1(X, self.hidden)
        last_step = lstm_1[-1] # get only output from last time step
        pitch_scores = self.pitch(last_step)
        rhythm_scores = self.rhythm(last_step)
        return pitch_scores, rhythm_scores

def assert_embed_model():
    embed = EmbedModel()
    input_tensor = torch.randint(0, 88, (BATCH_SIZE,)).long()
    output = embed(input_tensor)
    assert output.shape == (BATCH_SIZE, EMBED_DIM), "Invalid output shape."

def assert_harmony_model():
    # +1 for rest and +1 for part number
    input_dim = EMBED_DIM + 2
    harmony = HarmonyModel(3 * input_dim, PITCH_VOCAB_SIZE)
    input_tensor = torch.randint(0,88,(BATCH_SIZE,3,input_dim,)) \
                        .view(BATCH_SIZE, 3*input_dim)
    output = harmony(input_tensor)
    assert output.shape == (BATCH_SIZE, PITCH_VOCAB_SIZE), \
        "Invalid output shape."

def assert_combine_model():
    # +1 for rest
    input_dim = EMBED_DIM + 1
    harmony = HarmonyModel(3 * input_dim, PITCH_VOCAB_SIZE)
    input_tensor = torch.randint(0,88,(BATCH_SIZE,3,input_dim,)) \
                        .view(BATCH_SIZE, 3*input_dim)
    output = harmony(input_tensor)
    assert output.shape == (BATCH_SIZE, PITCH_VOCAB_SIZE), \
                "Invalid output shape."

def assert_note_model():
    input_dim = EMBED_DIM + 1
    seq_len = 32
    forward_pitch = NoteModel(input_dim, PITCH_VOCAB_SIZE, 100)
    input_tensor = torch.randint(0,88,(seq_len,BATCH_SIZE,input_dim))
    pitches, rhythms = forward_pitch(input_tensor)
    assert pitches.shape == (BATCH_SIZE, PITCH_VOCAB_SIZE), \
            "Invalid output shape."
    assert rhythms.shape == (BATCH_SIZE, 1), "Invalid output shape."

if __name__ == "__main__":
    assert_harmony_model()
    assert_combine_model()
    assert_embed_model()
    assert_note_model()
    pass
