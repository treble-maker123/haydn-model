import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace

# num of dimensions for embedding
EMBED_DIM = 5
# size of the pitch vocabulary, 0 to PITCH_VOCAB_SIZE-2 for midi pitches,
# PITCH_VOCAB_SIZE - 1for rest.
# REST IS A PITCH!
PITCH_VOCAB_SIZE = 120
# default batch size
BATCH_SIZE = 32

class PitchEmbedModel(nn.Module):
    def __init__(self, vocab_size=PITCH_VOCAB_SIZE, embed_dim=EMBED_DIM):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, X):
        return self.embed(X)

class HarmonyModel(nn.Module):
    '''
    Given three embedded note vectors, this model will determine a fourth note
    that will fit within the context in terms of harmony.

    Default input_shape has 4 parts for first dim, one of which will be blank,
    and 5 embed size and 4 part indicators as second dim.
    '''
    def __init__(self, input_shape=(4,9),
                 vocab_size=PITCH_VOCAB_SIZE, **kwargs):
        super().__init__()
        # compress the representation for each part into latent vectors
        num_parts = input_shape[0]
        hidden_dim = kwargs.get("hidden_dim", 4)
        self.pt1 = nn.Linear(9, hidden_dim)
        nn.init.kaiming_normal_(self.pt1.weight)
        self.pt2 = nn.Linear(9, hidden_dim)
        nn.init.kaiming_normal_(self.pt2.weight)
        self.pt3 = nn.Linear(9, hidden_dim)
        nn.init.kaiming_normal_(self.pt3.weight)
        self.pt4 = nn.Linear(9, hidden_dim)
        nn.init.kaiming_normal_(self.pt4.weight)
        # +4 to make room for part prediction
        self.fc1 = nn.Linear(num_parts * hidden_dim, vocab_size + num_parts)

    def forward(self, X):
        pt1 = self.pt1(X[:,0,:])
        pt2 = self.pt2(X[:,1,:])
        pt3 = self.pt3(X[:,2,:])
        pt4 = self.pt4(X[:,3,:])
        comb = F.relu(torch.cat((pt1, pt2, pt3, pt4), dim=1))
        return self.fc1(comb)

class JudgeModel(nn.Module):
    '''
    Given three note suggestions, this model will judge which one to use.
    '''
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        num_notes, num_dim = input_dim
        self.fc1 = nn.Linear(num_notes * num_dim, hidden_dim)
        nn.init.kaiming_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, X):
        flatten = X.view(len(X), -1)
        l1 = F.relu(self.fc1(flatten))
        return self.fc2(l1)

class NoteModel(nn.Module):
    '''
    Given a sequence of notes, this model will predict the next note.
    '''
    def __init__(self, input_dim, hidden_dim=100, **kwargs):
        super().__init__()
        vocab_size = kwargs.get("vocab_size", PITCH_VOCAB_SIZE)
        batch_size = kwargs.get("batch_size", 32)
        self.num_layers = kwargs.get("num_layers", 1)
        self.hidden_dim = hidden_dim
        self.hidden = self.init_hidden(batch_size=batch_size)

        self.lstm = nn.LSTM(input_dim, self.hidden_dim)
        self.pitch = nn.Linear(self.hidden_dim, vocab_size)
        nn.init.kaiming_normal_(self.pitch.weight)
        self.rhythm = nn.Linear(self.hidden_dim, 1)
        nn.init.kaiming_normal_(self.rhythm.weight)

    def init_hidden(self, batch_size=32):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim))

    def forward(self, X):
        # adjust the batch size according to the input
        input_batch = X.shape[1]
        hidden_batch = self.hidden[0].shape[1]
        if input_batch != hidden_batch:
            self.hidden = self.init_hidden(input_batch)

        # lstm.shape == [seq_len, batch_size, hidden_size]
        lstm, self.hidden = self.lstm(X, self.hidden)
        last_step = lstm[-1] # get only output from last time step
        pitch_scores = self.pitch(last_step)
        rhythm_scores = self.rhythm(last_step)
        return pitch_scores, torch.sigmoid(rhythm_scores)

def assert_pitch_embed_model():
    input_tensor = torch.randint(0, 88, (BATCH_SIZE,)).long()
    embed = PitchEmbedModel()
    output = embed(input_tensor)
    assert output.shape == (BATCH_SIZE, EMBED_DIM), "Invalid output shape."

def assert_harmony_model():
    num_parts = 4
    input_shape= (num_parts, EMBED_DIM + 4)
    output_dim = PITCH_VOCAB_SIZE + 4

    input_tensor = torch.randint(0,88,(BATCH_SIZE,)+input_shape)
    harmony = HarmonyModel(input_shape, PITCH_VOCAB_SIZE)
    output = harmony(input_tensor)
    assert output.shape == (BATCH_SIZE, output_dim), \
        "Invalid output shape."

def assert_judge_model():
    input_dim = (3, EMBED_DIM)
    hidden_dim = 64
    output_dim = PITCH_VOCAB_SIZE

    input_tensor = torch.randint(0,88,(BATCH_SIZE,)+input_dim)
    harmony = JudgeModel(input_dim, hidden_dim, output_dim)
    output = harmony(input_tensor)

    assert output.shape == (BATCH_SIZE, PITCH_VOCAB_SIZE), \
                "Invalid output shape."

def assert_note_model():
    pitch_dim = PITCH_VOCAB_SIZE
    rhythm_dim = 1
    input_dim = EMBED_DIM + rhythm_dim
    seq_len, hidden_dim = 32, 64
    lower, upper = 0, 88 # bounds of the pitches

    input_tensor = torch.randint(lower,upper,(seq_len,BATCH_SIZE,input_dim))
    forward_pitch = NoteModel(input_dim, hidden_dim, vocab_size=pitch_dim)
    pitches, rhythms = forward_pitch(input_tensor)

    assert pitches.shape == (BATCH_SIZE, pitch_dim), \
            "Invalid output shape."
    assert rhythms.shape == (BATCH_SIZE, rhythm_dim), "Invalid output shape."

if __name__ == "__main__":
    assert_harmony_model()
    assert_judge_model()
    assert_pitch_embed_model()
    assert_note_model()
