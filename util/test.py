from dataset import PianoMidiDataset
from transform import sample_to_tensor

dataset = PianoMidiDataset()
tensor = sample_to_tensor(dataset, 0)
