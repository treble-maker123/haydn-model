from pdb import set_trace

def sample_to_tensor(dataset, idx):
  '''
  Turns a sample at the given index into a tensor for training.

  Args:
    dataset (PianoMidiDataset): The input dataset where the sample is drawn
      from.
    idx (int): The index of the sample from the dataset.

  Returns:
    pytorch.Tensor
  '''
  sample = dataset[idx]
  ticks_per_beat = sample.ticks_per_beat
  times = dataset._times[idx]
  chord_encoder = dataset.chord_encoder # chord2vec
  time_num = 0 # numerator in time signature
  time_denom = 0 # denominator in time signature

  # TODO: How do I keep track of time signature update?
  for msg in sample.tracks[0]:
    if msg.type == "time_signature":
      time_num = msg.numerator
      time_denom = msg.denominator
