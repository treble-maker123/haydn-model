import os
from torch.utils.data import Dataset
from music21 import converter as conv

class HaydnQuartetDataset(Dataset):
  '''
  A class for interacting with the dataset. All preprocessing classes can inherit this class.
  '''

  def __init__(self, data_dir="data/scores"):
    self.data_dir = data_dir
    self.score_names = os.listdir('data/scores')
    if len(self.score_names) == 0: raise Exception("No scores found!")

  def __len__(self):
    return len(self.score_names)

  def __getitem__(self, idx):
    path = self.data_dir + "/" + self.score_names[idx]
    return conv.thaw(path)
