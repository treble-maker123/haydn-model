import os
from torch.utils.data import Dataset
from music21 import converter
from music21.stream import Score
from time import time
from multiprocessing import Pool
from preprocess import Transform, ToTensorWithoutRhythm

class HaydnQuartetRawDataset(Dataset):
  '''
  A class for interacting with the dataset. All preprocessing classes can inherit this class.
  '''

  def __init__(self, **kwargs):
    transform = kwargs.get('transform', None)
    filtered = kwargs.get('filtered', True)
    self.data_dir = kwargs.get('data_dir', "data/scores")
    self.score_names = os.listdir(self.data_dir)
    if transform.__bases__[0] is Transform: self.transform = transform()
    if len(self.score_names) == 0: raise Exception("No scores found!")
    if filtered: self.__filter_scores__()

  def __len__(self):
    return len(self.score_names)

  def __getitem__(self, idx):
    score = converter.thaw(self.__get_path__(self.score_names[idx]))
    if self.transform: score = self.transform(score)
    return score

  def __filter_scores__(self, multi_proc=True):
    start = time()
    print("Filtering by number of parts...")
    if multi_proc:
      num_proc = max(1, os.cpu_count() - 1)
      pool = Pool(num_proc)
      output = pool.map(self.__has_four_parts__, self.score_names)
      pool.close()
      pool.join()
      score_names = [ score for four_parts, score in output if four_parts]
      self.score_names = score_names
    else:
      need_four_parts = filter(self.__has_four_parts__, self.score_names)
      self.score_names = list(need_four_parts)
    print("Took {:.2f} seconds.".format(time() - start))

  def __get_path__(self, fn):
    return self.data_dir + "/" + fn

  def __has_four_parts__(self, score_or_fn):
    '''
    Checks if the score has four parts or not. Some of them only have 4.
    Optionally adds the score to the queue.
    '''
    num_parts = 4
    if type(score_or_fn) is Score:
      result = len(score_or_fn) == num_parts
    else:
      path = self.__get_path__(score_or_fn)
      result = len(converter.thaw(path)) == num_parts
    return result, score_or_fn

if __name__ == '__main__':
  dataset = HaydnQuartetRawDataset(transform=ToTensorWithoutRhythm)
