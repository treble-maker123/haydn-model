import os
import json
from time import time
from torch.utils.data import Dataset
from music21 import converter
from music21.stream import Score
from multiprocessing import Pool
from preprocess import DataTransform

class HaydnQuartetRawDataset(Dataset):
  '''
  A class for interacting with the Haydn string quartet dataset.
  '''

  def __init__(self, **kwargs):
    self.transform = kwargs.get('transform', None)
    self.analysis = kwargs.get('analysis', None)
    self.analysis_cache = {}

    # check that there are data
    self.data_dir = kwargs.get('data_dir', "data/scores")
    self.score_names = os.listdir(self.data_dir)
    if len(self.score_names) == 0: raise Exception("No scores found!")

    # filtering scores
    filtered = kwargs.get('filtered', True)
    if filtered: self._filter_scores()

    # apply the analysis object, needs to be before transform
    if self.analysis and self.analysis.__name__ is "MusicAnalysis":
      self.analysis = self.analysis(self)
      self.analysis_cache = self.analysis.cache

    # saves the Transform object
    if self.transform and self.transform.__bases__[0] is DataTransform:
      self.transform = self.transform() # used in __getitem__()

  def __len__(self):
    return len(self.score_names)

  def __getitem__(self, idx):
    score = converter.thaw(self._get_path(self.score_names[idx]))
    # is transforming and has initalized
    if self.transform and not type(self.transform) == type:
      score = self.transform(score, self.analysis_cache)
    return score

  def _filter_scores(self, verbose=True):
    '''
    Filter the socres in self.score_names by criteria and store the filtered scores list in self.score_names

    Args:
      multi_proc (bool): Whether to use multiprocessing.
    '''
    if verbose:
      start = time()
      print("Filtering by number of parts...")
    pool = Pool(max(1, os.cpu_count() - 1))
    output = pool.map(self._has_four_parts, self.score_names)
    pool.close()
    pool.join()
    score_names = [ score for four_parts, score in output if four_parts]
    self.score_names = score_names

    if verbose: print("Took {:.2f} seconds.".format(time() - start))

  def _get_path(self, fn):
    '''
    Return the full path of the file given a file name.

    Args:
      fn (string): File name.
    '''
    return self.data_dir + "/" + fn

  def _has_four_parts(self, score_or_fn):
    '''
    Checks if the score has four parts or not. Some of them only have 4.
    Optionally adds the score to the queue.

    Args:
      score_or_fn (music21.stream.Score or string): The Score object or the file name of the score to be checked.

    Returns:
      bool, music21.stream.Score or string: A tuple of boolean specifying whether the particular score has four parts, as well as the input.
    '''
    num_parts = 4
    if type(score_or_fn) is Score:
      result = len(score_or_fn) == num_parts
    else:
      path = self._get_path(score_or_fn)
      result = len(converter.thaw(path)) == num_parts

    return result, score_or_fn
