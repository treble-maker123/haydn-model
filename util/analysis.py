import os
import numpy as np
from time import time
from functools import reduce
from multiprocessing import Pool

class MusicAnalysis:
  def __init__(self, dataset, **kwargs):
    self.cache = {
      "lower_bound": None,
      "upper_bound": None
    }
    self._dataset = dataset
    self._compute_min_max_pitch()

  def to_pitch_steps(self, verbose=True):
    '''
    Turns the dataset into a nested List of pitch steps. The Lists have the
    dimensions (# of pieces) x (# of parts) x (# of notes).

    Returns:
      List: A multi-dimensional of List containing the pitch steps.
    '''
    return list(map(self._score_to_pitch_steps, self._dataset))

  def _compute_min_max_pitch(self, verbose=True):
    '''
    Computes the min and the max pitch step in the dataset.

    Returns:
      float, float: A tuple containing the minimum and maximum pitch step, in
      that order.
    '''
    # Default for Haydn quartets
    self.cache["lower_bound"], self.cache["upper_bound"] = 38.0, 89.0
    return

    if self.cache["lower_bound"] and self.cache["upper_bound"]:
      return self.cache["lower_bound"], self.cache["upper_bound"]

    if verbose:
      start = time()
      print("Finding min/max pitches...")
    pitches = self.to_pitch_steps()
    self.cache["upper_bound"] = reduce(max, reduce(max, reduce(max, pitches)))
    self.cache["lower_bound"] = reduce(min, reduce(min, reduce(min, pitches)))
    if verbose:
      print("Took {:.2f} seconds.".format(time() - start))

  def _score_to_pitch_steps(self, score, verbose=True):
    return list(map(self._part_to_pitch_steps, score.parts))

  def _part_to_pitch_steps(self, part, verbose=True):
    return list(map(lambda x: x.ps, part.pitches))
