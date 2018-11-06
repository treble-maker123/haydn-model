import os
import numpy as np
from functools import reduce
from time import time
from dataset import HaydnQuartetRawDataset

class MusicAnalysis:
  def __init__(self):
    self.dataset = HaydnQuartetRawDataset()

  def min_max_pitch(self, verbose=True):
    if verbose:
      start = time()
      print("Finding min/max pitches...")
    pitches = self.to_pitch_steps()
    max_pitch = reduce(max, reduce(max, reduce(max, pitches)))
    min_pitch = reduce(min, reduce(min, reduce(min, pitches)))
    if verbose:
      print("Took {:.2f} seconds.".format(time() - start))
    return min_pitch, max_pitch

  def to_pitch_steps(self, verbose=True):
    return list(map(self.__score_to_pitch_steps__, self.dataset))

  def __score_to_pitch_steps__(self, score, verbose=True):
    return list(map(self.__part_to_pitch_steps__, score.parts))

  def __part_to_pitch_steps__(self, part, verbose=True):
    return list(map(lambda x: x.ps, part.pitches))

if __name__ == '__main__':
  analysis = MusicAnalysis()
  result = analysis.min_max_pitch()
