import torch
import numpy as np
from analysis import MusicAnalysis

class DataTransform(object):
  def __call__(self, **kwargs):
    raise Exception("Do not use Transform directly, instead subclass from this class.")

class DataToTensor16thMin(DataTransform):
  def __call__(self, sample, analysis={}):
    '''
    '''
    lower_bound, upper_bound = \
      analysis.get("lower_bound", 12), analysis.get("upper_bound", 100)
    import pdb; pdb.set_trace()

class DataToTensorWithoutRhythm(DataTransform):
  def __call__(self, sample):
    '''
    Converts the sample clip to a simple tensor that contain only pitches and no duration.
    '''
    # allowed pitch range, all pitches beyond range is clipped.
    _, upper_bound = 12, 100
    max_len = max(len(sample.parts[0].notes),
                  len(sample.parts[1].notes),
                  len(sample.parts[2].notes),
                  len(sample.parts[3].notes))
    parts = [np.zeros((max_len, upper_bound), dtype=np.int8),
             np.zeros((max_len, upper_bound), dtype=np.int8),
             np.zeros((max_len, upper_bound), dtype=np.int8),
             np.zeros((max_len, upper_bound), dtype=np.int8)]

    for idx, part in enumerate(sample.parts):
      pitches = list(map(lambda pitch: pitch.ps, part.notes.pitches))
      pitches = [ pitch if not pitch >= upper_bound else pitch - 12
                    for pitch in pitches ]
      pitches = list(map(int, pitches))
      parts[idx][pitches] = 1

    return torch.HalfTensor(parts, dtype=np.float16)
