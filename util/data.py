import os
import numpy as np
from time import time
from math import ceil
from pdb import set_trace
from mido import MidiFile
from functools import reduce
from music21 import converter
from music21.chord import Chord
from music21.note import Note, Rest
from multiprocessing import Pool
from torch.utils.data import Dataset

DATA_DIR = "data" # data folder name

file_names = []
file_paths = []
midi_files = []
score_files = []

def load_data_paths(fns):
  assert len(fns) > 0
  return list(map(lambda fn: DATA_DIR + "/" + fn, fns))

def load_data(reload_data=False):
  '''
  Some bad hombres in the mix, filtering out the tracks that are corrupted or
  have invalid number of tracks. Save the filtered file paths in the file_paths
  global variable.

  Args:
    reload_data (bool): Forces the data to be reloaded instead of loading from pgz file when possible.

  Returns:
    List: score_files.
  '''
  global file_names, file_paths, midi_files, score_files
  # get files ending in ".mid"
  file_names = list(filter(lambda fn: fn[-3:]=="pgz", os.listdir(DATA_DIR)))

  if not reload_data and len(file_names) > 0:
    start = time()
    print("Serialized scores found, loading...")
    file_paths = load_data_paths(file_names)
    score_files = list(map(converter.thaw, file_paths))
    print("Scores loaded in {:.2f} seconds.".format(time() - start))
    return score_files
  else:
    print("No serialized data found, loading data from source.")

  # get files ending in ".mid"
  file_names = list(filter(lambda fn: fn[-3:]=="mid", os.listdir(DATA_DIR)))
  # assemble file paths for easy access
  file_paths = load_data_paths(file_names)
  if len(file_paths) == 0: raise Exception("No midi files found!")

  def convert_midi(file_path):
    try:
      return MidiFile(file_path)
    except Exception as error:
      print(error)
      return None

  start = time()
  print("Converting files into midi.")
  midis = list(filter(lambda m: m != None, map(convert_midi, file_paths)))
  print("Converting files into midis took {:.2f} seconds."
          .format(time() - start))

  # expecting track 1, 2, 3, 4 (index starts at 0) to contain the parts
  def correct_format(midi):
    key = "Viol"
    return key in midi.tracks[1].name and \
           key in midi.tracks[2].name and \
           key in midi.tracks[3].name and \
           key in midi.tracks[4].name
  midis = list(filter(correct_format, midis))
  # save the results
  midi_files = midis
  file_paths = list(map(lambda m: m.filename, midis))
  file_names = list(map(lambda fp: fp.split("/")[1], file_paths))

  # get the scores
  start = time()
  print("Converting files into music21 Scores.")
  with Pool(max(1, os.cpu_count() - 2)) as pool:
    score_files = pool.map(converter.parse, file_paths)
  score_files = list(filter(lambda sf: len(sf.parts) == 4, score_files))
  print("Converting files into music21 Scores took {:.2f} seconds."
          .format(time() - start))

  output_paths = list(map(
    lambda f: os.getcwd()+"/"+DATA_DIR+"/"+str(f.id)+".pgz", score_files))
  bundle = list(zip(score_files, output_paths))
  for file, path in bundle:
    converter.freeze(file, fp=path)

  return score_files

class HaydnDataset(Dataset):
  def __init__(self, data=None):
    start = time()
    print("Building dataset...")
    self.unit_length = 0.25

    self._scores = data

    # figure out pitch bounds
    pitches = list(map(self._score_to_pitches, self._scores))
    max_pitch = reduce(max, reduce(max, reduce(max, pitches)))
    min_pitch = reduce(min, reduce(min, reduce(min, pitches)))
    # +6 to allow room for transposition
    self._upper_bound = max_pitch + 6
    self._lower_bound = min_pitch - 6
    # +1 for lowest note, and +1 for rest
    self._pitch_span = self._upper_bound - self._lower_bound + 2
    print("Finished building dataset in {:.2f} seconds.".format(time() - start))

  def __len__(self):
    return len(self._scores)

  def __getitem__(self, idx):
    if len(self._scores) > 0:
      return self._score_to_matrix(self._scores[idx])
    else:
      return None

  def _score_to_pitches(self, score):
    '''
    Turns a score into Lists of MIDI pitches.
    '''
    return list(map(self._part_to_pitches, score))

  def _part_to_pitches(self, part):
    return list(
            map(lambda el:
              el.pitch.midi if type(el) is Note else max(el).pitch.midi,
                part.flat.notes))

  def _calc_total_len(self, part):
    '''
    Calculate the total length of the part, length is fraction of quarter note time.
    '''
    return sum(map(lambda n: n.duration.quarterLength, part.notesAndRests))

  def _get_midi_value(self, note_or_chord):
    if type(note_or_chord) is Chord:
      # return the highest note
      return max(note_or_chord).pitch.midi - self._lower_bound
    elif type(note_or_chord) is Note:
      return note_or_chord.pitch.midi - self._lower_bound
    else:
      # last unit
      return self._pitch_span - 1

  def _score_to_matrix(self, score, verbose=False):
      '''
      Transform the files in score_files into tensors.

      Args:
          idx (int): The index of the score in score_files to turn into a state matrix.

      Returns:
          numpy.Array
      '''
      parts = score.parts
      assert len(parts) == 4, "Data does not have 4 parts."

      # total quarter length of the parts
      lengths = list(map(self._calc_total_len, parts))
      max_len = max(lengths)

      # current notes and their indices
      cur_notes = []
      cur_note_idx = np.array([-1] * 4)
      # total number of notes and rests
      # total_notes = list(map(lambda part: len(part.notesAndRests), parts))
      # keeping track of leftover duration of the current note
      cur_dur = np.array([0.0] * 4)

      # how many total ticks
      ticks = ceil(max_len / self.unit_length)
      # output dimension
      # 1st dim - num of parts
      # 2nd dim - range of notes, rest, articulated
      output_dim = (4, ticks, self._pitch_span + 1)
      # final state matrix
      state = np.zeros(output_dim, dtype=np.float16)

      for current_tick in range(ticks):
        # check which note has run out of duration
        notes_need_refresh = np.squeeze(
          np.argwhere(cur_dur < self.unit_length), axis=1)
        # articulate if is a new note
        articulated = np.array([0] * 4)[notes_need_refresh] + 1

        if len(notes_need_refresh) > 0:
          # update the note idx for the parts that have new notes
          cur_note_idx[notes_need_refresh] += 1
          # if there are no next notes, probably inconsistency in the score, break out early
          # and not worry about the rest
          try:
            cur_notes = \
              [part.notesAndRests[idx] for part,idx in zip(parts,cur_note_idx)]
            # duration of all of the notes
            durations = list(map(
              lambda el: el.duration.quarterLength, cur_notes))
            # update the leftover duration of notes from parts that were just updated
            cur_dur[notes_need_refresh] = \
              np.array(durations)[notes_need_refresh]
          except Exception as error:
            if verbose: print(error)
            # The parts aren't usually of the same length, so terminate once we
            # go through the shortest one and slice out the empty time columns.
            state = state[:, :current_tick, :]
            break

        # pitches
        note_pitches = list(map(self._get_midi_value, cur_notes))
        state[:, current_tick, note_pitches] = 1
        # articulation
        state[:, current_tick, -1][articulated] = 1
        # TODO: MORE ELEMENTS HERE

        cur_dur -= self.unit_length
      return state
