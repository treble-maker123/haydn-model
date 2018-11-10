import os
import itertools
from time import time
from mido import MidiFile
from music21 import converter
from pdb import set_trace
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool

class PianoMidiDataset(Dataset):
  '''
  A class for interacting with the Piano MIDI dataset from
  http://www.piano-midi.de.
  '''

  def __init__(self, **kwargs):
    self._chords = []
    self._times = []
    self._reduce_mode = "chords"

    start = time()
    print("Loading data...")
    # get a list of paths
    self._data_dir = kwargs.get('data_dir', "data")
    # get a list of all of the file names in /data
    file_paths = os.listdir(self._data_dir)
    # only keep the ones end with "mid"
    file_paths = list(filter(lambda x: x[-3:]=="mid", file_paths))
    if len(file_paths) == 0: raise Exception("No midi files found!")
    # get a list of music21 objects
    # self._dataset = list(map(converter.parse, paths))
    self._dataset = self._get_scores(file_paths)
    print("Loading data took {:.2f} seconds.".format(time() - start))

    self.chord_encoder = LabelEncoder()
    self._build_vocab()

  def __len__(self):
    return len(self._dataset)

  def __getitem__(self, idx):
    return self._dataset[idx]

  def _get_scores(self, paths, verbose=True):
    '''
    Convert midi files in self._data_dir directory into music21.stream.Score objects and serialize them into self._data_dir + "/scores".
    '''
    out_path = self._data_dir + "/scores"
    if not os.path.exists(out_path):
      os.mkdir(out_path)
      print("Created directory \"{}\".".format(out_path))
    else:
      if verbose: print("Directory \"{}\" already exists.".format(out_path))

    file_names = os.listdir(out_path)
    file_names = list(map(lambda x: out_path + "/" + x, file_names))

    scores = []
    pool = Pool(max(1, os.cpu_count() - 1))
    start = time()
    if len(file_names) > 0:
      if verbose: print("Serialized files found, thawing...")
      scores = pool.map(converter.thaw, file_names)
    else:
      if verbose: print("Serialized files not found, converting midi...")
      scores = pool.map(self._process_mid, paths)

    pool.close()
    pool.join()
    print("Took {:.2f} seconds".format(time() - start))
    return scores

  def _build_vocab(self):
    '''
    Returns a list of tuples consist of MIDI note values. Each tuple or "word" is a chord in the music.
    '''
    if not len(self._chords) > 0:
      start = time()
      print("Building chords...")
      self._reduce_mode = "chords"
      # convert dataset to a list of tokens
      tokens = list(map(self._tokenize_score, self._dataset))
      # filter out the None tokens, happens if there's an invalid file
      tokens = list(filter(lambda x: x != None, tokens))
      # merge all of the pieces and remove the duplicate tokens
      chords = list(set(list(itertools.chain.from_iterable(tokens))))
      print("Building chords took {:.2f} seconds.".format(time() - start))
      self._chords = chords
      self.chord_encoder.fit(chords)

    if not len(self._times) > 0:
      start = time()
      print("Finding min ticks in each corpus...")
      self._reduce_mode = "times"
      tokens = list(map(self._tokenize_score, self._dataset))
      times = list(filter(lambda x: x != None, tokens))
      print("Finding min ticks took {:.2f} seconds.".format(time() - start))
      self._times = times

    return self._chords, self._times

  def _tokenize_score(self, score):
    '''
    Tokenize the tracks into a List of tuples containing unique chords from the
    corpus.
    '''
    if score:
      return self._tokenize_parts(score.tracks[1]) \
              + self._tokenize_parts(score.tracks[2])
    else:
      return None

  def _tokenize_parts(self, part):
    '''
    Indentify all of the chords in MIDI and turn them into a tuple.
    '''
    tokens = []
    token = []
    for msg in part:
      if self._reduce_mode == "chords":
        if msg.type == "note_on":
          # if msg.time has a none-zero value, it's the start of a new chord.
          if msg.time != 0:
            tokens.append(self._clean_chord_token(token))
            token = []
          token.append(msg.note)
        elif msg.type == "end_of_track":
          tokens.append(self._clean_chord_token(token))
        else:
          token = []
          continue
      elif self._reduce_mode == "times":
        if msg.type == "note_on":
          if msg.time != 0:
            tokens.append(msg.time)
        else:
          continue

    return tokens

  def _clean_chord_token(self, token):
    '''
    Sort the notes in the token and eliminate duplicates.
    '''
    token = list(set(token))
    token.sort()
    return tuple(token)

  def _process_mid(self, file_name):
    # convert to music21.stream.Score object
    score = converter.parse(self._data_dir + "/" + file_name)
    # replace ".mid" with ".pgz"
    fn = file_name.split(".")[0] + ".pgz"
    out_path = os.getcwd() + "/" + self._data_dir + "/scores/" + fn
    converter.freeze(score, fp=out_path)
    return score
