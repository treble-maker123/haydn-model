import os
from music21 import converter as conv
from load_data import list_downloaded_data
from pdb import set_trace as st

def convert_midi(verbose=True):
  '''
  Convert midi files in `/data/score` directory into music21.stream.Score objects and serialize them into the same directory.

  NOTE: This method should be run from the "cs682-project" directory, because it uses os.getcwd().
  '''
  # checks if data/scores exists, if not, create it.
  path = "data/scores"
  if not os.path.exists(path):
    os.mkdir(path)
    print("Created \"data/scores\" directory.")
  else:
    if verbose: print("\"data/scores\" directory already exists.")

  data = list_downloaded_data()
  total = len(data)
  num_processed = 0
  if verbose:
    print("Processing {} pieces of data.".format(total))

  for filename in data:
    try:
      # converts .mid into a score object.
      score = conv.parse("data/" + filename)
      # replace file extension .mid with .pgz
      fn = filename.split('.')[0] + '.pgz'
      # output to current_path/data/scores/file_name.pgz
      output_fp = os.getcwd() + "/" + path + "/" + fn
      conv.freeze(score, fp=output_fp)
      num_processed += 1
      if verbose:
        print("Processed {}/{}".format(num_processed, total))
    except Exception as error:
      if verbose:
        print("Encountered error {} with {}.".format(error, filename))
