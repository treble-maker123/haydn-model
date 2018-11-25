import os
from mido import MidiFile

DATA_DIR = "data"


def load_data_paths(fns):
    assert len(fns) > 0
    return list(map(lambda fn: DATA_DIR + "/" + fn, fns))


def convert_midi(file_path):
    try:
        return MidiFile(file_path)
    except Exception as error:
        print(error)
        return None


def incorrect_format(midi):
    key = "Viol"
    return key not in midi.tracks[1].name or \
        key not in midi.tracks[2].name or \
        key not in midi.tracks[3].name or \
        key not in midi.tracks[4].name


def rename_tracks(midi):
    tracks = midi.tracks
    tracks[1].name = "Violin I."
    tracks[2].name = "Violin II."
    tracks[3].name = "Viola."
    tracks[4].name = "Violoncello"
    midi.save(midi.filename)


def mark_as_erraneous(midi):
    # marks a file as erraneous by appending ".error" at the end of the file name
    midi.save(midi.filename + ".error")


files = os.listdir(DATA_DIR)
files = file_names = list(filter(lambda fn: fn[-3:] == "mid", files))
file_paths = load_data_paths(file_names)
midis = list(filter(lambda m: m != None, map(convert_midi, file_paths)))
erraneous = list(filter(incorrect_format, midis))

# doing the code below manually
# erraneous[0].tracks to examine tracks
# erraneous.pop(0) to remove tracks
# rename_tracks to rename tracks with incorrect names

list(map(mark_as_erraneous, erraneous))
