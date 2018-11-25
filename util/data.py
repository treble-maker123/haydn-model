import os
import sys
import traceback
import numpy as np
from time import time
from math import ceil
from pdb import set_trace, pm
from mido import MidiFile
from functools import reduce
from music21 import converter
from music21.chord import Chord
from music21.note import Note, Rest
from multiprocessing import Pool
from torch.utils.data import Dataset

DATA_DIR = "data"  # data folder name
CHECK_ERROR = True # Will run all of the assert statements in the code

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
    file_names = list(
        filter(lambda fn: fn[-3:] == "pgz", os.listdir(DATA_DIR)))

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
    file_names = list(
        filter(lambda fn: fn[-3:] == "mid", os.listdir(DATA_DIR)))
    # assemble file paths for easy access
    file_paths = load_data_paths(file_names)
    if len(file_paths) == 0:
        raise Exception("No midi files found!")

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
        self._max_pitch = max([max(max(part)) for part in pitches])
        self._min_pitch = min([min(min(part)) for part in pitches])
        # +6 to allow room for transposition
        self._upper_bound = self._max_pitch + 6
        self._lower_bound = self._min_pitch - 6
        # +1 for lowest note, +1 for rest
        self._pitch_span = self._upper_bound - self._lower_bound + 2
        print("Finished building dataset in {:.2f} seconds.".format(
            time() - start))

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
            return max(note_or_chord).pitch.midi - self._min_pitch + 1
        elif type(note_or_chord) is Note:
            # return the note directly
            return note_or_chord.pitch.midi - self._min_pitch + 1
        elif type(note_or_chord) is Rest:
            # return pitch_span - 1, which is reserved for rest
            return self._pitch_span - 1
        else:
            raise("Unrecognizablel input for note_or_chord to _get_midi_value")

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
        # keeping track of leftover duration of the current note
        cur_dur = np.array([0.0] * 4)

        # how many total ticks
        ticks = ceil(max_len / self.unit_length)
        # output dimension
        # 1st dim - num of parts
        # 2nd dim - number of time slices or "ticks"
        # 3rd dim - range of notes, +1 for articulation
        output_dim = (4, ticks, self._pitch_span + 1)
        # final state matrix
        state = np.zeros(output_dim, dtype=np.float16)

        for current_tick in range(ticks):
            # check which note has run out of duration
            notes_need_refresh = np.squeeze(
                np.argwhere(cur_dur < self.unit_length), axis=1)

            if len(notes_need_refresh) > 0:
                # update the note idx for the parts that have new notes
                cur_note_idx[notes_need_refresh] += 1
                # if there are no next notes, probably inconsistency in the
                # score, break out early and not worry about the rest
                try:
                    cur_notes = \
                        [part.notesAndRests[idx]
                            for part, idx in zip(parts, cur_note_idx)]

                    # duration of all of the notes
                    durations = list(map(
                        lambda el: el.duration.quarterLength, cur_notes))

                    # update the leftover duration of notes from parts that
                    # were just updated
                    cur_dur[notes_need_refresh] = \
                        np.array(durations)[notes_need_refresh]
                except Exception as error:
                    if verbose:
                        print(error)
                    # The parts aren't usually of the same length, so terminate
                    # once we go through the shortest one and slice out the
                    # empty time columns.
                    state = state[:, :current_tick, :]
                    break

            # pitch assignment
            note_pitches = list(map(self._get_midi_value, cur_notes))
            state[:, current_tick][np.arange(4), note_pitches] = 1
            if CHECK_ERROR:
                # checks assignment is correct
                idx = state[:, current_tick, :self._pitch_span].argmax(axis=1)
                assert (idx == note_pitches).sum() == 4, \
                    "Invalid pitch assignment."

            # articulation
            state[:, current_tick, -1][notes_need_refresh] = 1
            # remove articulation for rests
            rests_idx = [ idx for idx, note in enumerate(cur_notes)
                            if type(note) is Rest]
            state[:, current_tick, -1][rests_idx] = 0

            # TODO: MORE ELEMENTS HERE

            # decrease the leftover duration of all of the notes
            cur_dur -= self.unit_length

        if CHECK_ERROR:
            # each of the time slice dimensions should not sum to greater than 2
            assert (state.sum(axis=2) > 2).sum() <= 0, \
                    "Dimensions with sum greater than 2."

        if state.shape[1] == 0:  # no ticks in the dataset
            return None

        # transpose up and down half an octave in each direction
        transposed = self._transpose_score(state)

        return transposed

    def _transpose_score(self, state):
        transposed = np.zeros((13,) + state.shape)
        # number of ticks from all four parts
        num_ticks = reduce(lambda x,y: x*y, state.shape[:2])
        # number of array elements in state
        total_elements = reduce(lambda x,y: x*y, state.shape)

        try:
            for step in range(-6, 7):  # 7 because range end is exclusive
                # get all of the one-hot encoded pitches, -1 to exclude rest
                pitches = state[:, :, :self._pitch_span-1].copy()

                # determine whether the vector contains a pitch, if it's a rest
                # then the sum of the vector would be 0
                has_pitch = pitches.sum(axis=2) > 0
                if CHECK_ERROR:
                    # *not* has_pitch should match rest
                    assert (~has_pitch == state[:,:,-2]).sum() == num_ticks, \
                        "Mismatch between rests and empty pitches."

                # turn the one-hot encoded values to midi value
                midi_pitches = pitches.argmax(axis=2)
                if CHECK_ERROR:
                    # highest pitch should be less than self._pitch_span
                    assert midi_pitches.max() < self._pitch_span, \
                        "Max pitch is higher than pitch span."

                # transpose all of the pitches by step, mask out the locations
                # where there weren't pitches with has_pitch
                transposed_pitches = (midi_pitches + step) * has_pitch
                # one-hot encode the pitches
                num_pitch_classes = pitches.shape[2]
                mask = np.repeat(
                    has_pitch[:, :, None], num_pitch_classes, axis=2)
                one_hot = np.eye(num_pitch_classes)[transposed_pitches] * mask
                # save it, step+6 because step starts at -6
                # -1 to exclude rest
                transposed[step+6, :, :, :self._pitch_span-1] = one_hot
                # transfer the rests over
                transposed[step+6, :, :, -2] = state[:, :, -2]
                # transfer the articulations over
                transposed[step+6, :, :, -1] = state[:, :, -1]

                if CHECK_ERROR and step == 0:
                    # tranposition with 0 step should be the same as the state.
                    num_matches = (transposed[step+6] == state).sum()
                    assert num_matches == total_elements, \
                        "Mismatch between original state and transposed state."
        except Exception as error:
            print("Encountered {} at step {}.".format(error, step))
            traceback.print_tb(error.__traceback__)
            set_trace()

        return transposed
