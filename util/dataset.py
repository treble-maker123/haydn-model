import os
import sys
import traceback
import numpy as np
from time import time
from math import ceil
from pdb import set_trace, pm
from mido import MidiFile
from functools import reduce
from music21 import converter, instrument, clef
from music21.chord import Chord
from music21.note import Note, Rest
from music21.stream import Score, Part
from multiprocessing import Pool
from torch.utils.data import Dataset
import torchvision.transforms as T

DATA_DIR = "data"  # data folder name
CHECK_ERROR = True  # Will run all of the assert statements in the code
TEST_MODE = False # won't load as many data
TESTING_SAMPLE = 3

if TEST_MODE:
    print("WARN: test mode is ON.")

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
    if TEST_MODE: file_names = file_names[:TESTING_SAMPLE]

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
    def __init__(self, **kwargs):
        start = time()
        print("Building dataset...")
        # with quarter length being 1.0, 0.25 is the length of a 16th note in
        # music21
        self.unit_length = 0.25

        # call the load_data method if no specific data is passed in.
        data = kwargs.get("data", load_data())
        self._scores = data[:TESTING_SAMPLE] if TEST_MODE else data

        # setting a larger size than actual range
        self.pitch_vocab_size = kwargs.get("pitch_vocab_size", 120)
        # value of the rest pitch, should be 119 by default
        self.rest = self.pitch_vocab_size - 1

    def __len__(self):
        return len(self._scores)

    def __getitem__(self, idx):
        if len(self._scores) > 0:
            return self._score_to_matrix(self._scores[idx])
        else:
            return None

    def _calc_total_len(self, part):
        '''
        Calculate the total length of the part, length is fraction of quarter note time.
        '''
        return sum(map(lambda n: n.duration.quarterLength, part.notesAndRests))

    def _midi_to_input(self, note_or_chord):
        if type(note_or_chord) is Chord:
            # return the highest note
            return max(note_or_chord).pitch.midi
        elif type(note_or_chord) is Note:
            # return the note directly
            return note_or_chord.pitch.midi
        elif type(note_or_chord) is Rest:
            return self.rest
        else:
            raise("Unrecognizablel input for note_or_chord to _midi_to_input")

    def _score_to_matrix(self, score, verbose=False):
        '''
        Transform the files in score_files into tensors.

        Args:
            score (music21.stream.Score)

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
        # 3rd dim - 2, 1 for midi pitch and 1 for articulation
        output_dim = (4, ticks, 2)
        # final state matrix
        state = np.zeros(output_dim, dtype=np.int8)

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
            note_pitches = list(map(self._midi_to_input, cur_notes))
            state[:, current_tick, 0] = note_pitches
            # articulation
            state[:, current_tick, 1][notes_need_refresh] = 1

            # TODO: MORE ELEMENTS HERE

            # decrease the leftover duration of all of the notes
            cur_dur -= self.unit_length

        if state.shape[1] == 0:  # no ticks in the dataset
            return None

        if CHECK_ERROR:
            # articulations for first tick should always be on
            assert state[:, 0, -1].sum() == 4.0, \
                "Not starting with articulation."

        # transpose up and down half an octave in each direction
        transposed = self._transpose_score(state)

        return transposed

    def _transpose_score(self, state):
        return state[None, :, :, :] # no transposition

        result = np.zeros((13,) + state.shape)
        for step in range(-6, 7):  # 7 because range end is exclusive
            pitches = state[:, :, 0].copy()
            # determine whether the vector contains a pitch, if it's a rest
            # then the sum of the vector would be 0
            has_pitch = pitches < self.rest
            # transpose the pitches
            transposed = (pitches + step) * has_pitch
            # add the rests back
            transposed += ~has_pitch * self.rest
            # move the pitches into result
            result[step+6,:,:,0] = transposed
            # move the rhythms into result
            result[step+6,:,:,1] = state[:,:,1]
        return result

    def matrix_to_score(self, matrix, verbose=False):
        '''
        Takes a matrix of (P, T, 2) and turn it into a music21.stream.Score object, where P is the number of parts, T is the number of time slices, and dim is the note vector.
        '''
        # (4 parts, # ticks, 2)
        assert len(matrix.shape) == 3, \
            "Input matrix needs to have 3-dimensions."

        num_parts, num_ticks, num_dim = matrix.shape
        assert num_parts == 4, "Input matrix needs to have 4 parts."
        assert num_ticks > 0, "No time slices in this matrix."
        assert num_dim == 2, "Note vector size mismatch."

        # need to make sure all pieces start with an articulated note, even if
        # it's a rest.
        matrix[:, 0, 1] = [1,1,1,1]

        score = Score()
        parts = list(map(self._matrix_to_part, matrix))

        parts[0].insert(0, instrument.Violin())
        parts[0].partName = "Violin I"
        parts[0].clef = clef.TrebleClef()

        parts[1].insert(0, instrument.Violin())
        parts[1].partName = "Violin II"
        parts[1].clef = clef.TrebleClef()

        parts[2].insert(0, instrument.Viola())
        parts[2].clef = clef.AltoClef()

        parts[3].insert(0, instrument.Violoncello())
        parts[3].clef = clef.BassClef()
        _ = list(map(lambda part: score.append(part), parts))

        return score

    def _matrix_to_part(self, submatrix):
        '''
        Takes a submatrix of size (T, D) and turn it into a music21.stream.Part
        object, where T is the number of time slices, and dim is the note
        vector.
        '''
        part = Part()
        pitches = submatrix[:, 0]
        articulations = submatrix[:, 1]

        current_note = None
        for current_tick in range(len(submatrix)):
            if articulations[current_tick]:  # if articulate
                # append the old note
                if current_note is not None:  # for the first note
                    part.append(current_note)

                # create a new note
                if pitches[current_tick] < self.rest:
                    current_note = Note()
                    # assign pitch, inverse of self._midi_to_input()
                    current_note.pitch.midi = pitches[current_tick]
                else:
                    current_note = Rest()
                # resets the duration to the smallest amount
                current_note.duration.quarterLength = self.unit_length
            else:
                current_note.duration.quarterLength += self.unit_length

        return part

class ChunksDataset(Dataset):
    def __init__(self, transforms=None, **kwargs):
        # length of each sequence or "ticks"
        self.seq_len = kwargs.get("seq_len", 32)
        # number of time steps to slide for each chunk
        self.stride = kwargs.get("stride", 1)
        # length of the sliding window
        # seq_len x 2 + 1 because bidirecitonal (x2) and prediction (+1)
        self.win_len = self.seq_len * 2 + 1
        # sort the dataset out
        self.dataset = kwargs.get("dataset", None)
        # placeholder for complementary set
        self.comp_set = None

        if self.dataset is None:
            print("WARN: should pass in a dataset= argument for ChunksDataset to avoid duplicate instantiation of HaydnDataset.")
            self.dataset = HaydnDataset()

        # if the whole HaydnDataset is passed in, process and split it
        if type(self.dataset).__name__ == "HaydnDataset":
            # filter out the blank/invalid datasets
            dataset = list(filter(lambda ds: ds is not None, self.dataset))
            # train gets 1 - val_split of the data, and val gets val_split data
            mode = kwargs.get("mode", "train")
            assert mode in ["train", "val", "all"], "Invalid mode."

            if mode == "all":
                self.dataset = dataset
            else:
                # percentage of data going to validations
                val_split = kwargs.get("val_split", 0.2)
                assert val_split < 1.0, "Invalid validation split value."

                num_datasets = len(dataset)
                # number of items in this dataset
                set_size = round(num_datasets * val_split) \
                                if mode == "val" \
                                else round(num_datasets * (1 - val_split))
                all_idx = np.linspace(0, num_datasets, num_datasets,
                                      endpoint=False, dtype=np.int8)
                np.random.shuffle(all_idx)
                # random index for this set
                set_idx = all_idx[:set_size]
                # complementary index, in all_idx but not set_idx
                comp_set_idx = np.setdiff1d(all_idx, set_idx).astype("int8")
                # update this set
                self.dataset = [ dataset[idx] for idx in set_idx ]
                # complementary set
                self.comp_set = [ dataset[idx] for idx in comp_set_idx ]
        elif type(self.dataset) is list:
            # already in self.dataset
            pass
        else:
            raise Exception("Unrecognized chunk data type!")

        # number of chunks each piece has, each chunk is seq_len+1 long
        # +1 for validation, the network should take in seq_len values
        def calc_chunks(data, win_len=self.win_len, stride=self.stride):
            num_ticks = data.shape[2]
            return (num_ticks - win_len) // stride + 1
        self.num_chunks = list(map(calc_chunks, self.dataset))
        # total number of chunks
        self.total_chunks = sum(self.num_chunks)

        if transforms is not None:
            self.transforms = T.Compose(transforms)
        else:
            self.transforms = None

    def __len__(self):
        return self.total_chunks

    def __getitem__(self, idx):
        # [ num_chunks ], [ num_chunks ]... x num_transpose
        transpose_idx = idx // self.total_chunks
        chunk_idx = idx % self.total_chunks
        for corpus_idx in range(len(self.dataset)):
            # go through the chunks in each corpus, if the chunk index is
            # greater than the chunks in the corpus, go to the next corpus.
            if chunk_idx > self.num_chunks[corpus_idx]:
                chunk_idx -= self.num_chunks[corpus_idx]
            else:
                corpus = self.dataset[corpus_idx]

                # exclude the first chunk and last chunk
                start_tick_idx = chunk_idx * self.stride
                end_tick_idx = chunk_idx * self.stride + self.win_len
                cidx = np.s_[transpose_idx,:,start_tick_idx:end_tick_idx,:]
                chunk = corpus[cidx]

                # pad it if it's not long enough
                if chunk.shape[1] < self.win_len:
                    diff = self.win_len - chunk.shape[1]
                    chunk = np.pad(chunk,((0,0), (0,diff), (0,0)),
                                   mode="constant")

                if self.transforms is not None:
                    chunk = self.transforms(chunk)

                return chunk

def assert_correct_sizes():
    dataset = HaydnDataset()
    chunks = ChunksDataset(dataset=dataset, mode="train", val_split=0.2)
    comp_chunks = ChunksDataset(dataset=chunks.comp_set)
    all_chunks = ChunksDataset(dataset=dataset, mode="all")
    seq_len = all_chunks.seq_len
    assert len(chunks) != 0, "Empty chunks!"
    assert len(comp_chunks) != 0, "Empty complementary chunks!"
    # seq_len x 2 + 1, x 2 for bidirectional, + 1 for the target note.
    assert chunks[5].shape == (4, (seq_len*2+1), 2), "Invalid chunk shape."
    # make sure last chunk exists
    assert all_chunks[len(all_chunks)-1] is not None, "Invalid last chunk"
    # make sure all chunks have three dims
    for chunk in all_chunks:
        assert len(chunk.shape) == 3, "Invalid chunk!"
    # make sure all chunks are of length seq_len
    lens = list(map(lambda x: x.shape[1], all_chunks))
    parts = list(map(lambda x: x.shape[2], all_chunks))
    wrong_lens = [ x for x in lens if x != (seq_len*2+1) ]
    wrong_parts = [ x for x in parts if x != 2]
    assert len(wrong_lens) == 0, "There are {} chunks with wrong lengths." \
                                    .format(len(wrong_lens))
    assert len(wrong_parts) == 0, \
        "There are {} chunks with wrong parts.".format(len(wrong_parts))

if __name__ == "__main__":
    # dataset = HaydnDataset()
    # data = dataset[2]
    # score = dataset.matrix_to_score(dataset[2][6,:,:,:])
    # score.show()

    assert_correct_sizes()
    pass
