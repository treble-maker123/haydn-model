import torch
import numpy as np
from pdb import set_trace
from functools import reduce

from util.run import forward_pass

def sample(models, **kwargs):
    '''
    Give the number of ticks (each tick is the length of an eighth note), use the model to populate the music.
    '''
    num_parts = kwargs.get("num_parts", 4)
    # total number of time slices, default 8 measures
    num_ticks = kwargs.get("num_ticks", 128)
    # number of pitches
    vocab_size = kwargs.get("vocab_size", 140)
    # sequence length that the network takes in, default 2 measures
    seq_len = kwargs.get("seq_len", 32)
    num_repeats = kwargs.get("num_repeats", 1)

    part_nums = [0, 1, 2, 3]

    for key in models:
        models[key].eval()

    result = np.zeros((num_parts, num_ticks, 2))

    # initialize all of the parts
    for part_num in part_nums:
        result[part_num, :, :] = _populate_part(num_ticks, vocab_size)

    # go back and revise it
    num_cells = reduce(lambda x, y: x*y, result.shape[:-1])
    # number of "nodes" to revise
    max_iters = num_cells * num_repeats
    # list of indices to revise
    part_indices = np.random.randint(0, num_parts, max_iters)
    tick_indices = np.random.randint(0+seq_len, num_ticks-seq_len-1, max_iters)

    # start revising
    for iter_i, (i_part, i_tick) in enumerate(zip(part_indices, tick_indices)):
        with torch.no_grad():
            start_idx = i_tick - seq_len
            end_idx = i_tick + seq_len + 1
            center_idx = (start_idx - end_idx) // 2 + start_idx
            chunk = result[:,start_idx:end_idx,:]
            chunk_tensor = torch.Tensor(chunk.reshape((1,) + chunk.shape))

            _, forward_rhythm, _, _, _, judge_decision \
                = forward_pass(models, chunk_tensor, i_part)

            rhythm = (forward_rhythm > 0.5).item()
            pitch = judge_decision.squeeze(dim=0).argmax().item()
            result[i_part, center_idx, :] = [pitch, rhythm]

        if (iter_i % 100) == 0:
            print("current iter: {x}/{y}".format(x=iter_i, y=max_iters))

    return result


def _populate_part(num_ticks, vocab_size):
    '''
    Initial run through the part to populate the part, with only pitch_next and rhythm_next layer.
    '''
    part = np.zeros((num_ticks, 2))
    pitches = np.random.randint(0, vocab_size, num_ticks)
    rhythm = np.random.randint(0, 2, num_ticks)
    part[:, 0] = pitches
    part[:, 1] = rhythm

    return part
