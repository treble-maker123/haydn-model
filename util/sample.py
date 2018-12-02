import numpy as np
from pdb import set_trace
from functools import reduce

# Number of times to repeat the sampling process, this is a multiplier of the
# number of "nodes".
# Example
#   Four parts and 128 ticks gives 512 nodes, num_repeats = 10 will run the
#       sampling process 5120 times.
num_repeats = 10


def sample(**kwargs):
    '''
    Give the number of ticks (each tick is the length of an eighth note), use the model to populate the music.
    '''
    global num_repeats
    num_parts = kwargs.get("num_parts", 4)
    # total number of time slices, default 64 measures
    num_ticks = kwargs.get("num_ticks", 1024)
    num_dims = kwargs.get("num_dims", 73)
    # sequence length that the network takes in, default 1 measures
    seq_len = kwargs.get("seq_len", 32)
    part_nums = [0, 1, 2, 3]

    # models
    models = {}
    for part_num in part_nums:
        models["pitch_next_"+str(part_num)] = \
            kwargs.get("pitch_next_"+str(part_num), None)
        models["pitch_prev_"+str(part_num)] = \
            kwargs.get("pitch_prev_"+str(part_num), None)
        models["harmony_"+str(part_num)] = \
            kwargs.get("harmony_"+str(part_num), None)
        models["pitch_combined_"+str(part_num)] = \
            kwargs.get("combined_"+str(part_num), None)
        models["rhythm_next_"+str(part_num)] = \
            kwargs.get("rhythm_next_"+str(part_num), None)

    for key, val in models.items():
        val.eval()

    result = np.zeros((num_parts, num_ticks, num_dims))

    # initialize all of the parts randomly
    for part_num in part_nums:
        result[part_num, :, :] = _populate_part(num_ticks, num_dims,
                                                part_num, models)

    # go back and revise it
    num_cells = reduce(lambda x, y: x*y, result.shape[:-1])
    # number of "nodes" to revise
    max_iters = num_cells * num_repeats
    # list of indices to revise
    part_indices = np.random.randint(0, num_parts, max_iters)
    tick_indices = np.random.randint(0+seq_len, num_ticks-seq_len, max_iters)
    for iter_i, (i_part, i_tick) in \
            enumerate(zip(part_indices, tick_indices)):
        # prev_seq.shape = (seq_len, num_dims)
        prev_seq = result[i_part, i_tick-seq_len:i_tick, :]
        prev_seq = prev_seq.reshape((seq_len, 1, num_dims)).copy()

        # next_seq.shape = (seq_len, num_dims)
        next_seq = result[i_part, i_tick:i_tick+seq_len, :]
        next_seq = next_seq.reshape((seq_len, 1, num_dims)).copy()

        # harmony.shape = (num_parts, num_dims)
        harmony = result[:, i_tick, :].copy()
        # harmony.shape = (num_parts - 1, num_dims)
        harmony = np.delete(harmony, i_part, axis=0)
        x, y = harmony.shape
        harmony = harmony.view((1, x*y))

        # getting outputs from pitch models
        pn_output = models["pitch_next_"+str(i_part)](prev_seq)
        pn_output = pn_output[-1, :]
        assert pn_output.shape == (num_dims - 1,), \
            "Invalid pitch_next model output shape."
        pp_output = models["pitch_prev_"+str(i_part)](next_seq)
        pp_output = pp_output[-1, :]
        assert pp_output.shape == (num_dims - 1,), \
            "Invalid pitch_prev model output shape."
        hm_output = models["harmony_"+str(i_part)](harmony).squeeze()
        assert hm_output.shape == (num_dims - 1,), \
            "Invalid harmony model output shape."
        # getting outputs from rhythm model
        rm_output = models["rhythm_next_"+str(part_num)](prev_seq).squeeze()
        assert rm_output.shape == (1,), \
            "Invalid rhythm model output shape."

        # input for the pitch_combined model
        comb_input = torch.zeros((3, num_dims - 1)).float()
        comb_input[0, :] = pn_output
        comb_input[1, :] = pp_output
        comb_input[2, :] = hm_output
        cb_output = models["pitch_combined_"+str(i_part)](comb_input).squeeze()
        assert cb_output.shape == (num_dims - 1,), \
            "Invalid combined output shape."

        # update the pitch based on the pitch_combined output
        result[i_part, i_tick, cb_output.argmax()] = 1
        # update the rhythm
        result[i_part, i_tick, -1] = rm_output

    return result


def _populate_part(num_ticks, num_dims):
    '''
    Initial run through the part to populate the part, with only pitch_next and rhythm_next layer.
    '''
    part = np.zeros((num_ticks, num_dims))
    pitches = np.random.randint(0, num_dims - 1, num_ticks)
    rhythm = np.random.randint(0, 2, num_ticks)
    part[np.arange(num_ticks), pitches] = 1
    part[np.arange(num_ticks), -1] = rhythm

    return part
