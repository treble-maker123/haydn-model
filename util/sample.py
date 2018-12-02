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
    # total number of time slices, default 8 measures
    num_ticks = kwargs.get("num_ticks", 128)
    num_dims = kwargs.get("num_dims", 73)
    # sequence length that the network takes in, default 2 measures
    seq_len = kwargs.get("seq_len", 32)
    part_nums = [0,1,2,3]

    # models
    models = {}
    for part_num in part_nums:
        models["pitch_next_"+str(part_num)] = \
            kwargs.get("pitch_next_"+str(part_num), None)
        models["pitch_prev_"+str(part_num)]= \
            kwargs.get("pitch_prev_"+str(part_num), None)
        models["harmony_"+str(part_num)] = \
            kwargs.get("harmony_"+str(part_num), None)
        models["pitch_combined_"+str(part_num)] = \
            kwargs.get("combined_"+str(part_num), None)
        models["rhythm_next_"+str(part_num)] = \
            kwargs.get("rhythm_next_"+str(part_num), None)

    result = np.zeros((num_parts, num_ticks, num_dims))

    # initialize the first time slice randomly
    pitches = np.random.randint(0, 72, 4) # pitch between 0 and 71 inclusive
    rhythm = np.random.randint(0, 2, 4)
    result[np.arange(num_parts), 0, pitches] = 1
    result[:, 0, -1] = rhythm

    # initialize all of the parts
    for part_num in part_nums:
        result[part_num, :, :] = _populate_part(num_ticks, num_dims,
                                                result[part_num, 0, :],
                                                part_num, models)

    # go back and revise it
    num_cells = reduce(lambda x,y: x*y, result.shape[:-1])
    # number of "nodes" to revise
    max_iters = num_cells * num_repeats
    # list of indices to revise
    part_indices = np.random.randint(0, num_parts, max_iters)
    tick_indices = np.random.randint(0+seq_len, num_ticks-seq_len, max_iters)
    for i_part, i_tick in zip(part_indices, tick_indices):
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

        # getting outputs from pitch models
        pn_output = models["pitch_next_"+str(i_part)](prev_seq)
        assert pn_output.shape(num_dims - 1,), \
            "Invalid pitch_next model output shape."
        pp_output = models["pitch_prev_"+str(i_part)](next_seq)
        assert pp_output.shape(num_dims - 1,), \
            "Invalid pitch_prev model output shape."
        hm_output = models["harmony_"+str(i_part)](harmony)
        assert hm_output.shape(num_dims - 1,), \
            "Invalid harmony model output shape."
        # getting outputs from rhythm model
        rm_output = models["rhythm_next_"+str(part_num)](prev_seq)
        assert rm_output.shape(1,), \
            "Invalid rhythm model output shape."

        # input for the pitch_combined model
        comb_input = np.zeros((3, num_dims - 1))
        comb_input[0, :] = pn_output
        comb_input[1, :] = pp_output
        comb_input[2, :] = hm_output
        cb_output = models["pitch_combined_"+str(i_part)](comb_input)
        assert cb_output.shape(num_dims - 1,), \
            "Invalid combined output shape."

        # update the pitch based on the pitch_combined output
        result[i_part, i_tick, cb_output.argmax()] = 1
        # update the rhythm
        result[i_part, i_tick, -1] = rm_output

    return result

def _populate_part(num_ticks, num_dims, init_state, part_num=0, models={}):
    '''
    Initial run through the part to populate the part, with only pitch_next and rhythm_next layer.
    '''
    part = np.zeros((num_ticks, num_dims))
    assert init_state.shape == (num_dims,), "Invalid init_state shape."
    # load the initial state into the first time tick
    part[0, :] = init_state
    # store all of the necessary models for convenience
    pitch_next = models["pitch_next_"+str(part_num)]
    rhythm_next = models["rhythm_next_"+str(part_num)]
    # iterate through ticks to produce initial results
    for tick in range(num_ticks):
        cur_tick = tick + 1
        input_state = part[tick, :]
        pn = pitch_next(input_state)
        assert pn.shape == (num_ticks - 1), \
            "Invalid output shape from pitch_next"
        rn = rhythm_next(input_state)
        assert rn.shape == (1, ), \
            "Invalid output shape from rhythm_next"
        part[cur_tick] = np.concatenate((pn, rn))

    return part
