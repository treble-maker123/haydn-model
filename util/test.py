import os
import data
from pdb import set_trace
from multiprocessing import Pool

scores = data.load_data()
dataset = data.HaydnDataset(data=scores)
print("There are {} items in the dataset.".format(len(dataset)))

states = list(filter(lambda ds: ds is not None, dataset))
total_ticks = sum(map(lambda state: state.shape[2], states)) \
                * states[0].shape[0]
print("There are {} final corpi with a total of {} ticks."
        .format(len(states), total_ticks))

score = dataset.matrix_to_score(states[0][6, :])
score.show()
