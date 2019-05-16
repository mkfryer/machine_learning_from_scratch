"""
Temporary Text Preprocessor
TODO: GENERALIZE THIS LATER.
"""
from scipy.io import arff




def prepare_voting_data():
    dataset, meta = arff.loadarff("voting_task.arff") 
    dataset = dataset.tolist()

    #preprocess
    for i in range(len(dataset)):
        dataset[i] = list(dataset[i])
        for j, y in enumerate(dataset[i]):
            y = y.decode("utf-8") 
            if y == '\'y\'' or y == '\'democrat\'':
                dataset[i][j] = 1
            else:
                dataset[i][j] = 0

    return dataset