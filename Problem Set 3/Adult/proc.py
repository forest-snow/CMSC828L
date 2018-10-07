#from https://archive.ics.uci.edu/ml/datasets/adult

import pandas as pd
import numpy as np
from sklearn.utils import shuffle

np.random.seed(42)

columns = ["age", "type_employer", "fnlwgt", "education", 
                "education_num","marital", "occupation", "relationship", "race","sex",
                "capital_gain", "capital_loss", "hr_per_week","country", "income"]

data = pd.read_csv('adult.data', names = columns)
test = pd.read_csv('adult.test', names = columns)

data = data.append(test)

del test

data = data[~data.isin(['?'])]
data = data.dropna()
data = shuffle(data)

data = pd.get_dummies(data)

del data['fnlwgt']
del data['income_ <=50K.']
del data['income_ >50K']
del data['income_ >50K.']
del data['occupation_ ?']
del data['type_employer_ ?']

labels = data['income_ <=50K'].copy()

del data['income_ <=50K']

data = data.values.astype('uint8')
labels = labels.values.astype('uint8')
np.save('data.npy',data)
np.save('labels.npy',labels)