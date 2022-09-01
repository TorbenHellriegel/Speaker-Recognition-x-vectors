import pandas as pd
import numpy as np
from ast import literal_eval

x_vector = []
for i in range(30):
    x_vector.append(['string' + str(i), i, np.array([j for j in range(i, i+7)], dtype=np.float64)])

x_vector = pd.DataFrame(x_vector)
x_vector.to_csv('TESTTEST.csv')

x_vectors_test = np.genfromtxt('TESTTEST.csv',delimiter=',',dtype=np.unicode)[1:]
x_id_test = []
x_label_test = []
x_vec_test = []
for x in x_vectors_test:
    _, x_id, x_label, x_vec = x
    x_id_test.append(x_id)
    x_label_test.append(int(x_label))
    x_vec_test.append(np.array(x_vec[1:-1].split(), dtype=np.float64))

print(x_vec_test)
print(x_label_test)
print(x_id_test)
print('done')