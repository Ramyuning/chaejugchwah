import numpy as np
import pandas as pd
# population = []
# for i in range(3):
#     schedule = np.random.randint(2, size=(20 , 21))
#     population.append(schedule)
# # print(population)
# for i in population:
#     for schedule in i:
#         print (schedule[0])

dt = pd.read_csv('/Users/jojeonghyeon/Documents/WorkSpace/PYTHON/chaejugchwah/output.csv')
ndt = pd.read_csv('/Users/jojeonghyeon/Documents/WorkSpace/PYTHON/chaejugchwah/nurse_schedule.csv')
print(ndt.iloc[0,:21]*[2 for _ in range(21)])