import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


rawdataPath = "D:\ml\DBP391/archive\ptbdb_abnormal.csv"


train_data = pd.read_csv(rawdataPath, header=None)

plt.plot(train_data.columns.to_numpy().reshape((1,188)), np.array(train_data[:1]))
plt.show()