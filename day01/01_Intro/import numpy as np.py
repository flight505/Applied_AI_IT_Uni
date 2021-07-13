import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

url="https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv"
df=pd.read_csv(url ,sep='\t')
print(df.head(5))


points = np.random.random((100,2))
# points
D = np.zeros((100,100))
for i in range(100):
    for j in range(100):
        point_1 = points[i,:]
        point_2 = points[j,:]
        D[i,j] = np.sqrt(((point_1 + point_2)**2))




