import pandas as pd
import os
import gc
from surprise import SVD,SVDpp
from surprise import Dataset
from surprise import evaluate, print_perf
from surprise.dataset import Reader





filePlace = "C:\\Users\\22560\\PycharmProjects\\lastFM\\networkData\\"

gc.collect()
# read train data
os.chdir("C:\\Users\\22560\\PycharmProjects\\lastFM\\hetrec2011-lastfm-2k")
train = pd.read_csv("trainAfterReg.csv")


algo = SVD()
reader = Reader(rating_scale=(train.weight.min(),train.weight.max()))
data = Dataset.load_from_df(train[['userID', 'artistID', 'weight']], reader)

data.split(3)

perf = evaluate(algo, data, measures=['RMSE', 'MAE'])

print_perf(perf)














import numpy as np
from sklearn.manifold import TSNE
import h5py
from numpy import  vstack
h5f = h5py.File("C:\\Users\\22560\\PycharmProjects\\lastFM\\hetrec2011-lastfm-2k\\latentFactor.hdf5")
item = h5f['itemLatentFactor'].value
user = h5f['userLatentFactor'].value

item = item/np.sqrt((item*item).sum(1).reshape((item.shape[0],1)))
user = user/np.sqrt((user*user).sum(1).reshape((user.shape[0],1)))



isize = 600
usize = 600
h = vstack([item[:isize],user[:usize]])

X_embedded = TSNE(n_components=2).fit_transform(h)

item_embedded = X_embedded[:600,:]
user_embedded = X_embedded[600:,:]


import seaborn as sns
import matplotlib.pyplot as plt

# sns.set(style="white", color_codes=True)
# grid = sns.JointGrid(X_embedded[:,0], X_embedded[:,1], space=0, size=6, ratio=50)
# grid.plot_joint(plt.scatter, color="g")
# grid.plot_marginals(sns.rugplot, height=1, color="g")
#

sns.set(style="darkgrid")
f, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect("equal")

# Draw the two density plots
ax = sns.kdeplot(item_embedded[:,0], item_embedded[:,1],
                 cmap="Reds", shade=True, shade_lowest=False)
ax = sns.kdeplot(user_embedded[:,0],user_embedded[:,1],
                 cmap="Blues", shade=True, shade_lowest=False)




import gc
import numpy as np
import  pandas as pd
import  os
from sltools import load_pickle
from scipy.sparse import  vstack
from scipy.sparse import  csr_matrix
from scipy.sparse import  diags
from scipy.sparse import coo_matrix
from itemNetDisPrepare import extractItemInfo
from userNetDisPrepare import extractUserInfo
import dask.array as da
import h5sparse
import h5py
import  time
from regression import regression
from usefulTool import generate_quantile
from    ALS                   import als

filePlace = "C:\\Users\\22560\\PycharmProjects\\lastFM\\networkData\\"

gc.collect()
# read train data
os.chdir("C:\\Users\\22560\\PycharmProjects\\lastFM\\hetrec2011-lastfm-2k")
train = pd.read_csv("user_artistsAfterlog.csv")

user_id = "userID"
item_id = "artistID"
target_id = "weight"

trainAfterReg = pd.read_csv("trainAfterReg.csv")
row = trainAfterReg['userID'].get_values()
col = trainAfterReg['artistID'].get_values()
data = trainAfterReg['weight'].get_values()

rating_matrix = coo_matrix((data, (row, col)), dtype=np.float)

del train, row, col, data
gc.collect()

# y_observed = pd.read_csv('y_observed.csv')
# row = y_observed['userID'].get_values()
# col = y_observed['artistID'].get_values()
# data = y_observed['y'].get_values()
# y_observed_matrix = coo_matrix((data, (row, col)), dtype=np.float)
# del y_observed, row, col, data
# gc.collect()

path = 'C:\\Users\\22560\\PycharmProjects\\lastFM\\networkData\\yPrepare.h5'
name1 = '/yData/y'
name2 = '/yData/y_trans'

result = als(rating_matrix, path, name1, name2, factor_num=30, method='l2', iteration_num=20, user_loop_num=3,
             item_loop_num=3, lambda_user=0.01, lambda_item=0.1)
print(result)










