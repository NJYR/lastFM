import os
import pandas as pd
import numpy as np
import h5py


filepath = "C:\\Users\\22560\\PycharmProjects\\lastFM\\hetrec2011-lastfm-2k\\"
os.chdir(filepath)
# step1 : split data set
trainFilePlace = "user_artists.csv"
userID = 'userID'
itemID = 'artistID'
targetID = 'weight'
ifrandom = True
valiPercent = 0.2

trainSetName = "trainSet.csv"
validationSetName = "validationSet.csv"

trainAfterTransform = "user_artistsAfterlog.csv"
validationAfterTrans = "validationAfterlog.csv"

userContinueFileName = "user_continuous_covariates.csv"
itemContinueFileName = "artist_continuous_covariates.csv"

userRelationshipFile = "user_friends.csv"
networkSavingplace = 'C:\\Users\\22560\\PycharmProjects\\lastFM\\networkData\\'

itemRelationshipFile = "artist_continuous_covariates.csv"
itemTagFile = "artist_tags_matrix.csv"

trainAfterDealingName = "trainAfterDealing.csv"

prepare_path = 'C:\\Users\\22560\\PycharmProjects\\lastFM\\networkData\\yPrepare.h5'
latentFilePlace = "latentFactor.hdf5"
name1 = '/yData/y'
name2 = '/yData/y_trans'
prepare_name = '/yData/y'
transpose_prepare_name = '/yData/y_trans'

# fixed parameters :
method = 'l2'
iteration_num = 10
user_loop_num = 3
item_loop_num = 3

trainFilePlace = trainAfterDealingName
latentFilePlace = "latentFactor.hdf5"
validationFilePlace = validationAfterTrans


prepare_y = h5py.File(prepare_path, 'r')
y = prepare_y[prepare_name]

latent_factor_file = h5py.File('latentFactor.hdf5', 'r')
transpose_y = prepare_y[transpose_prepare_name]
user_latent_factor_matrix = latent_factor_file['userLatentFactor']
item_latent_factor_matrix = latent_factor_file['itemLatentFactor']


train = pd.read_csv("user_artistsAfterlog.csv")
train['weight_hat'] = y.value[train.userID,train.artistID]



val = pd.read_csv("validationAfterlog.csv")
val['weight_hat'] = y.value[val.userID,val.artistID]


np.sum((val.weight - val.weight_hat)**2 / val.shape[0])



val[((val.weight - val.weight_hat)**2 > 1)].sort_values(by = 'userID')

#  userID  artistID    weight  weight_hat
# 8564        0        97  7.192934    4.514352
# 16369       0        57  8.374938    6.235282
# 2233        0        85  7.351800    4.451757
# 7591        0        50  9.366489    8.293168
# 14064       0        84  7.374002    4.582862
# 9455        0        51  9.337061    7.491913

# 13,19,21

def rmse(train):
    return (np.sum((train.weight - train.weight_hat) ** 2 / train.shape[0]))

rmse(val)