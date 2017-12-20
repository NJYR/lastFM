# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 19:35:30 2017

@author: 18380_000
"""

import pandas as pd;
import numpy as np;
from scipy.sparse import coo_matrix;
import dask.array as da;
import os;

os.chdir('F:/Data/last.FM/newDataSet');

user_artists = pd.read_csv('user_artists.csv',encoding = 'GBK');
user_friends_matrix = pd.read_csv('user_friends_matrix.csv');
artist_tags_matrix = pd.read_csv('artist_tags_matrix.csv');
user_continuous_covariates = pd.read_csv('user_continuous_covariates.csv',index_col = 'userID');
artist_continuous_covariates = pd.read_csv('artist_continuous_covariates.csv',index_col = 'artistID');

def sparseMatrix(frame,row,col,data,shape=None):
    data = frame[data].get_values();
    row = frame[row].get_values();
    col = frame[col].get_values();
    if(shape==None):
        return coo_matrix((data,(row,col)));
    else:
        return coo_matrix((data,(row,col)),shape=shape);
 
sparse_network_matrix = sparseMatrix(frame=user_friends_matrix,data='value',row='userID',col='friendID',shape=(1892,1892));
sparse_artist_matrix = sparseMatrix(frame=artist_tags_matrix,data='value',row='artistID',col='tagID',shape=(18022,11946));
 
user_continuous_covariates_matrix = np.mat(user_continuous_covariates.get_values());
artist_continuous_covariates_matrix = np.mat(artist_continuous_covariates.get_values());
 
#artist_similarity = sparse_artist_matrix.dot(sparse_artist_matrix.transpose());