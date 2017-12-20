# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 23:26:28 2017

@author: 18380_000
"""

import pandas as pd;
import os;

os.chdir('F:/Data/last.FM/newDataSet');
artists = pd.read_csv('artists.csv',encoding = 'GBK');
tags = pd.read_csv('tags.csv',encoding = 'GBK');
user_artists = pd.read_csv('user_artists.csv',encoding = 'GBK');
user_friends = pd.read_csv('user_friends.csv',encoding = 'GBK');
user_taggedartists = pd.read_csv('user_taggedartists.csv',encoding = 'GBK');
user_taggedartists_timestamps = pd.read_csv('user_taggedartists_timestamps.csv',encoding = 'GBK');
#生成artist及其对应tags的矩阵
artist_tags = user_taggedartists[['artistID','tagID']];
artist_tags.drop_duplicates(inplace=True);
artist_tags.sort_values(['artistID','tagID'],inplace=True);

user_friends['value'] = 1;
artist_tags['value'] = 1;

user_friends.to_csv('user_friends_matrix.csv',index = False);
artist_tags.to_csv('artist_tags_matrix.csv',index = False);

#计算分位数函数

def continuousCovariates(frame,by,calcu,quant=[0.5]):
    return frame.groupby(by)[calcu].quantile(quant).unstack()

#计算用户分位数

user_continuous_covariates = continuousCovariates(user_artists,'userID','weight',quant = [0.05,0.25,0.50,0.75,0.95]);
user_continuous_covariates = user_continuous_covariates.rename(columns={0.05:'userQ5',0.25:'userQ25',0.50:'userQ50',0.75:'userQ75',0.95:'userQ95'})

#计算歌手分位数  

artist_continuous_covariates = continuousCovariates(user_artists,'artistID','weight',quant = [0.05,0.25,0.50,0.75,0.95]);
artist_continuous_covariates = artist_continuous_covariates.rename(columns={0.05:'artistQ5',0.25:'artistQ25',0.50:'artistQ50',0.75:'artistQ75',0.95:'artistQ95'})

user_continuous_covariates.to_csv('user_continuous_covariates.csv');
artist_continuous_covariates.to_csv('artist_continuous_covariates.csv');

