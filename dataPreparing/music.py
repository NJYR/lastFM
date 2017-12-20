# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 18:20:59 2017

@author: 18380_000
"""
# this file is used for preparation


import pandas as pd;
import numpy as np;
import os;

os.chdir('C:\\Users\\22560\\PycharmProjects\\lastFM\\hetrec2011-lastfm-2k');
artists = pd.read_csv('artists.dat',sep = '\t');

# rename the id to artistID
artists = artists.rename(columns = {'id':'artistID'})


tags = pd.read_csv('tags.dat',sep = '\t',encoding = 'GBK');


user_artists = pd.read_csv('user_artists.dat',sep = '\t');

user_friends = pd.read_csv('user_friends.dat',sep = '\t');

user_taggedartists = pd.read_csv('user_taggedartists.dat',sep = '\t');

user_taggedartists_timestamps = pd.read_csv('user_taggedartists-timestamps.dat',sep = '\t');



#由于每个集合中用户Id不相同，需要对所有用户ID重新整合，建立新的映射。
friendID = user_friends[['friendID']];
friendID = friendID.rename(columns = {'friendID':'userID'});

userFrame = [user_artists,friendID,user_friends,user_taggedartists,user_taggedartists_timestamps];
allUserID = np.unique(np.concatenate([frame.userID.unique() for frame in userFrame],axis = 0));
allUserID.sort();
allUserIdDict = dict(zip(list(allUserID),list(range(len(allUserID)))));




for frame in userFrame:
    frame['userID'] = frame['userID'].map(allUserIdDict);

user_friends['friendID'] = friendID;
#artists同理
artistFrame = [artists,user_artists,user_taggedartists,user_taggedartists_timestamps];
allArtistID = np.unique(np.concatenate([frame.artistID.unique() for frame in artistFrame],axis = 0))
allArtistID.sort();
allArtistIdDict = dict(zip(list(allArtistID),list(range(len(allArtistID)))));

for frame in artistFrame:
    frame['artistID'] = frame['artistID'].map(allArtistIdDict);
    
#tag重编码
tagFrame = [tags,user_taggedartists,user_taggedartists_timestamps];
allTagID = np.unique(np.concatenate([frame.tagID.unique() for frame in tagFrame],axis = 0));
allTagID.sort();
allTagIdDict = dict(zip(list(allTagID),list(range(len(allTagID)))));

for frame in tagFrame:
    frame['tagID'] = frame['tagID'].map(allTagIdDict);
    
os.chdir('F:/Data/last.FM/newDataSet');
allFrame = {'artists':artists,'tags':tags,'user_artists':user_artists,'user_friends':user_friends,'user_taggedartists':user_taggedartists,'user_taggedartists_timestamps':user_taggedartists_timestamps};
for name,frame in allFrame.items():
    frame.to_csv('%s.csv' %name,index = False)
    
#输出三个映射字典
allDict = {'allArtistIdDict':allArtistIdDict,'allUserIdDict':allUserIdDict,'allTagIdDict':allTagIdDict};


for name,idDict in allDict.items():
    storeDict(idDict,'%s.txt' %name);

#存在问题 json的key不能为数值
"""
def dict2json(inputDict,filename):
    import json;
    with open(filename,'w') as fw:
        json.dump(inputDict,fw);

for name,idDict in allDict.items():
    dict2json(idDict,'%s.txt' %name);
"""