import  os
import  gc
import collections
import h5py
import h5sparse
# prepare User distance square
import  numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from usefulTool import LargeSparseMatrixCosine
from usefulTool import fillDscrtNAN
from usefulTool import fillCntnueNAN
from usefulTool import scaleCntnueVariable
from usefulTool import changeNameToID
from usefulTool import splitDF
from usefulTool import tagCombine
from usefulTool import findNetwork
from usefulTool import largeMatrixDis
from usefulTool import DealingDescreteMissingValue
from sltools    import save_pickle

def regression(train):

    os.chdir("C:\\Users\\22560\\PycharmProjects\\lastFM\\hetrec2011-lastfm-2k\\")
    # import artist info
    item_id = 'artistID'
    artist_continue = pd.read_csv("artist_continuous_covariates.csv")
    artist_continue.set_index(item_id, inplace=True)
    artist_max = artist_continue.index.max()
    artist_continue = artist_continue.reindex(list(range(artist_max + 1)))
    artist_continue.reset_index(inplace=True)

    artist_tag = pd.read_csv("artist_tags_matrix.csv", dtype={"tagID": str})

    artist_tag = pd.merge(artist_tag, artist_continue[[item_id]], on=item_id, how='right')
    artist_tag.pop('value')

    fillnawith = collections.OrderedDict()
    fillnawith['tagID'] = 'no_tag'
    artist_tag = fillDscrtNAN(artist_tag, fillnawith)

    # fill na with special value calculated from data

    continueList = artist_continue.columns.tolist()
    continueList.remove(item_id)
    aritstHasntCntnue = fillCntnueNAN(artist_continue, continueList, item_id)
    scaleCntnueVariable(artist_continue, continueList)

    tagList = artist_tag.columns.tolist()
    tagList.remove(item_id)

    itemWithTag = tagCombine(artist_tag, id=item_id, tagColList=tagList)

    (itemTagmatrix, itemNoAttr) = findNetwork(itemWithTag, fillnawith, split=r"&|\|")


    itemTagmatrix[itemTagmatrix<0] = 1


    # import user info

    # be attention of my way to deal with dates cols
    user_id = 'userID'
    userRelationship = "user_friends.csv"
    userRelationship = pd.read_csv(userRelationship)
    userRelationship.loc[:, 'value'] = 1

    usr_continous = pd.read_csv("user_continuous_covariates.csv")
    usr_continous.set_index(user_id, inplace=True)
    continueList = usr_continous.columns.tolist()
    scaleCntnueVariable(usr_continous, continueList)

    # userRelationship 和 usr_continous 中的id数目是一样的
    # 共计1892个人，无人缺失

    # shoud plus 1 , for the index begin with 0
    numOfUser = userRelationship.max().max() + 1




    # import train info

    usr_continous.reset_index(inplace=True)
    artist_continue.head()

    train_usr = pd.merge(train,usr_continous,how='left')
    train_usr_artist = pd.merge(train_usr,artist_continue,how='left')

    train_usr_artist.head()

    del train,train_usr,usr_continous,userRelationship,artist_continue
    gc.collect()

    tag = itemTagmatrix[train_usr_artist.artistID,:]

    target = train_usr_artist.weight
    train_usr_artist.drop(columns = ['userID','artistID','weight'],inplace =True)


    continueVar = csr_matrix(train_usr_artist.values)
    del train_usr_artist
    gc.collect()


    coVariate = hstack([continueVar,tag])

    from sklearn import linear_model
    clf = linear_model.LinearRegression()

    clf.fit(coVariate,target)
    residual = target - clf.predict(coVariate)
    return(residual)