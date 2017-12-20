import os
# prepare User distance square
import  numpy as np
import  pandas as pd
import  os
import  gc
import collections
import h5py
import h5sparse

from usefulTool import LargeSparseMatrixCosine
from usefulTool import fillDscrtNAN
from usefulTool import fillCntnueNAN
from usefulTool import scaleCntnueVariable
from usefulTool import changeNameToID
from usefulTool import splitDF
from usefulTool import tagCombine
from usefulTool import findNetwork
from usefulTool import relationToNetwork
from usefulTool import largeMatrixDis
from sltools    import save_pickle


def extractUserInfo():
    ####################           change dir       ######################
    os.chdir("C:\\Users\\22560\\PycharmProjects\\lastFM\\hetrec2011-lastfm-2k")

    # be attention of my way to deal with dates cols
    user_id = 'userID'
    userRelationship = "user_friends.csv"
    userRelationship = pd.read_csv(userRelationship)
    userRelationship.loc[:,'value'] = 1

    usr_continous = pd.read_csv("user_continuous_covariates.csv")
    usr_continous.set_index(user_id,inplace=True)
    continueList = usr_continous.columns.tolist()
    scaleCntnueVariable(usr_continous, continueList)

    # userRelationship 和 usr_continous 中的id数目是一样的
    # 共计1892个人，无人缺失

    # shoud plus 1 , for the index begin with 0
    numOfUser  = userRelationship.max().max()+1

    # ifHasitsOwn: a flag shows that the userRelationship contains the user's relationship
    # with it self or not
    ifHasitsOwn = False
    fileplace = 'C:\\Users\\22560\\PycharmProjects\\lastFM\\networkData\\'
    relationToNetwork(userRelationship,numOfUser=numOfUser,ifHasitsOwn = ifHasitsOwn,
                                ifBIGDATA = False,prefix = 'user',fileplace = fileplace)


    largeMatrixDis(usr_continous.values,ObjectHasntCntnue=[], num=2,
                   netFilePlace=fileplace ,prefix="user")

    #save_pickle(user_id_dict, fileplace + "user_id_dict")
    # return(user_id_dict)

