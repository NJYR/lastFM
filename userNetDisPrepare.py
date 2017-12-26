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


def extractUserInfo(userID,userRelationshipFile,
                    userContinueFileName,networkSavingplace,disCalculateIterNumber
                    ):

    userRelationship = pd.read_csv(userRelationshipFile)
    userRelationship.loc[:,'value'] = 1

    usr_continous = pd.read_csv(userContinueFileName)
    usr_continous.set_index(userID,inplace=True)
    user_max = usr_continous.index.max()
    usr_continous = usr_continous.reindex(list(range(user_max + 1)))
    usr_continous.reset_index(inplace=True)

    continueList = usr_continous.columns.tolist()
    continueList.remove(userID)
    usrHasntCntnue = fillCntnueNAN(usr_continous, continueList, userID)
    scaleCntnueVariable(usr_continous, continueList)

    # userRelationship 和 usr_continous 中的id数目是一样的
    # 共计1892个人，无人缺失

    # shoud plus 1 , for the index begin with 0
    numOfUser  = userRelationship.max().max()+1

    # ifHasitsOwn: a flag shows that the userRelationship contains the user's relationship
    # with it self or not
    ifHasitsOwn = False
    relationToNetwork(userRelationship,numOfUser=numOfUser,ifHasitsOwn = ifHasitsOwn,
                                ifBIGDATA = False,prefix = 'user',fileplace = networkSavingplace)


    # prepare largeDisMatrix
    usr_continous.set_index(userID,inplace = True)

    largeMatrixDis(usr_continous.values,ObjectHasntCntnue=usrHasntCntnue, num=disCalculateIterNumber,
                   netFilePlace=networkSavingplace ,prefix="user")

    #save_pickle(user_id_dict, fileplace + "user_id_dict")
    # return(user_id_dict)

