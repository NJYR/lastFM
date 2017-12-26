import  os
import  gc
import collections
import h5py
import h5sparse
# prepare User distance square
import  numpy as np
import pandas as pd
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





def extractItemInfo(itemID, itemRelationshipFile,itemTagFile
                    ,networkSavingplace,itemNetIterNum,itemDisIterNum,select
                    ):
    artist_continue = pd.read_csv( itemRelationshipFile)
    artist_continue.set_index(itemID,inplace=True)
    artist_max = artist_continue.index.max()
    artist_continue = artist_continue.reindex(list(range(artist_max+1)))
    artist_continue.reset_index(inplace=True)

    #  一共390个数据是na //没有观测到被听过 ,artist_continue.isna().sum(0)

    artist_tag = pd.read_csv(itemTagFile,dtype={"tagID":str})

    artist_tag = pd.merge(artist_tag,artist_continue[[itemID]],on=itemID,how='right')
    artist_tag.pop('value')
    #  5499 首没有tag artist_tag.isna().sum(0)
    ############################################

    # fill na
    fillnawith = collections.OrderedDict()
    fillnawith['tagID'] = 'no_tag'
    artist_tag = fillDscrtNAN(artist_tag, fillnawith)

    # fill na with special value calculated from data

    continueList  = artist_continue.columns.tolist()
    continueList.remove(itemID)
    aritstHasntCntnue = fillCntnueNAN(artist_continue, continueList,itemID)
    scaleCntnueVariable(artist_continue,continueList)




    # do the tag combine process

    tagList = artist_tag.columns.tolist()
    tagList.remove(itemID)

    itemWithTag = tagCombine(artist_tag, id=itemID, tagColList=tagList)



    (itemTagmatrix, itemNoAttr) = findNetwork(itemWithTag, fillnawith, split=r"&|\|")



    # if you want to do it using loop , you may set num > 2
    # if you set num = 2 ,it will do it once
    # save the social network here
    LargeSparseMatrixCosine(itemTagmatrix,itemNoAttr,num=itemNetIterNum, select= select,fileplace= networkSavingplace ,prefix="item")


    # read and check
    # h5f = h5sparse.File("C:\\Users\\22560\\PycharmProjects\\lastFM\\networkData\\itemdot_cosine.h5")
    # h5f['dot_cosineData/data'].value
    # sum(h5f['dot_cosineData/data'].value.sum(1) == 18022) // 5499 , check!

    # prepare largeDisMatrix
    artist_continue.set_index(itemID, inplace=True)



    largeMatrixDis(artist_continue.values,aritstHasntCntnue, num=itemDisIterNum,
                   netFilePlace= networkSavingplace,prefix="item")

    #save_pickle(user_id_dict, fileplace + "user_id_dict")


if __name__ == "__main__":
    extractItemInfo()