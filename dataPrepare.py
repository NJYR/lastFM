
import os
import pandas as pd
import collections
from usefulTool import LargeSparseMatrixCosine
from usefulTool import fillDscrtNAN
from usefulTool import fillCntnueNAN
from usefulTool import scaleCntnueVariable
from usefulTool import changeNameToID
from usefulTool import splitDF
from usefulTool import tagCombine
from usefulTool import findNetwork
from usefulTool import largeMatrixDis


# prepare artist continue info
os.chdir("C:\\Users\\22560\\PycharmProjects\\lastFM\\hetrec2011-lastfm-2k")
itemCntnue = pd.read_csv("artist_continuous_covariates.csv")
itemCntnue.set_index(itemCntnue.artistID,inplace=True)
itemCntnue = itemCntnue.reindex( list(range(itemCntnue.artistID.max()+1))  )
itemCntnue.artistID = itemCntnue.index


# prepare user continue info
userCntnue = pd.read_csv("user_continuous_covariates.csv")
userCntnue.set_index(userCntnue.userID,inplace=True)
userCntnue = userCntnue.reindex(list(range(userCntnue.userID.max()+1)) )




# prepare user-user relationship

user_friends =  pd.read_csv("user_friends.csv")



# prepare item-tag data

item_tag = pd.read_csv("artist_tags_matrix.csv",dtype={'tagID':str,'value':str})

# spread item_tag
item_tag= pd.merge(item_tag,itemCntnue[['artistID']],on='artistID',how = 'right')

item_tag.pop('value')

fillnawith = collections.OrderedDict()
fillnawith['tagID'] = 'no_tag'



item_tag = fillDscrtNAN(item_tag, fillnawith)

id = "artistID"
colList = item_tag.columns.tolist()
colList.remove(id)

itemWithTag = tagCombine(item_tag, id='artistID', tagColList=colList)


(itemTagmatrix, itemNoAttr) = findNetwork(itemWithTag, fillnawith, split=r"&|\|")


for row in itemNoAttr:
    itemTagmatrix[row, :] = -1


fileplace = "C:\\Users\\22560\\PycharmProjects\\lastFM\\hetrec2011-lastfm-2k"
LargeSparseMatrixCosine(itemTagmatrix, itemNoAttr,num=2, fileplace=fileplace,prefix="item")

















































