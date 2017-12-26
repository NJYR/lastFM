import gc
import numpy as np
import  pandas as pd
import  os
from scipy.sparse import  csr_matrix
from scipy.sparse import  diags
import h5sparse
import h5py
import  time
from regression import regression
from usefulTool import generate_quantile
from    ALS     import  trainAndValidation
from itertools  import product
from  numpy.random import  permutation

def splitTrain(trainFilePlace,
               userID,itemID,targetID,
               ifrandom ,valiPercent
                ,trainSetName, validationSetName
               ):
    # trainfilePlace = "C:\\Users\\22560\\PycharmProjects\\lastFM\\hetrec2011-lastfm-2k\\user_artists.csv"
    # ifrandom = True
    # valiPercent = 0.2
    #userID = 'userID'; itemID='artistID'; targetID = 'weight'
    trainOriginaldata = pd.read_csv(trainFilePlace)
    trainOriginaldata = trainOriginaldata.reindex(columns = [userID,itemID,targetID])

    # trainOriginaldata.reindex
    if ifrandom:
        while(True):
            # 此参数用于确定原始数据的切分是否需要按照random处理
            #trainOriginaldata = trainOriginaldata.head()
            numOfdata = trainOriginaldata.shape[0]
            permu = permutation(range(numOfdata))
            split = int((1-valiPercent)*numOfdata)

            train      = trainOriginaldata.loc[permu[:split],:]
            validation = trainOriginaldata.loc[permu[split:],:]

            train = pd.concat([train, validation.nlargest(1, columns=itemID),validation.nlargest(1, columns=userID)])
            train.drop_duplicates(keep='first', inplace=True)

            flag1 =  len(trainOriginaldata[userID].unique()) - len(train[userID].unique())

            flag2 =  len(trainOriginaldata[itemID].unique()) - len(train[itemID].unique())
            if(flag1 == 0 or flag2 == 0 ):
                 print(flag1,flag2)
                 break
            else:
                print(flag1,flag2)

        train.to_csv(trainSetName,index=False)
        validation.to_csv(validationSetName,index=False)
    else:
        raise Exception('data has time attr , do not split as random')




def transfromData(trainFilePlace,nameAfterTransform):

        train = pd.read_csv(trainFilePlace)
        train.weight = np.log(train.weight)
        train.to_csv(nameAfterTransform, index=False)


def generateContinousVar(nameAfterTransform,userID,itemID,targetID
                         ,userContinueFileName
                         ,itemContinueFileName
                         ):
    train = pd.read_csv(nameAfterTransform)
    userCntinue   = generate_quantile(train,userID,targetID)
    itemCntinue = generate_quantile(train, itemID, targetID)
    userCntinue.to_csv(userContinueFileName)
    itemCntinue.to_csv(itemContinueFileName)


def prepareY(doRregression,
             networkSavingplace,
                trainAfterTransform,
                    userID,itemID,targetID,
                        trainAfterDealingName
             ):

    # read train data
    train = pd.read_csv(trainAfterTransform)
    train = train.sort_values(by = [userID,itemID])


    # do not do regression at the first time trying

    if doRregression == True:
        weight = regression(train)
        train['weight'] = weight

    train.to_csv(trainAfterDealingName,index = False)
    gc.collect()


    print("train data has prepared !")

    # we are now start to prepare y
    user_num = train[userID].max() +1

    # calculate h later
    # h is the sum of ( median of useful DIS of user) and ( median of useful DIS of user)

    with h5sparse.File(networkSavingplace + "itemdis.h5") as item_dis,\
        h5sparse.File(networkSavingplace + "userdis.h5") as  user_dis:
        idis = item_dis['disData/data'].value.data.copy()
        idis = idis[idis >0]
        idis.sort()
        udis = user_dis['disData/data'].value.data.copy()
        udis = udis[udis>0]
        udis.sort()
        h = np.median(udis) + np.median(idis)
        del udis ,idis



    yPrepareForSmallData(user_num,networkSavingplace,itemID,userID,targetID,train,h)


def yPrepareForBigData(user_num ,user_id_dict,item_id_dict,filePlace,item_id,user_id,target_id,train):

   pass


def yPrepareForSmallData(user_num,filePlace,item_id,user_id,target_id,train,h,sep =10):
    # sep 指的是最后的block的值，使用int(user_num/sep)

    usrList =list(range(train[user_id].max()+1))
    usrList.sort()
    usrList = np.array(usrList)

    # select the data from train which is related with usrNowDealing
    with h5sparse.File(filePlace+"userdot_cosine.h5",'r') as user_net,\
        h5sparse.File(filePlace + "itemdot_cosine.h5",'r') as item_net,\
        h5sparse.File(filePlace + "itemdis.h5",'r') as item_dis,\
        h5sparse.File(filePlace + "userdis.h5",'r') as user_dis,\
        h5py.File(filePlace + "yPrepare.h5")  as yPrepare   :


        # # use for test
        # user_net = h5sparse.File(filePlace+"userdot_cosine.h5")
        # item_net = h5sparse.File(filePlace + "itemdot_cosine.h5")
        # item_dis = h5sparse.File(filePlace + "itemdis.h5")
        # user_dis = h5sparse.File(filePlace + "userdis.h5")
        # yPrepare =  h5py.File(filePlace + "yPrepare.h5")


        # code can be used several times

        # load the item relationship
        itemRelationship = item_net['dot_cosineData/data'].value
        itemDisRelationship = item_dis['disData/data'].value



        print("start preparing y !! , please be patient \n")
        # code related to usrNowdealing

        timeStart = time.time()
        totalTime = 0
        for userNowDealing in   range(user_num):


            # obtain the useful train dataset
            usrRelationshipUsed = user_net['dot_cosineData/data'][userNowDealing:(userNowDealing+1)].\
                                            toarray().ravel()

            usrRelationshipUsed = usrRelationshipUsed.astype(np.float32)# get data which has relationship with userNowDealing
            usrHasRelation = usrList[usrRelationshipUsed>0]
            trainHasRelation = train[train[user_id].\
                isin(usrHasRelation)].sort_values(by = [item_id])

            # ## for test ##
            # trainHasRelation = trainHasRelation[trainHasRelation.userID == 0]
            # ##############

            if trainHasRelation.empty:
                trainHasRelation = train.sample(frac = 0.1)



            relatedItem = trainHasRelation[item_id].values
            relatedUser = trainHasRelation[user_id].values
            relatedTarget = trainHasRelation[target_id]




            # get the item-train-sized  item net work
            itemToitemNetRelated = itemRelationship[:,relatedItem]

            gc.collect()



            # get the item-train-sized kernel data


            # get the item-train-sized  item  dis data
            itemToitemDisRelated = itemDisRelationship[:,relatedItem]

            gc.collect()

            # you can do the truncate process here or not




            # broadcast with the user-train-sized x data

            ## obtain user relationship with train
            userDisRelationship = user_dis['disData/data']\
                [userNowDealing:(userNowDealing+1)].todense()

            userDisRelationship = userDisRelationship[:,relatedUser]


            # turn it to sparse like matrix
            userDisRelationship = userDisRelationship.A.ravel()


            _idptr = itemToitemNetRelated.indptr
            _data  = userDisRelationship[itemToitemNetRelated.indices]
            _idces = itemToitemNetRelated.indices


            userDisRelationship  =  csr_matrix((_data,_idces,_idptr))


            del _idptr ; _data ; _idces
            gc.collect()

            # calculate kernel
            weight =  userDisRelationship + itemToitemDisRelated
            weight = weight.todense()



            weight = kernel(weight,h)
            weight[np.isnan(weight)] = 1
            weight = np.multiply(weight,itemToitemNetRelated.todense()).A


            weightSort = weight.copy()
            weightSort.sort()
            weightSort = weightSort[:, -5].reshape((weightSort.shape[0],1))
            # return a sparse
            weight[ weight < weightSort ] = 0
            weight = csr_matrix(weight)


            weight_sum =  weight.sum(1).A.ravel()

            weight_sum_reciprocal = diags( 1/weight_sum )

            del weight_sum ,userDisRelationship,itemToitemDisRelated

            gc.collect()

            weight = weight_sum_reciprocal.dot(weight)

            #print(sum((weight>0).sum(1) == weight.shape[1]))

            y = weight.dot(relatedTarget)




            # make sure your dataset is cleaned before iteration
            if userNowDealing == 0:
                for key in yPrepare.keys():
                    del yPrepare[key]



            if userNowDealing ==0:
                blockSize = int(user_num / sep)
                yset = yPrepare.create_dataset("/yData/y",shape = (1,len(y)), maxshape=(None,len(y)) ,chunks = (blockSize,len(y)),dtype=np.float32)
                yset_t = yPrepare.create_dataset("/yData/y_trans",shape= (len(y),1),maxshape=(len(y),None),chunks = (len(y),blockSize),dtype=np.float32)
                yset[:] = y
                yset_t[:] = y.reshape((len(y),1))
            else:

                yset.resize(userNowDealing+1,axis = 0)
                yset[userNowDealing,:] = y
                yset_t.resize(userNowDealing+1,axis = 1)
                yset_t[:,userNowDealing] = y

            del weight,weight_sum_reciprocal,trainHasRelation



            if(userNowDealing%15 ==0):
                timeEnd  = time.time()
                totalTime = totalTime + timeEnd - timeStart
                print("10 iteration costs :  ",round(timeEnd - timeStart,2))
                print("totalTime is       :  ",round(totalTime ,2))
                timeStart = time.time()

                print("the     "+ str(userNowDealing) +"     usr is now preparing")
                print("the   ",np.round((userNowDealing+1)/(user_num-1),3 ),"  of data is prepared \n \n please be patient")


def kernel(weight,h):
    return np.exp(- np.sqrt(weight)/h )
