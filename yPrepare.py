import gc
import numpy as np
import  pandas as pd
import  os
from sltools import load_pickle
from scipy.sparse import  vstack
from scipy.sparse import  csr_matrix
from scipy.sparse import  diags
from scipy.sparse import coo_matrix
from itemNetDisPrepare import extractItemInfo
from userNetDisPrepare import extractUserInfo
import dask.array as da
import h5sparse
import h5py
import  time
from regression import regression
from usefulTool import generate_quantile
from    ALS                   import als


def transfromData():
    filePlace = "C:\\Users\\22560\\PycharmProjects\\lastFM\\hetrec2011-lastfm-2k\\"
    train = pd.read_csv(filePlace+"user_artists.csv")
    train.weight = np.log(train.weight)
    train.to_csv(filePlace+"user_artistsAfterlog.csv",index = False)










def generateContinousVar(filePlace = "C:\\Users\\22560\\PycharmProjects\\lastFM\\hetrec2011-lastfm-2k\\"):
    os.chdir(filePlace)
    train = pd.read_csv("user_artistsAfterlog.csv")
    userCntinue   = generate_quantile(train,'userID','weight')
    artistCntinue = generate_quantile(train, 'artistID', 'weight')
    userCntinue.to_csv("user_continuous_covariates.csv")
    artistCntinue.to_csv("artist_continuous_covariates.csv")



def kernel(weight,h):
    return np.exp(- np.sqrt(weight)/h )


def yPrepareForBigData(user_num ,user_id_dict,item_id_dict,filePlace,item_id,user_id,target_id,train):

   pass


def yPrepareForSmallData(user_num,filePlace,item_id,user_id,target_id,train,h,sep =10):
    # sep 指的是最后的block的值，使用int(user_num/sep)

    usrList = train[user_id].unique()
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
            weightSort = weightSort[:, -4].reshape((weightSort.shape[0],1))
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



def main():


    filePlace = "C:\\Users\\22560\\PycharmProjects\\lastFM\\networkData\\"


    gc.collect()
    # read train data
    os.chdir("C:\\Users\\22560\\PycharmProjects\\lastFM\\hetrec2011-lastfm-2k")
    train = pd.read_csv("user_artistsAfterlog.csv")

    user_id = "userID"
    item_id = "artistID"
    target_id  = "weight"




    train = train.sort_values(by = [user_id,item_id])
    #
    #
    # trainMean = train.groupby('userID')['weight'].transform('mean')
    # train['weight'] = train['weight']-trainMean



    # do not do regression at the first time trying

    # weight = regression(train)
    # train['weight'] = weight

    train.to_csv("trainAfterReg.csv")
    gc.collect()


    print("train data has prepared !")

    # we are now start to prepare y
    user_num = train[user_id].max() +1




    # calculate h later
    # h is the sum of ( median of useful DIS of user) and ( median of useful DIS of user)

    with h5sparse.File(filePlace + "itemdis.h5") as item_dis,\
        h5sparse.File(filePlace + "userdis.h5") as  user_dis:
        idis = item_dis['disData/data'].value.data.copy()
        idis = idis[idis >0]
        idis.sort()

        udis = user_dis['disData/data'].value.data.copy()
        udis = udis[udis>0]
        udis.sort()
        h = np.median(udis) + np.median(idis)
        del udis ,idis
        # item_dis = h5sparse.File(filePlace + "itemdis.h5")
        # user_dis = h5sparse.File(filePlace + "userdis.h5")



    yPrepareForSmallData(user_num,filePlace,item_id,user_id,target_id,train,h)



    trainAfterReg = pd.read_csv("trainAfterReg.csv")
    row = trainAfterReg['userID'].get_values()
    col = trainAfterReg['artistID'].get_values()
    data = trainAfterReg['weight'].get_values()

    rating_matrix = coo_matrix((data, (row, col)), dtype=np.float)

    del train, row, col, data
    gc.collect()

    # y_observed = pd.read_csv('y_observed.csv')
    # row = y_observed['userID'].get_values()
    # col = y_observed['artistID'].get_values()
    # data = y_observed['y'].get_values()
    # y_observed_matrix = coo_matrix((data, (row, col)), dtype=np.float)
    # del y_observed, row, col, data
    # gc.collect()

    path = 'C:\\Users\\22560\\PycharmProjects\\lastFM\\networkData\\yPrepare.h5'
    name1 = '/yData/y'
    name2 = '/yData/y_trans'

    result = als(rating_matrix, path, name1, name2, factor_num=5, method='l2', iteration_num=100, user_loop_num=1,
                 item_loop_num=1, lambda_user=3, lambda_item=3)
    print(result)










if __name__=="__main__":

    transfromData()
    generateContinousVar()
    extractUserInfo()
    extractItemInfo()
    main()
    # print(graph(result[1:]))


    # # test
    # h5f = h5py.File("C:\\Users\\22560\\PycharmProjects\\lastFM\\networkData\\yPrepare.h5",'r')
    # h5f['/yData/y_trans'][-1,:]
    # h5f['yData/y'].value



    # usrNet = calculateNet(1,1,1*1e4)
    #
    # # generate two dis for test later method
    # usrDis = calculateDis(1,2,1*1e4)
    # itemDis = calculateDis(1,2,22*1e4)
    #
    # (ulist, ilist , rlist) = subtractIdx(1,180*1e4,encoded=True)
    #