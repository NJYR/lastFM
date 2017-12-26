import gc
import numpy as np
import  pandas as pd
import  os
from itemNetDisPrepare import extractItemInfo
from userNetDisPrepare import extractUserInfo
from regression import regression
from itertools  import product
from collections import OrderedDict
from prepare_for_y import *
from sltools import save_pickle
from sltools import load_pickle







if __name__=="__main__":
    np.set_printoptions(precision=4,suppress=True)

    filepath = "C:\\Users\\22560\\PycharmProjects\\lastFM\\hetrec2011-lastfm-2k\\"
    os.chdir(filepath)
    # step1 : split data set
    trainFilePlace = "user_artists.csv"
    userID = 'userID'
    itemID = 'artistID'
    targetID = 'weight'
    ifrandom = True
    valiPercent = 0.2

    trainSetName = "trainSet.csv"
    validationSetName = "validationSet.csv"


    trainAfterTransform = "user_artistsAfterlog.csv"
    validationAfterTrans = "validationAfterlog.csv"

    userContinueFileName = "user_continuous_covariates.csv"
    itemContinueFileName = "artist_continuous_covariates.csv"

    userRelationshipFile = "user_friends.csv"
    networkSavingplace = 'C:\\Users\\22560\\PycharmProjects\\lastFM\\networkData\\'

    itemRelationshipFile = "artist_continuous_covariates.csv"
    itemTagFile = "artist_tags_matrix.csv"

    trainAfterDealingName = "trainAfterDealing.csv"


   ############# you need do it only once #############

    #
    # splitTrain(trainFilePlace,userID,itemID,targetID,
    #            ifrandom,valiPercent ,
    #             trainSetName,validationSetName
    # )



    #  step2 transformData

    # 对train/validation 做log变换
    ## do the transform to the original data, like log


    ############# you need do it only once #############
    # transfromData(trainSetName,trainAfterTransform)
    # transfromData(validationSetName, validationAfterTrans)


    #  step3 generate continous variable
    # 注意：这里必须用部分的trainSet，而不是最原始的train

    ############# you need do it only once #############
    # generateContinousVar(trainAfterTransform,
    #                      userID,itemID,targetID
    #                      , userContinueFileName
    #                      , itemContinueFileName
    #                      )

    # step4 prepare user distance & network data

    ############# you need do it only once #############
    # disCalculateIterNumber = 3
    # extractUserInfo(userID,userRelationshipFile,
    #                 userContinueFileName,networkSavingplace,disCalculateIterNumber
    #                 )

    print("usr info prepared !")

    #  step 5 prepare item info

    ############# you need do it only once #############
    # itemNetIterNum = 5
    # itemDisIterNum = 5
    # extractItemInfo(itemID, itemRelationshipFile,itemTagFile
    #                 ,networkSavingplace,itemNetIterNum,itemDisIterNum,select = 0
    #                 )

    print("item info prepared !")

    # step 6 preparing y
    doRregression = False
    # 会对train做一个排序（和回归），trainAfterDealingName  是处理后文件名
    ############# you need do it only once #############
    # prepareY(doRregression,
    #          networkSavingplace,
    #             trainAfterTransform,
    #                 userID,itemID,targetID,
    #                     trainAfterDealingName)



    # # step 7 estimate the P,Q
    prepare_path    = 'C:\\Users\\22560\\PycharmProjects\\lastFM\\networkData\\yPrepare.h5'
    name1 = '/yData/y'
    name2 = '/yData/y_trans'
    prepare_name='/yData/y'
    transpose_prepare_name ='/yData/y_trans'

    # fixed parameters :
    method = 'l2'
    iteration_num = 100
    user_loop_num = 3
    item_loop_num = 3

    trainFilePlace = trainAfterDealingName
    latentFilePlace = "latentFactor.hdf5"
    validationFilePlace =  validationAfterTrans


    # tuning parameters :
    # lambda_user = [0.01,0.05,0.1,0.5]
    # lambda_item = [0.01,0.05,0.1,0.5]
    # factor_num  = [5,20,50]

    lambda_user = [50]
    lambda_item = [50]
    factor_num  = [200]


    iter = product(lambda_user,lambda_item,factor_num) #iter 是迭代器
    record = []
    counter = 0
    for parameters in iter:

        lambda_user, lambda_item, factor_num = parameters

        print("choosing parameters are : \n"
              ,"lambda_user : ",lambda_user , "\n"
              ,"lambda_item : ",lambda_item , "\n"
              ,"factor_num  ： ",factor_num , "\n"
              )
        # trainAndValidation 返回在训练集和测试集合上的rmse
        rmse, rmse_vali = trainAndValidation(trainFilePlace, prepare_path, prepare_name, transpose_prepare_name,
                                             factor_num, method, iteration_num,
                                             user_loop_num, item_loop_num, lambda_user, lambda_item,
                                             latentFilePlace, validationFilePlace)
        result = OrderedDict()
        result['parameters'] = parameters
        result['rmse'] = rmse[-1]
        result['rmse_vali'] = rmse_vali[-1]
        counter = counter + 1
        record.append(result)
        if (counter % 5 == 0 ):
            save_pickle(record, "record.rcd")

    minRmse = min([ i['rmse_vali'] for  i in record])
    for i in record:
        if i['rmse_vali'] == minRmse:
            print(i)


    load_pickle("record.rcd")
    #
