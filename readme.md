## 关于对于dis计算中涉及到的处理技巧
* step 0 :
    - 由于连续变量的表中缺少很多的item/user列
    - 在生成train的数据集中，要尽量包含最大的userID,artistID的观测
    - 注意：现在默认是train中最大的item，若不是，在itemNetDisPrepare中还要多一些处理
    -  artist_max = artist_continue.index.max() 这里要换成最大的itemNo

* step 1 : 对于均值变量表
    - 首先检查是否有观测完全无连续变量
    - 若有
    - 使用均值填充缺失
    - 并记录那些没有连续变量的观测（fillNAN函数会返回这些观测）

* step 2 ： 计算dis 时：
    - 所有没有连续属性的观测，和其他的观测的距离设置为负数
    - 从而保证在计算权值时能被检测到

* step 3： 计算加权值：
    - 在计算加权值时，检测那些item，它没有连续变量
    - 则对于user 对那些 item的评价的方法为：
    - 将权值全部设为1 并归一化，相当于用户及其朋友听歌次数的均值来进行预测


## 关于sparse中存储的地址
* 注意：
    - item : item_dis['disData/data']
    - user : user_dis['disData/data']
    - y    :   name1 = '/yData/y' , name2 = '/yData/y_trans'
    - target : item = h5f['itemLatentFactor'].value ; user = h5f['userLatentFactor'].value


