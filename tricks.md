# written by xuan
## 关于写这个的目的

首先我们一直强调的是，我们做的是交流而不是报告，我们想在这个过程中告诉大家我们自己做了哪些东西，遇到了哪些坑，并告诉大家如何避免这些坑，当然，也是为了展示这个内容的工作量有多大

### 本文分成以下基本部分

- 数据清洗过程中的坑
- 用户/商品相似度 计算过程中的坑
- 用户/商品距离 计算中遇到的坑
- 权重计算中遇到的坑
- 交叉验证中遇到的坑

### part 1 数据清洗过程中遇到的坑


一切的开始首先要做的事情肯定是进行数据清洗


首先把数据读进来


#### 注意：这里读入数据有几个技巧

- 使用dtype 在读取数据时可以声明类型
- iterator 可以逐步的读入数据，在测试时可以加速运行
- 关于时间的处理技巧：
    - 虽然这里没有涉及时间的变量，但是简单的提一下，pandas 支持在输入时处理时间
    - 方法如下
    
```python
    # dataparse 声明时间类型的格式，parse_dates指明哪些数据会处理为时间类型
    dateparse = lambda x: pd.datetime.strptime(x, '%Y%m%d')
    user = pd.read_csv(usertable,encoding = "UTF-8",
                parse_dates= ['registration_init_time','expiration_date']
                ,date_parser=dateparse
                ,iterator= True
                       )
    # 顺便，获取时间差的方法为：
    user['cntinue'] =user.expiration_date\
                        -user.registration_init_time

    user.cntinue = user.cntinue.dt.days

```


```python
import pandas as pd

fileplace = "C:\\Users\\22560\\Desktop\\recommand Sys\\recommand Sys\\songsCSV.csv"
song = pd.read_csv(fileplace,dtype ={
        "song_id" : str,
        "genre_ids" : str   
    },iterator = True
)
chunksize =100 #30000
song  = song.get_chunk(chunksize)
song = song.loc[8:9,['song_id','genre_ids']]
song.reset_index(inplace =True)
song.song_id = song.song_id.str.slice(0,5)
song.drop(columns='index',inplace  = True)
song.loc[0,'genre_ids'] = '352'
song.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>song_id</th>
      <th>genre_ids</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>oTi7o</td>
      <td>352</td>
    </tr>
    <tr>
      <th>1</th>
      <td>btcG0</td>
      <td>352|1995</td>
    </tr>
  </tbody>
</table>
</div>



注意到这里有两个问题需要处理：


- 需要对于song_id 编码
- 一首歌有可能对应多个genre_ids,对此我们需要进行处理

接下来我们一步一步解决


#### 关于编码

推荐的方式为使用sklearn 的encoder 方法


```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(song['song_id'])
originalName = song['song_id']
song['song_id'] = le.transform(song['song_id'])
```


```python
# 可以看到 song_id 被编码了
song.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>song_id</th>
      <th>genre_ids</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>352</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>352|1995</td>
    </tr>
  </tbody>
</table>
</div>



#### 关于处理genre_id

我们希望把数据处理成这个样子,即每一行都只有一首歌对应一个体裁


```python
# part1 执行结束后再来执行
song.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>song_id</th>
      <th>genre_ids</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>352</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>352|1995</td>
    </tr>
  </tbody>
</table>
</div>



下面来讲如何做到这样

####  step 1 利用 series.str.* 这些预置文本函数处理文本
首先，我们需要利用pandas的str类函数 ， 把genre_ids split 开


```python
song.genre_ids = song.genre_ids.str.split(r"|")
song.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>song_id</th>
      <th>genre_ids</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>[352]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>[352, 1995]</td>
    </tr>
  </tbody>
</table>
</div>







先做些准备工作：
#### step2 使用apply 函数生成中间使用的变量

首先，希望把歌曲id变成和体裁一样的形式（处理结果为song_id_forSpread），

方便后面展开


```python
song['numofType'] = song.genre_ids.apply(lambda x: len(x))

song['song_id_forSpread'] = song.song_id.apply(lambda x: [x] ) * song.numofType

song.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>song_id</th>
      <th>genre_ids</th>
      <th>numofType</th>
      <th>song_id_forSpread</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>[352]</td>
      <td>1</td>
      <td>[1]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>[352, 1995]</td>
      <td>2</td>
      <td>[0, 0]</td>
    </tr>
  </tbody>
</table>
</div>



#### step3 用python风格的for来生成数据


然后就会遇到新的问题，对于一个list ,  a = [[465],[352,1995]] ，

如何展开成a = [465,352,1995]

这个处理是非常python的：




```python
a = [[1,2,3],[1,2]]

spreada = [j for i in a for j in i]

spreada
```




    [1, 2, 3, 1, 2]



现在用这个思路来处理数据


```python

genre_ids = [ j for  i in song.genre_ids for j in i]

song_ids = [j for i in song.song_id_forSpread for j in i ]

song = pd.DataFrame({'song_id': song_ids,'genre_id' : genre_ids})



```


```python
song = song.reindex(columns  = ['song_id','genre_id'])
song.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>song_id</th>
      <th>genre_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>352</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>352</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1995</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

可以看出已经实现了想要的结果


### part 2 用户/商品相似度 计算过程中的坑

在将数据导入后，就需要将用户和用户的相似度计算出来

这部分的问题主要有两个

- 相似度如何计算？

- 计算后的数据如何存储

- 关于用户/商品缺失tag

#### 2.1 相似度如何计算？

相似度的计算分为两个部分：

- item-tag 稀疏矩阵的创建
- 余弦相似度的计算

下面一一介绍这些内容

##### 2.1.1 item-tag 稀疏矩阵的创建

在进行数据的清洗后，我们获得了如下的数据形式：


```python
song.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>song_id</th>
      <th>genre_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>352</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>352</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1995</td>
    </tr>
  </tbody>
</table>
</div>



首先，我们需要对于genre_id进行编码


```python
le = LabelEncoder()
le.fit(song['genre_id'])
originalName = song['genre_id']
song['genre_id'] = le.transform(song['genre_id'])
song.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>song_id</th>
      <th>genre_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



现在，我们希望把这个数据化为sparse矩阵的形式，使得每行代表一首歌，每列代表一个tag，

稀疏矩阵中，x[i,j] = 1 表示第i个商品有第j个tag，x[i,j] = 0 表示没有这个tag

使用csr_matrix 工具，我们可以轻易的完成这个步骤


```python
from  scipy.sparse import csr_matrix
```


```python
# 设置需要填充进稀疏矩阵的数值
song['value'] = 1

# 创建矩阵

song_tag = csr_matrix((song.value,(song.song_id,song.genre_id)))
```


```python
song_tag.todense()
```




    matrix([[1, 1],
            [0, 1]], dtype=int64)



##### 2.1.2 item-item 余弦相似度的计算

在获得了item-tag 矩阵后，我们依据这个矩阵，对于每个item，都有一个对应的向量

而根据这个向量，可以计算item-item的相似度

扩展：余弦相似度如何计算


```python

```
