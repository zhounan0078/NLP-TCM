# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 09:35:04 2022

@author: Zhou N
"""
#%%
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from collections import Counter, OrderedDict
import copy as cp

#%%
# 实例化
cvect = CountVectorizer(lowercase=False)
tf_idf = TfidfVectorizer()

#%%
# file=open('C:\Myfiles\方剂\矩阵数据.txt','r',encoding='utf-8',errors='ignore')
# txt=file.read()
# file.close()

# 读文档，这里把方名set成index了，所以dataframe中只有草药
txt = pd.read_csv('English example.csv')
txt = pd.DataFrame(txt)
txt = txt.set_index('Prescription name')
#%%
# 遍历dataframe中的草药，把所有的草药怼成一个长字符串，空格分割
sentence = ""
for index, row in txt.iterrows():
    for sen in row:
        sentence = sentence+sen+','

# 把长字符转打散成一个list，这里的ls会在CountVectorizer和TfidfVectorizer中用到，但是TreebankWordTokenizer不用
ls = sentence.split(sep=' ')
co=Counter(ls)
most_common_herb = co.most_common()
df = pd.DataFrame.from_dict(co, orient='index')
#%%
# TreebankWordTokenizer
tbww = tbw.tokenize(sentence)
word_bag = Counter(ls)
word_bag.most_common()

# CountVectorizer
cvect.fit_transform(word_bag)
cvect.vocabulary_
#%%
# TfidfVectorizer
tf_idf = TfidfVectorizer(min_df=1)
b=tf_idf.fit_transform(ls)

#%%
# 重新读文件，这里不设方名为index
file = pd.read_csv('C:\Myfiles\方剂\矩阵.csv')
file = pd.DataFrame(file)
file = file.set_index('方名')
file_dict = pd.DataFrame(columns=['name', 'vector'])
#%%
# 做成以{方名：组成}为键值对的字典
file_dict = dict()
for index, row in file.iterrows():
    for w in row:
        per_vect = []
        ws = w.split(sep=' ')
        ','.join(ws)
        for herb in ws:
            per_vect.append(herb)
        file_dict[index] = per_vect
#%%

# 做成矩阵
file_dataframe = pd.DataFrame(columns=['herb'])
for index, row in file.iterrows():
    for h in row:
        per_frame = []
        hs = h.split(sep=' ')
        ','.join(hs)
        for herb in hs:
            herb = pd.DataFrame(herb, index=[index], columns=['herb'])
            file_dataframe = pd.concat(
                [file_dataframe, herb], axis=0, join='outer')
file_dataframe['count'] = 1
file_dataframe = file_dataframe.pivot_table(
    'count', index=file_dataframe.index, columns=['herb']).fillna(0)
#%%

# doc_vector是一个手动计算的tf向量
doc_len = len(ls)
doc_vector = []
for index, value in word_bag.most_common():
    doc_vector.append(value/doc_len)
#%%
# TF：词项归一化，某个词的出现频率除以文档中的词项总数
# 基于TF的点积字典
lexicon = sorted(set(ls))
zero_vector = OrderedDict((token, 0) for token in lexicon)
prescription_vector = dict()
for prescr_name in file_dict:
    ini_vect = cp.copy(zero_vector)
    herbs = file_dict.get(prescr_name)
    herbs_counts = Counter(herbs)
    for index, value in herbs_counts.items():
        ini_vect[index] = value/len(lexicon)
    prescription_vector[prescr_name] = ini_vect
#%%
# TF/IDF还没有做


# 点积公式  dist = np.dot(a_vect, b_vect)/np.linalg.norm(a_vect)*np.linalg.norm(b_vect)

# 基于TF的余弦相似度
tf_dot = pd.DataFrame()
#%%
for h1 in prescription_vector:
    matrix = pd.DataFrame()
    h_list = prescription_vector.get(h1)
    h1_vec = []
    for index1 in h_list:
        h1_vec.append(h_list.setdefault(index1))
    h1_vect = h1_vec
    for h2 in prescription_vector:

        h2_ordict = prescription_vector.get(h2)
        h2_vec = []
        for index2 in h2_ordict:
            h2_vec.append(h2_ordict.setdefault(index2))
        h2_vect = h2_vec
        h1_2_dot = np.dot(h1_vect, h2_vect) / \
            (np.linalg.norm(h1_vect)*np.linalg.norm(h2_vect))
        h1_2_dot = pd.DataFrame([h1_2_dot], columns=[h2], index=[h1])
        matrix = matrix.join(h1_2_dot, how='right')
    tf_dot = pd.concat([tf_dot, matrix], axis=0, join="outer")

#%%
# 基于密集矩阵的点积矩阵
matrix_dot = pd.DataFrame()
for index1, row1 in file_dataframe.iterrows():
    matrix = pd.DataFrame()
    series1 = np.array(file_dataframe.loc[index1])
    for index2, row2 in file_dataframe.iterrows():
        series2 = np.array(file_dataframe.loc[index2])
        series1_2_dot = series1.dot(series2)
        series1_2_dot = pd.DataFrame([series1_2_dot], columns=[
                                     index2], index=[index1])
        matrix = matrix.join(series1_2_dot, how='right')
    matrix_dot = pd.concat([matrix_dot, matrix], axis=0, join="outer")



#%%
#遍历txt，一个方一个字符串
tf_vect=[]
for index,row in txt.iterrows():
    for sen in row:
        sen_row=[]
        sent=sen.split(sep=' ')
        ','.join(sent)
        for herb in sent:
            sen_row.append(herb)
        tf_vect.append(sen_row)



#%%
tf_idf_dict = dict()
for tf_idf_name in file_dict:
    herbs = file_dict.get(tf_idf_name)
    tf_idf_value_list=[]
    tf_idf_value = tf_idf.transform(herbs)
    tf_idf_value = tf_idf_value.toarray().round(3)
    tf_idf_value_list.append(tf_idf_value)
    tf_idf_dict[tf_idf_name]=tf_idf_value_list


#%%
lexicon = sorted(set(ls))

#%%
tf_idf_dict = dict()

#%%
# 手动算tf-idf值，做成一个tf-idf的字典
for tf_pres_name in file_dict:
    ini_tf_vect = dict()
    herbs = file_dict.get(tf_pres_name)
    herbs_counts = Counter(herbs)
    for index, value in herbs_counts.items():
        docs_contain_key=0
        for herb_row in tf_vect:
            if (index in herb_row)==True:
                docs_contain_key = docs_contain_key+1
        tf = value / len(lexicon)
        if docs_contain_key:
            idf = len(txt.index)/docs_contain_key
        else:
            idf = 0
        ini_tf_vect[index] = tf * idf
    tf_idf_dict[tf_pres_name]=ini_tf_vect


#%%
tf_idf_matrix = pd.DataFrame()
#%%
# 把每一个方剂的tf-idf提取出来，并且找出最小的，就是公约，找出最大的，就是代表
for pre_name in tf_idf_dict:
    h_list = tf_idf_dict.get(pre_name)

#%%
# 最小值
min_h = min(zip(h_list.keys(),h_list.values()))
# 最大值
max_price = max(zip(prices.values(), prices.keys()))
# 排序，并且显示前4个
h_list_sorted = sorted(zip(h_list.keys(), h_list.values()))[0:4]

#%%
# pd.merge(tf_cos_sim, h1_2_dot,how='left')
# ([tf_cos_sim,h1_2_dot],axis=0,join='inner',
# ignore_index=False,join_axes=[tf_cos_sim.columns])
# 使用dataframe计算点积，更适合，能体现有多少味药物重合
# 余弦相似度计算，应该在基于TF-IDF的字典上计算，能够在忽略方剂中药物数量的情况下计算相似度
# 基于密集矩阵的点积能够体现出方剂中有多少味药物相同，但是相同并不能代表相近
# 基于词频向量的余弦相似度能够体现方剂的相似程度，而忽略方剂长度的影响









