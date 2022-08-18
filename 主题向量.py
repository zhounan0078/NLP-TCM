# %%
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from collections import Counter, OrderedDict
import copy as cp
from sklearn.decomposition import LatentDirichletAllocation as LDiA
import matplotlib.pyplot as plt
import gensim
import streamlit as st  # 导入streamlit


# %%
# 读文档，这里把方名set成index了，所以dataframe中只有草药
txt = pd.read_csv('English example.csv')
txt = pd.DataFrame(txt)
txt = txt.set_index('Prescription name')
# %%
# 遍历dataframe中的草药，把所有的草药怼成一个长字符串，空格分割
sentence = ""
for index, row in txt.iterrows():
    for sen in row:
        sentence = sentence + sen + ','
ls = sentence.split(sep=',')
# %%
word = sentence.split(',')
#' '.join(word)

# 把长字符转打散成一个list，这里的ls会在CountVectorized和TfidfVectorizer中用到，但是TreebankWordTokenizer不用
# %%

word_bag = Counter(ls)
word_bag.most_common()
print(len(word_bag))
print(len(ls))
# 重新读文件，这里不设方名为index
# %%
file = pd.read_csv('English example.csv')
file = pd.DataFrame(file)
# %%
file = file.set_index('Prescription name')

# %%
# 做成以{方名：组成}为键值对的字典
file_dict = dict()
for index, row in file.iterrows():
    for w in row:
        per_vect = []
        ws = w.split(sep=',')
        ','.join(ws)
        for herb in ws:
            per_vect.append(herb)
        file_dict[index] = per_vect
# %%
# 做成矩阵
herb_dense_dataframe = pd.DataFrame(columns=['herb'])
for index, row in file.iterrows():
    for h in row:
        per_frame = []
        hs = h.split(sep=' ')
        ','.join(hs)
        for herb in hs:
            herb = pd.DataFrame(herb, index=[index], columns=['herb'])
            herb_dense_dataframe = pd.concat(
                [herb_dense_dataframe, herb], axis=0, join='outer')
# %%
herb_dense_dataframe = pd.DataFrame(columns=['pres_name', 'herb_name'])
for pres_name in file_dict:
    herb_list = file_dict.get(pres_name)
    pres_name = [pres_name]
    pres_name = pd.DataFrame(pres_name, columns=['pres_name'])
    herb_dense_dataframe = pd.concat([herb_dense_dataframe, pres_name], axis=0, join='outer')
    for herb in herb_list:
        herb_df = pd.DataFrame(columns=['herb_name'])
        herb = [herb]
        herb = pd.DataFrame(herb, columns=['herb_name'])
        herb_df = pd.concat([herb_df, herb], axis=0, join='outer')
        herb_dense_dataframe = pd.concat([herb_dense_dataframe, herb_df], axis=0, join='outer')
# %%
herb_dense_dataframe['count'] = 1
# %%
herb_dense_dataframe['pres_name'] = herb_dense_dataframe['pres_name'].fillna(method='ffill')
# %%
herb_dense_dataframe.dropna(subset=['herb_name'], axis=0, inplace=True, how="any")
# %%
herb_dense_dataframe = herb_dense_dataframe.pivot_table(
    'count', index=herb_dense_dataframe['pres_name'], columns=['herb_name']).fillna(0)

# %%
herb_dense_dataframe.to_csv('herb_dense_dataframe.csv')

# %%
# TF：词项归一化，某个词的出现频率除以文档中的词项总数
# 基于TF的点积Ordered字典
lexicon = sorted(set(ls))
zero_vector_1 = OrderedDict((token, 0) for token in lexicon)
tf_vector_Orderdict = dict()
for prescr_name in file_dict:
    ini_vect = cp.copy(zero_vector_1)
    herbs = file_dict.get(prescr_name)
    herbs_counts = Counter(herbs)
    for index, value in herbs_counts.items():
        ini_vect[index] = value / len(lexicon)
    tf_vector_Orderdict[prescr_name] = ini_vect
# %%
lexicon = sorted(set(ls))
zero_vector_2 = dict()
tf_vector_dict = dict()
for prescr_name in file_dict:
    ini_vect = cp.copy(zero_vector_2)
    herbs = file_dict.get(prescr_name)
    herbs_counts = Counter(herbs)
    for index, value in herbs_counts.items():
        ini_vect[index] = value / len(lexicon)
    tf_vector_dict[prescr_name] = ini_vect

# %%
# 基于TF的余弦相似度
tf_cos_sim = pd.DataFrame()
# %%
for h1 in tf_vector_Orderdict:
    matrix = pd.DataFrame()
    h_list = tf_vector_Orderdict.get(h1)
    h1_vec = []
    for index1 in h_list:
        h1_vec.append(h_list.setdefault(index1))
    h1_vect = h1_vec
    for h2 in tf_vector_Orderdict:
        h2_ordict = tf_vector_Orderdict.get(h2)
        h2_vec = []
        for index2 in h2_ordict:
            h2_vec.append(h2_ordict.setdefault(index2))
        h2_vect = h2_vec
        h1_2_dot = np.dot(h1_vect, h2_vect) / \
                   (np.linalg.norm(h1_vect) * np.linalg.norm(h2_vect))
        h1_2_dot = pd.DataFrame([h1_2_dot], columns=[h2], index=[h1])
        matrix = matrix.join(h1_2_dot, how='right')
    tf_cos_sim = pd.concat([tf_cos_sim, matrix], axis=0, join="outer")

# %%
# 基于密集矩阵绝对值的点积矩阵
dense_dot = pd.DataFrame()
for index1, row1 in herb_dense_dataframe.iterrows():
    matrix = pd.DataFrame()
    series1 = np.array(herb_dense_dataframe.loc[index1])
    for index2, row2 in herb_dense_dataframe.iterrows():
        series2 = np.array(herb_dense_dataframe.loc[index2])
        series1_2_dot = np.dot(series1, series2) / \
                   (np.linalg.norm(series1) * np.linalg.norm(series2))
        series1_2_dot = pd.DataFrame([series1_2_dot], columns=[
            index2], index=[index1])
        matrix = matrix.join(series1_2_dot, how='right')
    dense_dot = pd.concat([dense_dot, matrix], axis=0, join="outer")
# %%
cos_dot = pd.DataFrame()
for index1, row1 in herb_dense_dataframe.iterrows():
    matrix = pd.DataFrame()
    cos_dot1 = np.array(herb_dense_dataframe.loc[index1])
    for index2, row2 in herb_dense_dataframe.iterrows():
        cos_dot2 = np.array(herb_dense_dataframe.loc[index2])
        cos1_2_dot = np.dot(cos_dot1, cos_dot2) / \
                     (np.linalg.norm(cos_dot1) * np.linalg.norm(cos_dot2))
        cos1_2_dot = pd.DataFrame([cos1_2_dot], columns=[
            index2], index=[index1])
        matrix = matrix.join(cos1_2_dot, how='right')
    cos_dot = pd.concat([cos_dot, matrix], axis=0, join="outer")
# %%
cos_dict = dict()
num2 = st.select_slider(
    'Please select the dot product value of the top herbs you want to view (in descending order)',
    options=range(1, 50, 1))

for index,row in cos_dot.iterrows():
    for value in row:
        index1= index
        index2= cos_dot.columns[cos_dot.loc[index]==value].values[0]
        dic_index=str(index1)+'×'+str(index2)
        value=float(value)
        cos_dict[dic_index]=value
value_df = pd.DataFrame.from_dict(cos_dict,orient="index",columns=['Cosine'])

# %%
for index,row in value_df.iterrows():
    if row==1:
        value_df.drop(value_df[value_df['Cosine']==1.000000].index)

value_df = value_df.drop(value_df[value_df['Cosine']==1.000000].index)
#%%




# %%
# 遍历txt，一个方一个字符串
list_vect = []
for index, row in file.iterrows():
    for sen in row:
        sen_row = []
        sent = sen.split(sep=',')
        ','.join(sent)
        for herb in sent:
            sen_row.append(herb)
        list_vect.append(sen_row)

# %%
lexicon = sorted(set(ls))
tf_idf_dict = dict()

# %%
# 手动算tf-idf值，做成一个tf-idf的字典
for tf_pres_name in file_dict:
    ini_tf_vect = dict()
    herbs = file_dict.get(tf_pres_name)
    herbs_counts = Counter(herbs)
    for index, value in herbs_counts.items():
        docs_contain_key = 0
        for herb_row in list_vect:
            if (index in herb_row) == True:
                docs_contain_key = docs_contain_key + 1
        tf = value / len(lexicon)
        if docs_contain_key != 0:
            idf = len(txt.index) / docs_contain_key
        else:
            idf = 0
        ini_tf_vect[index] = tf * idf
    tf_idf_dict[tf_pres_name] = ini_tf_vect
# %%
#
tf_idf_dataframe = pd.DataFrame()
#%%
for pres_name in tf_idf_dict:
    herb_tf_idf_dict = tf_idf_dict.get(pres_name)
    pres_name = [pres_name]
    pres_name = pd.DataFrame(pres_name, columns=['pres_name'])
    tf_idf_dataframe = pd.concat([tf_idf_dataframe, pres_name], axis=0, join='outer')
    for herb_name in herb_tf_idf_dict:
        herb_df = pd.DataFrame(columns=['herb_name', 'herb_tf_idf_value'])
        herb_tf_value = herb_tf_idf_dict.get(herb_name)
        herb_name = [herb_name]
        herb_name = pd.DataFrame(herb_name, columns=['herb_name'])
        herb_df = pd.concat([herb_df, herb_name], axis=0, join='outer')
        herb_tf_value = round(herb_tf_value, 3)
        herb_tf_value = [herb_tf_value]
        herb_tf_value = pd.DataFrame(herb_tf_value, columns=['herb_tf_idf_value'])
        herb_df = pd.concat([herb_df, herb_tf_value], axis=0, join='outer')
        tf_idf_dataframe = pd.concat([tf_idf_dataframe, herb_df], axis=0, join='outer')
#%%
idf_df = cp.copy(tf_idf_dataframe)
idf_df['pres_name'] = idf_df['pres_name'].fillna(method='ffill')
idf_df['herb_name'] = idf_df['herb_name'].fillna(method='ffill')
idf_df.dropna(subset=['herb_tf_idf_value'], axis=0, inplace=True, how="any")

idf_df = idf_df.pivot_table('herb_tf_idf_value', index=['pres_name'], columns=['herb_name']).fillna(round(0, 3))


#%%
def convert_df(out):
    return out.to_csv().encode('utf-8')
#%%
tf_idf_matrix = convert_df(idf_df)
#%%

# %%


# %%
herb_dense_dataframe['count'] = 1
# %%
herb_dense_dataframe['pres_name'] = herb_dense_dataframe['pres_name'].fillna(method='ffill')
# %%
herb_dense_dataframe.dropna(subset=['herb_name'], axis=0, inplace=True, how="any")
# %%
herb_dense_dataframe = herb_dense_dataframe.pivot_table(
    'count', index=herb_dense_dataframe['pres_name'], columns=['herb_name']).fillna(0)
# %%
tf_idf_matrix = pd.DataFrame.from_dict(tf_idf_dict,orient='index')
def convert_df(out):
    return out.to_csv().encode('utf-8')
#%%
tf_idf_matrix = convert_df(tf_idf_matrix)
# %%
# 使用TfidfVectorizer()的tf-idf
tf_idf_vectorizer = TfidfVectorizer()
tf_idf_vectorizer = (tf_idf_vectorizer.fit_transform(list_vect)).todense()
tf_idf_vectorizer = pd.DataFrame(tf_idf_vectorizer)
# %%
# 把每一个方剂的tf-idf提取出来，并且找出最小的，就是公约，找出最大的，就是代表
for pre_name in tf_idf_dict:
    h_list = tf_idf_dict.get(pre_name)

# %%
# 最小值
min_h = min(zip(h_list.keys(), h_list.values()))
# 最大值
max_h = max(zip(h_list.values(), h_list.keys()))
# 排序，并且显示前4个
h_list_sorted = sorted(zip(h_list.keys(), h_list.values()))[0:4]

# %%
# 把tf_idf_dict中的值遍历出来，做成密集矩阵
tf_idf_dataframe = pd.DataFrame(columns=['pres_name', 'herb_name', 'herb_tf_idf_value'])
# %%
for pres_name in tf_idf_dict:
    herb_tf_idf_dict = tf_idf_dict.get(pres_name)
    pres_name = [pres_name]
    pres_name = pd.DataFrame(pres_name, columns=['pres_name'])
    tf_idf_dataframe = pd.concat([tf_idf_dataframe, pres_name], axis=0, join='outer')
    for herb_name in herb_tf_idf_dict:
        herb_df = pd.DataFrame(columns=['herb_name', 'herb_tf_idf_value'])
        herb_tf_value = herb_tf_idf_dict.get(herb_name)
        herb_name = [herb_name]
        herb_name = pd.DataFrame(herb_name, columns=['herb_name'])
        herb_df = pd.concat([herb_df, herb_name], axis=0, join='outer')
        herb_tf_value = round(herb_tf_value, 3)
        herb_tf_value = [herb_tf_value]
        herb_tf_value = pd.DataFrame(herb_tf_value, columns=['herb_tf_idf_value'])
        herb_df = pd.concat([herb_df, herb_tf_value], axis=0, join='outer')
        tf_idf_dataframe = pd.concat([tf_idf_dataframe, herb_df], axis=0, join='outer')
# %%
idf_df = cp.copy(tf_idf_dataframe)
idf_df['pres_name'] = idf_df['pres_name'].fillna(method='ffill')
idf_df['herb_name'] = idf_df['herb_name'].fillna(method='ffill')
idf_df.dropna(subset=['herb_tf_idf_value'], axis=0, inplace=True, how="any")

idf_df = idf_df.pivot_table('herb_tf_idf_value', index=['pres_name'], columns=['herb_name']).fillna(round(0, 3))
# idf_df是tf-idf矩阵


# %%
tfidf = TfidfVectorizer(lowercase=False)
tfidf_doc = tfidf.fit(txt.text)
tfidf_matrix = tfidf.transform(txt.text)
tfidf_matrix = tfidf_matrix.todense()
tfidf_matrix = pd.DataFrame(tfidf_matrix)
#%%
idf_df=idf_df.T

#%%
c = np.corrcoef(idf_df)
#%%
np.linalg.eigvals(c)

# %%

column_nums, terms = zip(*sorted(zip(tfidf.vocabulary_.values(), tfidf.vocabulary_.keys())))

# %%
# PCA的主题向量
pca = PCA(n_components=2, svd_solver='full', random_state=123)
pca = pca.fit(idf_df)
pca_topic_vector = pca.transform(idf_df)
# %%
columns = ['topic{}'.format(i) for i in range(pca.n_components)]
pca_topic = pd.DataFrame(pca_topic_vector, columns=columns, index=idf_df.index)
pca_matrix=pca_topic.round(3)
pca_matrix.to_csv('pca_matrix.csv')

# %%
columns = ['topic{}'.format(i) for i in range(pca.n_components)]
weight1 = pd.DataFrame(pca.components_, columns=idf_df.index, index=['topic{}'.format(i) for i in range(3)])
pd.options.display.max_columns = 8
pca_herb_weight=weight1.round(3)
pca_herb_weight.to_csv('pca_herb_weight.csv')
# %%
# PCA的图
x = []
y = []
for i in range(1, 100):
    explvara_value = 0
    pca = PCA(n_components=i, svd_solver='full', random_state=123)
    pca = pca.fit(tfidf_matrix)
    x.append(i)
    explvara_list = list(pca.explained_variance_ratio_)
    for var in explvara_list:
        explvara_value=explvara_value+var
    y.append(explvara_value)
# %%
fig = plt.figure()
fig.set_figheight(10)
fig.set_figwidth(10)
plt.plot(x, y, linewidth=2.0)
plt.xlim = (1, 50)
plt.ylim = (min(y), max(y))
plt.set_xticks = np.arange(1, 50)
plt.set_yticks = np.arange(1, max(y), 100)

plt.axhline(y=min(y), c='r', ls='--', lw=2)
plt.axvline(x=x[y.index(min(y))], c='r', ls='--', lw=2)
plt.show()

# %%
# SVD的主题向量
svd = TruncatedSVD(n_components=3, n_iter=100, random_state=123)
svd_model = svd.fit(idf_df)
svd_topic = svd.transform(idf_df)
columns = ['topic{}'.format(i) for i in range(svd.n_components)]
svd_topic = pd.DataFrame(svd_topic, columns=columns, index=idf_df.index)
a = svd_topic.round(3)
# %%
a.to_csv('svd.csv')
# %%
columns = ['topic{}'.format(i) for i in range(svd.n_components)]
weight2 = pd.DataFrame(svd.components_, columns=idf_df.columns, index=['topic{}'.format(i) for i in range(3)])
pd.options.display.max_columns = 8
weight2.round(3)
weight2 = weight2.T
weight2.to_csv('svd_weight.csv')


#%%
idf_df=idf_df.T
#%%
idf_df = idf_df-idf_df.mean()
# %%
# SVD的图
#x = []
#y = []
#for i in range(1, 10):

    explvara_value = []
    svd = TruncatedSVD(n_components=20,random_state=123)
    svd = svd.fit(idf_df)


    explvara_list = list(svd.explained_variance_ratio_)
    sing = svd.singular_values_
    expl_cum = np.cumsum(explvara_list)
    plt.plot(explvara_list)
    plt.plot(expl_cum)
    plt.plot(sing)

# %%
fig = plt.figure()
fig.set_figheight(10)
fig.set_figwidth(10)
plt.plot(x, y, linewidth=2.0)
plt.xlim = (1, 50)
plt.ylim = (min(y), max(y))
plt.set_xticks = np.arange(1, 50)
plt.set_yticks = np.arange(1, max(y), 100)

plt.axhline(y=min(y), c='r', ls='--', lw=2)
plt.axvline(x=x[y.index(min(y))], c='r', ls='--', lw=2)
plt.show()

# %%
# 下面的是把词频向量做成dataframe
tf_df = pd.DataFrame(columns=['pres_name', 'herb_name', 'herb_tf_value'])
# %%
for pres_name in tf_vector_dict:
    herb_dict = tf_vector_dict.get(pres_name)
    pres_name = [pres_name]
    pres_name = pd.DataFrame(pres_name, columns=['pres_name'])
    tf_df = pd.concat([tf_df, pres_name], axis=0, join='outer')
    for herb_name in herb_dict:
        herb_df = pd.DataFrame(columns=['herb_name', 'herb_tf_value'])
        herb_tf_value = herb_dict.get(herb_name)
        herb_name = [herb_name]
        herb_name = pd.DataFrame(herb_name, columns=['herb_name'])
        herb_df = pd.concat([herb_df, herb_name], axis=0, join='outer')
        herb_tf_value = round(herb_tf_value, 8)
        herb_tf_value = [herb_tf_value]
        herb_tf_value = pd.DataFrame(herb_tf_value, columns=['herb_tf_value'])
        herb_df = pd.concat([herb_df, herb_tf_value], axis=0, join='outer')
        tf_df = pd.concat([tf_df, herb_df], axis=0, join='outer')
# %%
tf_dataf = cp.copy(tf_df)
tf_dataf['pres_name'] = tf_dataf['pres_name'].fillna(method='ffill')
tf_dataf['herb_name'] = tf_dataf['herb_name'].fillna(method='ffill')
tf_dataf.dropna(subset=['herb_tf_value'], axis=0, inplace=True, how="any")

tf_dataf = tf_dataf.pivot_table('herb_tf_value', index=['pres_name'], columns=['herb_name']).fillna(round(0, 3))

# %%
df = pd.read_csv('herb_dense_dataframe.csv')
df = df.set_index("pres_name")


# %%

ldia = LDiA(n_components=2)
ldia = ldia.fit(herb_dense_dataframe)

# %%
columns = ['topic{}'.format(i) for i in range(ldia.n_components)]
# %%
components_herb = pd.DataFrame(ldia.components_.T, index=herb_dense_dataframe.columns, columns=columns)
components_herb.to_csv('ldia2.csv')
# %%
components_pres = ldia.transform(herb_dense_dataframe)
components_pres = pd.DataFrame(components_pres, index=herb_dense_dataframe.index, columns=columns)
components_pres.to_csv('ldia.csv')

#%%
components_pres = components_pres.set_index('pres_name')
#%%
components_pres.rename(columns={'topic0':'topic1','topic1':'topic2','topic2':'topic3'},inplace=True)
#%%
topic1=pd.DataFrame(columns=['topic1'])
topic2=pd.DataFrame(columns=['topic2'])
topic3=pd.DataFrame(columns=['topic3'])
#%%
for index, row in components_pres.iterrows():
    i_list = []
    for i in row:
        i_list.append(i)
    k=max(i_list)
    d=components_pres.columns[components_pres.loc[index]==k].values.tolist()
    index=str(index)
    index=[index]
    index=pd.DataFrame(index,columns=d)
    if ('topic1' in d) == True:
        topic1=pd.concat([topic1,index],axis=0,join='outer')
    if ('topic2' in d) == True:
        topic2=pd.concat([topic2,index],axis=0,join='outer')
    if ('topic3' in d) == True:
        topic3=pd.concat([topic3,index],axis=0,join='outer')
#%%
topic1=pd.DataFrame(columns=['topic1'])
topic2=pd.DataFrame(columns=['topic2'])
topic3=pd.DataFrame(columns=['topic3'])
#%%
for index, row in components_herb.iterrows():
    i_list = []
    for i in row:
        i_list.append(i)
    k=max(i_list)
    d=components_herb.columns[components_herb.loc[index]==k].values.tolist()
    index=str(index)
    index=[index]
    index=pd.DataFrame(index,columns=d)
    if ('topic1' in d) == True:
        topic1=pd.concat([topic1,index],axis=0,join='outer')
    if ('topic2' in d) == True:
        topic2=pd.concat([topic2,index],axis=0,join='outer')
    if ('topic3' in d) == True:
        topic3=pd.concat([topic3,index],axis=0,join='outer')


#%%
for index,row in components_pres.iterrows():
    i_list = []
    for i in row:
        i_list.append(i)
    locs=components_pres.loc(components_pres.loc(index)==max(i_list)).columns.tolist()[0]





# %%
# perplexity困惑度下降不明显的点就是合适的取值点，用for 循环一个n_components进去，求出不同的困惑度
df = pd.read_csv('herb_dense_dataframe.csv')
df = df.set_index("pres_name")
# %%
x = []
y = []
for i in range(1, 10):
    ldia = LDiA(n_components=i, learning_method='batch', evaluate_every=1, verbose=1, max_iter=100)
    ldia = ldia.fit(herb_dense_dataframe.csv)

    plex = ldia.perplexity(herb_dense_dataframe.csv)
    x.append(i)
    y.append(plex)
# %%
fig = plt.figure()
fig.set_figheight(10)
fig.set_figwidth(10)
plt.plot(x, y, linewidth=2.0)
plt.xlim = (1, 50)
plt.ylim = (min(y), max(y))
plt.set_xticks = np.arange(1, 50)
plt.set_yticks = np.arange(1, max(y), 100)

plt.axhline(y=min(y), c='r', ls='--', lw=2)
plt.axvline(x=x[y.index(min(y))], c='r', ls='--', lw=2)
plt.show()

#%%
word_vec_senten=[]
#%%
num_sen=0

#%%
for index,row in herb_dense_dataframe.iterrows():
    word_herb=herb_dense_dataframe.columns[herb_dense_dataframe.loc[index]==1].values.tolist()
    word_vec_senten.append(word_herb)


#%%
model = gensim.models.Word2Vec(word_vec_senten,sg=0,min_count=1,vector_size = 300,window=20)
#%%
model.wv['川芎']
#%%
a=pd.DataFrame(model.wv.index_to_key,columns=['name'])
#%%
b=pd.DataFrame(model.wv.vectors,index=a['name'])
#%%
pca = PCA(n_components=2,random_state=123)
#%%
pca = pca.fit(b)
pca_vectr = pca.transform(b)
#%%
columns = ['topic{}'.format(i) for i in range(pca.n_components)]
pca_topic = pd.DataFrame(pca_vectr, columns=columns, index=b.index)
pca_matrix=pca_topic.round(3)

#%%
x=pca_matrix['topic0']
y=pca_matrix['topic1']
#%%
plt.scatter(x,y,marker='*')




