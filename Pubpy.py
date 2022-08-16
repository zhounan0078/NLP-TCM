# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from collections import Counter, OrderedDict
import copy as cp
from sklearn.decomposition import LatentDirichletAllocation as LDiA
import gensim
from PIL import Image
import streamlit as st  # For the web app

# %%
# 全局设置区
sns.set_theme(style="whitegrid")
tab1, tab2, tab3, tab4,tab5 = st.tabs(
    ["Descriptive statistics", "Matrix display and download","Prescription similarity", "Topic distribution", "word embedding"])
mpl.rcParams['font.family'] = 'simhei.ttf'
plt.style.use('ggplot')
font = font_manager.FontProperties(fname="simhei.ttf", size=14)
# %%
# 定义文件转换csv函数
def convert_df(out):
    return out.to_csv().encode('utf-8')
#读取并转换示例数据
out1 = pd.read_csv('English example.csv')
out2 = pd.read_csv('中文示例.csv')
out1 = out1.set_index('Prescription name')
out2 = out2.set_index('方剂名称')
english_example = convert_df(out1)
chinese_example = convert_df(out2)
# %%
# 侧栏上传文件区域
with st.sidebar:
    file = st.file_uploader("Click “Browse files” to upload files", type=["csv", "xlsx", "xls"])
    st.write('Please upload a file no larger than 200MB')
    st.write('The file must be a .csv,.xls or .xlsx file')
    st.download_button('download sample data in English', data=english_example, file_name='sample data in English.csv',
                       mime='csv')
    st.download_button('下载中文示例数据', data=chinese_example, file_name='中文示例数据.csv', mime='csv')
# %%
# 描述性统计处理
if file != None:
    txt = pd.read_csv(file)
    txt = pd.DataFrame(txt)
    col = txt.columns
    txt = txt.set_index(col[0])
    sentence = ""
    for index, row in txt.iterrows():
        for sen in row:
            sentence = sentence + sen + ','
    herb_word_list = sentence.split(sep=',')
    # 做成字典
    file_dict = dict()
    for index, row in txt.iterrows():
        for sen in row:
            per_vect = []
            ws = sen.split(sep=',')
            for herb in ws:
                per_vect.append(herb)
        file_dict[index] = per_vect
    # 平均长度
    len_herb_list=0
    for index in file_dict:
        herb_list = file_dict.get(index)
        herb_list = list(set(herb_list))
        len_list=len(herb_list)
        len_herb_list=len_herb_list+len_list
    total_len=len(file_dict.keys())
    avg_len=len_herb_list/total_len
    # 词数统计
    Counter_every_herb = Counter(herb_word_list)
    total_herb_word_list = len(herb_word_list)
    #显示统计结果
    with tab1:
        st.write('1.The total number of herbs is: ', total_herb_word_list)
        st.write('2.The average length of prescription: ', round(avg_len,0))
        st.write('3.The most common herb')
        color = st.select_slider(
            'How many drugs do you need to display by frequency?',
            options=range(1, 50, 1))
        most_common_herb1 = Counter_every_herb.most_common(color)
        most_common_herb1 = pd.DataFrame(most_common_herb1, columns=['herb', 'count'])
        st.write('The most common herb is: ', most_common_herb1)
        #作图
        if most_common_herb1.empty == False:
            fig, ax = plt.subplots()
            x = most_common_herb1['herb']
            y = most_common_herb1['count']
            ax.barh(x, y, align='center', color='c', tick_label=list(x))
            plt.xlabel('herbs', fontsize=13, fontproperties=font)
            plt.yticks(x, fontproperties=font)
            st.pyplot(fig)
        most_common_herb2 = Counter_every_herb.most_common()
        most_common_herb2 = pd.DataFrame(most_common_herb2, columns=['herb', 'count'])
        full_common_data = convert_df(most_common_herb2)
        st.download_button(
            label="Download full herb frequency data",
            data=full_common_data,
            file_name='full_common_data.csv',
            mime='csv', )
    # %%
    # 做成矩阵
    with tab2:
        #密集矩阵
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
        herb_dense_dataframe['count'] = 1
        herb_dense_dataframe['pres_name'] = herb_dense_dataframe['pres_name'].fillna(method='ffill')
        herb_dense_dataframe.dropna(subset=['herb_name'], axis=0, inplace=True, how="any")
        herb_dense_dataframe = herb_dense_dataframe.pivot_table(
        'count', index=herb_dense_dataframe['pres_name'], columns=['herb_name']).fillna(0)
        st.table(herb_dense_dataframe)
        herb_dense_dataframe = convert_df(herb_dense_dataframe)
        st.download_button('download dense matrix', data=herb_dense_dataframe, file_name='dense matrix.csv',
                           mime='csv')
        #tf-idf矩阵
        list_vect = []
        for index, row in txt.iterrows():
            for sen in row:
                sen_row = []
                sent = sen.split(sep=' ')
                ','.join(sent)
                for herb in sent:
                    sen_row.append(herb)
        list_vect.append(sen_row)
        lexicon = sorted(set(herb_word_list))
        tf_idf_dict = dict()
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
        tf_idf_matrix = pd.DataFrame.from_dict(tf_idf_dict,orient='index')
        st.table(tf_idf_matrix)
        tf_idf_matrix = convert_df(tf_idf_matrix)
        st.download_button('download tf_idf_matrix', data=tf_idf_matrix, file_name='tf_idf_matrix.csv',
                           mime='csv')
