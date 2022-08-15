#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from collections import Counter, OrderedDict
import copy as cp
from sklearn.decomposition import LatentDirichletAllocation as LDiA
import gensim
from PIL import Image
import streamlit as st  # For the web app
#%%
tab1, tab2, tab3, tab4 = st.tabs(["Descriptive statistics", "Prescription similarity", "Topic distribution","word embedding"])
#%%
def convert_df(out):
    return out.to_csv().encode('utf-8')
out1=pd.read_csv('English example.csv')
out2=pd.read_csv('中文示例.csv')
#%%
out1=out1.set_index('Prescription name')
out2=out2.set_index('方剂名称')
english_example=convert_df(out1)
chinese_example=convert_df(out2)
#%%
with st.sidebar:
    file = st.file_uploader ("Click “Browse files” to upload files", type=["csv","xlsx", "xls"])
    st.write('Please upload a file no larger than 200MB')
    st.write('The file must be a .csv,.xls or .xlsx file')
    st.download_button('download sample data in English',data=english_example,file_name='sample data in English.csv',mime='csv')
    st.download_button('下载中文示例数据',data=chinese_example,file_name='中文示例数据.csv',mime='csv')
#%%
#file=pd.read_csv('English example.csv')
if file != None:
    txt = pd.read_csv(file)
    txt = pd.DataFrame(txt)
    col=txt.columns
    txt = txt.set_index(col[0])
#%%
    sentence = ""
    for index, row in txt.iterrows():
        for sen in row:
            sentence = sentence+sen+','
    herb_word_list = sentence.split(sep=',')
    Counter_every_herb = Counter(herb_word_list)
    total_herb_word_list = len(herb_word_list)
    most_common_herb = Counter_every_herb.most_common()
    most_common_herb = pd.DataFrame(most_common_herb, columns=['herb', 'count'])
    with tab1:
        st.write('The total number of herbs is: ',total_herb_word_list)
        st.write('The most common herb is: ',most_common_herb)
        st.write()
#%%
    file_dict = dict()
    for index, row in txt.iterrows():
        for sen in row:
            per_vect = []
            ws = sen.split(sep=',')
            for herb in ws:
                per_vect.append(herb)
        file_dict[index] = per_vect


