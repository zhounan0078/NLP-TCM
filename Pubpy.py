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
    txt = txt.set_index(col[1])
    st.dataframe(txt)