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

english_example=convert_df(out=pd.read_excel('English example.xlsx'))
chinese_example=convert_df(out=pd.read_excel('中文示例.xlsx'))


with st.sidebar:
    file = st.file_uploader ("Click “Browse files” to upload files", type=["csv","xlsx", "xls"])
    st.write("You uploaded:", file)
    st.write('Please upload a file no larger than 200MB')
    st.write('The file must be a .csv,.xls or .xlsx file')
    st.download_button('Click here to download sample data in English',data=english_example,mime='csv')
    st.download_button('点击此处下载中文示例数据',data=chinese_example,mime='csv')