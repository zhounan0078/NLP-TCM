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
import streamlit as st  # For the web app
#%%
with st.sidebar:
    file = st.file_uploader ("Click “Browse files” to upload files", type=["csv","xlsx", "xls"])
    st.write("You uploaded:", file)
    st.write('Please upload a file no larger than 200MB')
    st.write('The file must be a .csv,.xls or .xlsx file')
