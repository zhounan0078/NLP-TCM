# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
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
sns.set_theme(style="whitegrid")
tab1, tab2, tab3, tab4 = st.tabs(
    ["Descriptive statistics", "Prescription similarity", "Topic distribution", "word embedding"])
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
font = FontProperties(fname="./data/SimHei.ttf", size=14)


# %%
def convert_df(out):
    return out.to_csv().encode('utf-8')


out1 = pd.read_csv('English example.csv')
out2 = pd.read_csv('中文示例.csv')
# %%
out1 = out1.set_index('Prescription name')
out2 = out2.set_index('方剂名称')
english_example = convert_df(out1)
chinese_example = convert_df(out2)
# %%
with st.sidebar:
    file = st.file_uploader("Click “Browse files” to upload files", type=["csv", "xlsx", "xls"])
    st.write('Please upload a file no larger than 200MB')
    st.write('The file must be a .csv,.xls or .xlsx file')
    st.download_button('download sample data in English', data=english_example, file_name='sample data in English.csv',
                       mime='csv')
    st.download_button('下载中文示例数据', data=chinese_example, file_name='中文示例数据.csv', mime='csv')
# %%

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
    Counter_every_herb = Counter(herb_word_list)
    total_herb_word_list = len(herb_word_list)
    with tab1:
        st.write('1.The total number')
        st.write('The total number of herbs is: ', total_herb_word_list)
        st.write('2.The most common herb')
        color = st.select_slider(
            'How many drugs do you need to display by frequency?',
            options=range(1, 50, 1))
        most_common_herb1 = Counter_every_herb.most_common(color)
        most_common_herb1 = pd.DataFrame(most_common_herb1, columns=['herb', 'count'])
        st.write('The most common herb is: ', most_common_herb1)
        if not most_common_herb1.empty:
            fig, ax = plt.subplots()
            x = most_common_herb1['herb']
            y = most_common_herb1['count']
            ax.bar(x, y, align='center', color='c', tick_label=list(x))
            ax.set_xticklabel(x, rotation=45, ha='right',FontProperties=font)
            ax.set_title('The most common herb', FontProperties=font)
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
    file_dict = dict()
    for index, row in txt.iterrows():
        for sen in row:
            per_vect = []
            ws = sen.split(sep=',')
            for herb in ws:
                per_vect.append(herb)
        file_dict[index] = per_vect
