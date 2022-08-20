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
# 全局设置
sns.set_theme(style="whitegrid")
tab1, tab2, tab3, tab4,tab5,tab6 = st.tabs(
    ["Descriptive statistics","Prescription similarity", "Topic distribution", "word embedding","Matrix download","About the program"])
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
#定义文件读取函数
def txt_read(file):
    try:
        txt = pd.read_csv(file)
        txt = pd.DataFrame(txt)
        col = txt.columns
        txt = txt.set_index(col[0])
        return txt
    except:
        st.write("Please upload a file")
        txt = pd.DataFrame(out1)
        col = txt.columns
        txt = txt.set_index(col[0])
        return txt
# %%
# 侧栏上传文件区域
with st.sidebar:

    file = st.file_uploader("Click “Browse files” to upload files", type=["csv", "xlsx", "xls"])
    st.write('Please upload a file no larger than 200MB')
    st.write('The file must be a .csv,.xls or .xlsx file')
    st.download_button('download sample data in English', data=english_example, file_name='sample data in English.csv',
                       mime='csv')
    st.download_button('下载中文示例数据', data=chinese_example, file_name='中文示例数据.csv', mime='csv')
    st.write('Note: You can understand the workflow of this program by uploading sample data.')
    st.write('Note: When the program is running, there will be a little man doing sports in the upper right corner of the web page,don\`t refresh this page or do anything else until he stops.')
# %%
#测试区
#file=pd.read_csv("English example.csv")
#%%
# 描述性统计处理
#if file.empty == False:
#if file != None:
#    txt = pd.read_csv(file)
#    txt = pd.DataFrame(txt)
#    col = txt.columns
#    txt = txt.set_index(col[0])
txt_read(file)

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
total_herb_list = len(Counter_every_herb)
total_herb_word_list = len(herb_word_list)
#%%
#显示统计结果
with tab1:
    st.write('1.The total number of different herbs: ', total_herb_list)
    st.write('2.The total number of herbs is:', total_herb_word_list)
    st.write('3.The average length of prescription: ', round(avg_len,0))
    st.write('4.The most common herb')
    num1 = st.select_slider(
        'How many herbs do you need to display by frequency?',
        options=range(1, 50, 1),key=1)
    if st.button('Launch',key=2):
        most_common_herb1 = Counter_every_herb.most_common(num1)
        most_common_herb1 = pd.DataFrame(most_common_herb1, columns=['herb', 'count'])
        st.write('The most common herb is: ', most_common_herb1)
        #作图
        if most_common_herb1.empty == False:
            fig1, ax1 = plt.subplots()
            x = most_common_herb1['herb']
            y = most_common_herb1['count']
            y = list(y)
            y.reverse() # 倒序
            ax.barh(x, y, align='center', color='c', tick_label=list(x))
            plt.ylabel('herbs', fontsize=13, fontproperties=font)
            plt.yticks(x, fontproperties=font)
            st.pyplot(fig)
    most_common_herb2 = Counter_every_herb.most_common()
    most_common_herb2 = pd.DataFrame(most_common_herb2, columns=['herb', 'count'])
    #矩阵制作
    #频次矩阵
    full_common_data = convert_df(most_common_herb2)
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
    #tf-idf矩阵
    list_vect = []
    for index, row in txt.iterrows():
        for sen in row:
            sen_row = []
            sent = sen.split(sep=',')
            ','.join(sent)
            for herb in sent:
                sen_row.append(herb)
            list_vect.append(sen_row)
    # 手动计算tf-idf
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
    #遍历成dataframe
    tf_idf_dataframe = pd.DataFrame(columns=['pres_name', 'herb_name'])
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
    idf_df = cp.copy(tf_idf_dataframe)
    idf_df['pres_name'] = idf_df['pres_name'].fillna(method='ffill')
    idf_df['herb_name'] = idf_df['herb_name'].fillna(method='ffill')
    idf_df.dropna(subset=['herb_tf_idf_value'], axis=0, inplace=True, how="any")
    idf_df = idf_df.pivot_table('herb_tf_idf_value', index=['pres_name'], columns=['herb_name']).fillna(round(0, 3))

with tab2:
#Dot product calculation
    st.write('1.Dot product')
    st.write('The dot product value reflects how many of the same herbs are present between the two prescriptions.')
    dense_dot = pd.DataFrame()
    for index1, row1 in herb_dense_dataframe.iterrows():
        matrix = pd.DataFrame()
        series1 = np.array(herb_dense_dataframe.loc[index1])
        for index2, row2 in herb_dense_dataframe.iterrows():
            series2 = np.array(herb_dense_dataframe.loc[index2])
            series1_2_dot = series1.dot(series2)
            series1_2_dot = pd.DataFrame([series1_2_dot], columns=[
                index2], index=[index1])
            matrix = matrix.join(series1_2_dot, how='right')
        dense_dot = pd.concat([dense_dot, matrix], axis=0, join="outer")
    dot_df = pd.DataFrame(columns=['index1', 'index2', 'Quantity of the same herb'])
    for index,row in dense_dot.iterrows():
        for value1 in row:
            index1= index
            index2= dense_dot.columns[dense_dot.loc[index]==value1].values[0]
            if index1==index2:
                continue
            else:
                if (index1 in list(dot_df['index2']))==True and (index2 in list(dot_df['index1']))==True:
                    continue
                else:
                    dot_df = dot_df.append({'index1':index1,'index2':index2,'Quantity of the same herb':value1},ignore_index=True)
    dot_df["Prescription"] =dot_df["index1"].map(str) + '×' + dot_df["index2"].map(str)
    dot_df=dot_df.drop(['index1','index2'],axis=1)
    dot_df=dot_df.set_index("Prescription")
    dot_df = dot_df.sort_values(by=['Quantity of the same herb'], ascending=False)
    num2 = st.select_slider(
        'Please select the dot product value of the top prescription you want to view (in descending order)',
        options=range(1, 50, 1),key=3)
    if st.button('Launch',key=4):
        if dot_df.empty==False:
            st.table(dot_df.head(num2))
    #cosine similarity
    st.write('2.Cosine similarity')
    st.write('Cosine similarity reflects how similar two prescriptions use herbs.')
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
    cos_df = pd.DataFrame(columns=['index1', 'index2', 'Cosine similarity'])
    for index,row in cos_dot.iterrows():
        for value2 in row:
            index1= index
            index2= cos_dot.columns[cos_dot.loc[index]==value2].values[0]
            if index1==index2:
                continue
            else:
                if (index1 in list(cos_df['index2']))==True and (index2 in list(cos_df['index1']))==True:
                    continue
                else:
                    cos_df = cos_df.append({'index1':index1,'index2':index2,'Cosine similarity':value2},ignore_index=True)
    cos_df["Prescription"] =cos_df["index1"].map(str) + '×' + cos_df["index2"].map(str)
    cos_df=cos_df.drop(['index1','index2'],axis=1)
    cos_df=cos_df.set_index("Prescription")
    cos_df = cos_df.sort_values(by=['Cosine similarity'], ascending=False)
    num3 = st.select_slider(
        'Please select the cosine similarity of the top prescription you want to view (in descending order)',
        options=range(1, 50, 1),key=5)
    if st.button('Launch',key=6):
        if cos_df.empty==False:
            st.table(cos_df.head(num3))
    #Freedom of choice
    st.write('3.Focus on dot product and cosine similarity for a specific prescription')
    options=list(txt.index)
    select_result=st.multiselect(
        'Please select the name of the prescription you wish to follow',options=options,key=7)
    dense_dot_df=pd.DataFrame()
    cos_dot_df=pd.DataFrame()
    if st.button('Launch',key=8):
        for item1 in select_result:
            dense_dot_matrix=pd.DataFrame()
            cos_dot_matrix=pd.DataFrame()
            for item2 in select_result:
                dense_dot_result = dense_dot.loc[[item1],[item2]]
                dense_dot_result = pd.DataFrame(dense_dot_result, columns=[item2], index=[item1])
                cos_dot_result = cos_dot.loc[[item1],[item2]]
                cos_dot_result = pd.DataFrame(cos_dot_result, columns=[item2], index=[item1])
                dense_dot_matrix = dense_dot_matrix.join(dense_dot_result, how='right')
                cos_dot_matrix = cos_dot_matrix.join(cos_dot_result, how='right')
            dense_dot_df = pd.concat([dense_dot_df, dense_dot_matrix], axis=0, join="outer")
            cos_dot_df = pd.concat([cos_dot_df, cos_dot_matrix], axis=0, join="outer")
        fig2, ax2 = plt.subplots()
        sns.heatmap(dense_dot_df, annot=True,fmt=".2g", linewidths=.5, cmap='YlOrRd')
        ax2.set_title('Dot product')
        st.pyplot(fig2)
        fig3, ax3 = plt.subplots()
        sns.heatmap(cos_dot_df, annot=True,fmt=".2g", linewidths=.5, cmap='YlGnBu')
        ax3.set_title('Cosine similarity')
        st.pyplot(fig3)







# %%
# 矩阵下载
with tab5:
    #频次矩阵下载
    st.download_button(
        label="Download full herb frequency data",
        data=full_common_data,
        file_name='full_common_data.csv',
        mime='csv', )
    #密集矩阵下载
    if herb_dense_dataframe.empty == False:
        herb_dense_dataframe = convert_df(herb_dense_dataframe)
        st.download_button(
            label='Download dense matrix',
            data=herb_dense_dataframe,
            file_name='dense matrix.csv',
            mime='csv')
    #tf-idf矩阵下载
    if idf_df.empty == False:
        tf_idf_matrix = convert_df(idf_df)
        st.download_button(
            label='Download tf_idf_matrix',
            data=tf_idf_matrix,
            file_name='tf_idf_matrix.csv',
            mime='csv')
    #dot product矩阵下载
    if dense_dot.empty == False:
        dense_dot = convert_df(dense_dot)
        st.download_button(
            label='Download dot_product_matrix',
            data=dense_dot,
            file_name='dense dot product.csv',
            mime='csv')
    #cosine similarity矩阵下载
    if cos_dot.empty == False:
        cos_dot = convert_df(cos_dot)
        st.download_button(
            label='Download cosine similarity matrix',
            data=cos_dot,
            file_name='cosine similarity.csv',
            mime='csv')





with tab4:
    placeholder1=st.empty()
    file = placeholder1.file_uploader("Click “Browse files” to upload files", type=["csv", "xlsx", "xls"],key=3)
    txt=txt_read(file)
    placeholder2=st.empty()
    placeholder2.table(txt)

    with st.container():
        st.write("This is inside the container")

    # You can call any Streamlit command, including custom components:
        st.bar_chart(np.random.randn(50, 3))


with tab6:

    st.write('Author information:')
    st.write('Name: Zhou Nan')
    st.write('Current situation: PhD student,Universiti Tunku Abdul Rahman(UTAR)')
    st.write('Mail_1:zhounan@1utar.my')
    st.write('Mail_2:zhounan2020@foxmail.com')
    st.write('Due to Streamlit\'s IO capability limitations, this program does not perform well when dealing with larger data sets. If you think this program cannot meet your needs or is always stuck in use, you can contact the author directly, you will get help.')

