# %%
import numpy as np
import pandas as pd
from io import BytesIO
from xlsxwriter import Workbook
from pyxlsb import open_workbook as open_xlsb
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
import streamlit as st
import altair as alt

# %%
# 全局设置
sns.set_theme(style="whitegrid")

tab1, tab2, tab3, tab4, tab5, tab6,tab7,tab8 = st.tabs(
    ["Descriptive statistics", "Prescription similarity", "Featured and generic herbs",
     "LSA topic distribution", "LDiA topic distribution","word embedding", "Download",
     "About the program"])
mpl.rcParams['font.family'] = 'simhei.ttf'
plt.style.use('ggplot')
font = font_manager.FontProperties(fname="simhei.ttf", size=14)
sns.set(font='simhei.ttf')


# %%
# 定义文件转换csv函数
def convert_df(df):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Sheet1')
        writer.save()
        processed_data = buffer.getvalue()
    return processed_data



# 读取并转换示例数据
out1 = pd.read_excel('English example.xlsx')
out2 = pd.read_excel('中文示例.xlsx')
out1 = out1.set_index('Prescription name')
out2 = out2.set_index('方剂名称')
english_example = convert_df(out1)
chinese_example = convert_df(out2)
# %%
# 侧栏上传文件区域
with st.sidebar:
    file = st.file_uploader("Click “Browse files” to upload files", type=["xlsx"])
    st.write('Please upload a file no larger than 200MB')
    st.write('The file must be a .xlsx file')
    st.download_button('Download sample data in English', data=english_example, file_name='sample data in English.xlsx',
                       )
    st.download_button('下载中文示例数据', data=chinese_example, file_name='中文示例数据.xlsx')
    st.write('Note: You can understand the workflow of this program by uploading sample data.')
    st.write(
        'Note: When the program is running, there will be a little man doing sports in the upper right corner of the web page,don\`t refresh this page or do anything else until he stops.')


# %%
# 测试区
# file=pd.read_csv("English example.csv")
# %%
# 描述性统计处理
# 定义文件读取函数
def txt_read(files):
    if file != None:
        txt = pd.read_excel(files)
        txt = pd.DataFrame(txt)
        col = txt.columns
        txt = txt.set_index(col[0])
        return txt
    else:
        out1 = pd.read_excel('English example.xlsx')
        with tab1:
            st.header(
                "What you see so far is the result of running the English example data,please refer to the example upload data")
        txt = pd.DataFrame(out1)
        col = txt.columns
        txt = txt.set_index(col[0])
        return txt


txt = txt_read(files=file)
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
len_herb_list = 0
for index in file_dict:
    herb_list = file_dict.get(index)
    herb_list = list(set(herb_list))
    len_list = len(herb_list)
    len_herb_list = len_herb_list + len_list
total_len = len(file_dict.keys())
avg_len = len_herb_list / total_len
# 词数统计
Counter_every_herb = Counter(herb_word_list)
total_herb_list = len(Counter_every_herb)
total_herb_word_list = len(herb_word_list)
# %%
# 显示统计结果
st.write('You can use the cursor keys "←" and "→" to see more tags')
with tab1:
    st.write('1.The total number of different herbs: ', total_herb_list)
    st.write('2.The total number of herbs is:', total_herb_word_list)
    st.write('3.The average length of prescription: ', round(avg_len, 0))
    st.write('4.The most common herb')
    num1 = st.select_slider(
        'How many herbs do you need to display by frequency?',
        options=range(1, 50, 1), key=1)

    most_common_herb1 = Counter_every_herb.most_common(num1)
    most_common_herb1 = pd.DataFrame(most_common_herb1, columns=['herb', 'count'])
    if st.button('Launch', key=2):

          st.write('The most common herb is: ')
          st.table(most_common_herb1,use_column_names=True)


        #
          if most_common_herb1.empty == False:
              fig1, ax1 = plt.subplots()
              x = most_common_herb1['herb']
              y = most_common_herb1['count']
              y = list(y)
              y.reverse()  # 倒序
              ax1.barh(x, y, align='center', color='c', tick_label=list(x))
              plt.ylabel('herbs', fontsize=13, fontproperties=font)
              plt.yticks(x, fontproperties=font)
              st.pyplot(fig1)
    most_common_herb2 = Counter_every_herb.most_common()
    most_common_herb2 = pd.DataFrame(most_common_herb2, columns=['herb', 'count'])
    # 矩阵制作
    # 频次矩阵
    full_common_data = most_common_herb2.copy()
    # 密集矩阵
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
    # tf-idf矩阵
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
    # 遍历成dataframe
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
    # Dot product calculation
    st.subheader('1.Dot product')
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
    for index, row in dense_dot.iterrows():
        for value1 in row:
            index1 = index
            index2 = dense_dot.columns[dense_dot.loc[index] == value1].values[0]
            if index1 == index2:
                continue
            else:
                if (index1 in list(dot_df['index2'])) == True and (index2 in list(dot_df['index1'])) == True:
                    continue
                else:
                    dot_df = dot_df.append({'index1': index1, 'index2': index2, 'Quantity of the same herb': value1},
                                           ignore_index=True)
    dot_df["Prescription"] = dot_df["index1"].map(str) + '×' + dot_df["index2"].map(str)
    dot_df = dot_df.drop(['index1', 'index2'], axis=1)
    dot_df = dot_df.set_index("Prescription")
    dot_df = dot_df.sort_values(by=['Quantity of the same herb'], ascending=False)
    num2 = st.select_slider(
        'Please select the dot product value of the top prescription you want to view (in descending order)',
        options=range(1, 50, 1), key=3)
    if st.button('Launch', key=4):
        if dot_df.empty == False:
            st.table(dot_df.head(num2))
    # cosine similarity
    st.subheader('2.Cosine similarity')
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
    for index, row in cos_dot.iterrows():
        for value2 in row:
            index1 = index
            index2 = cos_dot.columns[cos_dot.loc[index] == value2].values[0]
            if index1 == index2:
                continue
            else:
                if (index1 in list(cos_df['index2'])) == True and (index2 in list(cos_df['index1'])) == True:
                    continue
                else:
                    cos_df = cos_df.append({'index1': index1, 'index2': index2, 'Cosine similarity': value2},
                                           ignore_index=True)
    cos_df["Prescription"] = cos_df["index1"].map(str) + '×' + cos_df["index2"].map(str)
    cos_df = cos_df.drop(['index1', 'index2'], axis=1)
    cos_df = cos_df.set_index("Prescription")
    cos_df = cos_df.sort_values(by=['Cosine similarity'], ascending=False)
    num3 = st.select_slider(
        'Please select the cosine similarity of the top prescription you want to view (in descending order)',
        options=range(1, 50, 1), key=5)
    if st.button('Launch', key=6):
        if cos_df.empty == False:
            st.table(cos_df.head(num3))
    # Freedom of choice
    st.write('3.Focus on dot product and cosine similarity for a specific prescription')
    options = list(txt.index)
    select_result = st.multiselect(
        'Please select the name of the prescription you wish to follow', options=options, key=7)
    dense_dot_df = pd.DataFrame()
    cos_dot_df = pd.DataFrame()
    if st.button('Launch', key=8):
        for item1 in select_result:
            dense_dot_matrix = pd.DataFrame()
            cos_dot_matrix = pd.DataFrame()
            for item2 in select_result:
                dense_dot_result = dense_dot.loc[[item1], [item2]]
                dense_dot_result = pd.DataFrame(dense_dot_result, columns=[item2], index=[item1])
                cos_dot_result = cos_dot.loc[[item1], [item2]]
                cos_dot_result = pd.DataFrame(cos_dot_result, columns=[item2], index=[item1])
                dense_dot_matrix = dense_dot_matrix.join(dense_dot_result, how='right')
                cos_dot_matrix = cos_dot_matrix.join(cos_dot_result, how='right')
            dense_dot_df = pd.concat([dense_dot_df, dense_dot_matrix], axis=0, join="outer")
            cos_dot_df = pd.concat([cos_dot_df, cos_dot_matrix], axis=0, join="outer")

        fig2, ax2 = plt.subplots()
        sns.heatmap(dense_dot_df, annot=True, fmt=".2g", linewidths=.5, cmap='YlOrRd')
        ax2.set_title('Dot product')
        plt.xticks(font=font)
        plt.yticks(font=font)
        st.pyplot(fig2)

        fig3, ax3 = plt.subplots()
        sns.heatmap(cos_dot_df, annot=True, fmt=".2g", linewidths=.5, cmap='YlGnBu')
        ax3.set_title('Cosine similarity')
        plt.xticks(font=font)
        plt.yticks(font=font)
        st.pyplot(fig3)
with tab3:
    tf_idf_sort= pd.DataFrame(columns=['herb', 'tf_idf_value'])
    for index, row in idf_df.iterrows():
        for idf_i in row:
            idf_value= idf_i
            herb_name= idf_df.columns[idf_df.loc[index] == idf_i].values[0]
            tf_idf_sort= tf_idf_sort.append({'herb': herb_name, 'tf_idf_value': idf_value}, ignore_index=True)
    tf_idf_sort= tf_idf_sort.sort_values(by=['tf_idf_value'], ascending=False)
    tf_idf_sort = tf_idf_sort.set_index("herb")
    st.success("The calculation of Featured and generic herbs has been completed, please select the number of herbs you need to present")
    num7 = st.select_slider(
        'Please select the number of themes you wish to try',
        options=range(1, 50, 1), key=7)
    tab3_col1,tab3_col2 = st.columns(2)

    idf_button_con = st.button('Continue', key=13)
    if idf_button_con:
        with tab3_col1:
            st.write('Top 10 most important herbs')
            st.table(tf_idf_sort.head(num7))
            idf_x1= list((tf_idf_sort.head(num7)).index)
            idf_y1= list((tf_idf_sort['tf_idf_value'].head(num7)))
            plt.bar(idf_x1,idf_y1)
            st.pyplot(plt)
        with tab3_col2:
            st.write('Bottom 10 most important herbs')
            st.table(tf_idf_sort.tail(num7))
            idf_x2= list((tf_idf_sort.tail(num7)).index)
            idf_y2= list((tf_idf_sort['tf_idf_value'].tail(num7)))
            plt.bar(idf_x2,idf_y2)
            st.pyplot(plt)

with tab4:
    st.subheader('Topic classification based on Latent Semantic Analysis (LSA)')
    num4 = st.select_slider(
        'Please select the number of themes you wish to try',
        options=range(1, 100, 1), key=5)
    svd_button_pressed = st.button('Launch', key=9)
    # svd = TruncatedSVD()
    if svd_button_pressed == True:
        if num4 < len(txt.index):
            df = idf_df.T
            svd = TruncatedSVD(n_components=num4, n_iter=10, random_state=123)
            svd_model = svd.fit(df)
            svd_topic = svd.transform(df)
            explvara_list = list(svd.explained_variance_ratio_)
            sing = svd_model.singular_values_
            expl_cum = np.cumsum(explvara_list)
            lsa_topic= pd.DataFrame({'topic': range(1, num4+1), 'explained_variance': explvara_list, 'cumulative_explained_variance': expl_cum, 'singular_values': sing})
            lsa_topic= lsa_topic.set_index('topic')
            st.line_chart(lsa_topic)
            with st.expander("See explanation"):
                st.write('Explained variance ratio: The amount of information extracted by the topic can be understood as the weight of different topics. The higher the weight, the more information the topic can extract from the document. The lower the weight, the less information the topic can extract from the document. The weight of a topic is the square root of the sum of the square of the singular values of the topic. The weight of a topic is the square root of the sum of the square of the singular values of the topic.')
                st.write('Cumulative explained variance ratio: Under the current number of topics, the total amount of information extracted by all topics, this indicator needs to be at least greater than 50%')
                st.write('Singular values: The singular values of the topic are the square root of the sum of the square of the singular values of the topic,determines the number of topics when the downtrend in this value begins to flatten')
        else:
            st.write(
                'Please select a smaller number,you cannot choose a number larger than the number of prescriptions in the dataset')
    st.write('If you confirm the number of topics you want to get based on the line chart, please fill in the blank and click "Continue" to get the specific topic matrix')
    num4_con = st.number_input('Enter the number of topics you have confirmed',step=1,format='%d',key=10)
    svd_button_con = st.button('Continue', key=10)
    if svd_button_con:
        df = idf_df.T
        svd = TruncatedSVD(n_components=num4, n_iter=10, random_state=123)
        svd_model = svd.fit(df)
        svd_topic = svd.transform(df)
        columns = ['topic{}'.format(i) for i in range(svd.n_components)]
        pres_svd_topic = pd.DataFrame(svd_topic, columns=columns, index=df.index)
        herb_svd_weight = pd.DataFrame(svd.components_, columns=df.columns,
                                   index=columns)
        herb_svd_weight = herb_svd_weight.T
        #plt.scatter(herb_svd_weight['topic0'], herb_svd_weight['topic1'])
        #plt.scatter(pres_svd_topic['topic0'], pres_svd_topic['topic1'])
        #st.pyplot(plt)
        st.table(pres_svd_topic.head(5))
        st.table(herb_svd_weight.head(5))
        st.success('The topic classification based on LSA is done,you can download this matrix in the "Download" tab')

with tab5:
    st.subheader('Topic classification based on Latent Dirichlet Distribution (LDiA)')
    num5 = st.select_slider(
        'Please select the maximum number of themes you wish to try',
        options=range(1, 100, 1), key=6)
    ldia_button_pressed = st.button('Launch', key=10)
    #st.info('This may take a long time')
    if ldia_button_pressed == True:
        x = []
        y = []
        for i in range(1, num5):
            ldia = LDiA(n_components=i, learning_method='batch', evaluate_every=1, verbose=1, max_iter=50,random_state=123)
            ldia = ldia.fit(herb_dense_dataframe)
            plex = ldia.perplexity(herb_dense_dataframe)
            x.append(i)
            y.append(plex)
        ldia_topic = pd.DataFrame(y, columns=['perplexity'], index=x)
        st.line_chart(ldia_topic)
        with st.expander("See explanation"):
            st.write('Perplexity is an important reference indicator for determining the number of topics in the LDiA model. When perplexity is at its lowest point, we can take the value here as the number of reserved topics')
    st.write('If you confirm the number of topics you want to get based on the line chart, please fill in the blank and click "Continue" to get the specific topic matrix')
    num5_con = st.number_input('Enter the number of topics you have confirmed',step=1,format='%d',key=11)
    ldia_button_con = st.button('Continue', key=11)
    tab5_col1, tab5_col2, tab5_col3 = st.columns(3)
    if ldia_button_con:
        ldia = LDiA(n_components=num5_con,learning_method='batch', evaluate_every=1, verbose=1, max_iter=50,random_state=123)
        ldia = ldia.fit(herb_dense_dataframe)
        columns = ['topic{}'.format(i) for i in range(ldia.n_components)]
        components_herb = pd.DataFrame(ldia.components_.T, index=herb_dense_dataframe.columns, columns=columns)
        components_pres = ldia.transform(herb_dense_dataframe)
        components_pres = pd.DataFrame(components_pres, index=herb_dense_dataframe.index, columns=columns)
        st.table(components_pres.head(5))
        st.table(components_herb.head(5))
        st.success('The topic classification based on LDiA is done,you can download this matrix in the "Download" tab')


with tab6:
    model = gensim.models.Word2Vec(list_vect,sg=0,min_count=1,vector_size = 100,window=avg_len)
    a=pd.DataFrame(model.wv.index_to_key,columns=['name'])
    b=pd.DataFrame(model.wv.vectors,index=a['name'])
    pca = PCA(n_components=2,random_state=123)
    pca = pca.fit(b)
    pca_vectr = pca.transform(b)
    full_common_data = full_common_data.set_index('herb')
    columns = ['topic{}'.format(i) for i in range(pca.n_components)]
    pca_topic = pd.DataFrame(pca_vectr, columns=columns, index=b.index)
    pca_matrix=pca_topic.round(3)
    pca_matrix = pca_matrix.join(full_common_data)
    pca_matrix = pca_matrix.reset_index()
    x=pca_matrix['topic0']
    y=pca_matrix['topic1']
    st.success('The topic classification based on Word2Vec is done,you can download this matrix in the "Download" tab')
    w2v_data= alt.Chart(pca_matrix).mark_circle().encode(
        x='topic0', y='topic1', size='count', color='count', tooltip=['name', 'count'])
    st.altair_chart(w2v_data, use_container_width=True)
    op_w2v=st.radio('What\'s your favorite movie genre?',('Similar herbal search', 'Herbal analogy', 'Compatibility assessment'))
    if op_w2v=='Similar herbal search':
        st.subheader('Similar herbal search')
        st.write('Please enter the name of the herb you want to search')
        search_herb = st.text_input('Enter the herb',key=12)
        search_button = st.button('Search', key=12)
        if search_button:
            feed_herb=model.wv.most_similar(positive=[search_herb],topn=10)
            feed_herb=pd.DataFrame(feed_herb,columns=['herb','similarity'])
            st.table(feed_herb)

    if op_w2v=='Herbal analogy':
        st.subheader('Herbal analogy')
        st.write('Please enter the name of the herb you want to search')
        st.write('If you want to directly compare the similarity of two herbs, please enter the herb name in Herb 1 and Herb 2')
        search_herb_p1 = st.text_input('Herb 1',key=13)
        search_herb_n1 = st.text_input('Herb 2',key=14)
        st.write('If you want to use the method of analogy to explore the law of paired combination of herbs, please also fill in Analogy Item')
        search_herb_p2 = st.text_input('Analogy Item',key=13)

        search_button = st.button('Search', key=13)

        if search_button:
            if len(search_herb_p2)==0:
                feed_herb=model.wv.similarity(search_herb_p1,search_herb_n1)
                st.write('The similarity of {} and {} is {}'.format(search_herb_p1,search_herb_p2,feed_herb))
            if len(search_herb_p2)>0:
                feed_herb=model.wv.most_similar(positive=[search_herb_p2,search_herb_p1],negative=[search_herb_n1],topn=10)
                feed_herb=pd.DataFrame(feed_herb,columns=['herb','vector_similarity'])
                feed_herb=feed_herb.sort_values(by='vector_similarity',ascending=False)
                best_match=feed_herb.iloc[0,0]
                st.write('Imitating the combination rule of {} and {}, {}is a more matching herb with {}'.format(search_herb_p1,search_herb_n1,best_match,search_herb_p2))
                st.write('Alternative herbs that can be paired with {} in the table below'.format(search_herb_p2))
                st.table(feed_herb)

    if op_w2v=='Compatibility assessment':
        st.subheader('Compatibility assessment')
        st.write('Please enter the herb list you want to assessment')
        input_herb = st.text_input('Use "," (English format) to separate the herbs',key=15)
        if st.button('Assessment', key=15):
            if len(input_herb)>0:
                input_herb_list = input_herb.split(',')
                feed_herb=model.wv.doesnt_match(input_herb_list)
                st.write('In this list of herbs, {} has the farthest vector distance from other herbs. Please evaluate whether the use of {} is reasonable in combination with the needs of clinical practice.'.format(feed_herb,feed_herb))

# %%
# 矩阵下载
with tab7:
    # 频次矩阵下载
    if full_common_data.empty == False:
        full_common_data = convert_df(full_common_data)
        st.download_button(
            label="Download full herb frequency data",
            data=full_common_data,
            file_name='full_common_data.xlsx',
            mime='xlsx')
    # 密集矩阵下载
    if herb_dense_dataframe.empty == False:
        herb_dense_dataframe = convert_df(herb_dense_dataframe)
        st.download_button(
            label='Download dense matrix',
            data=herb_dense_dataframe,
            file_name='dense matrix.xlsx')
    # tf-idf矩阵下载
    if idf_df.empty == False:
        tf_idf_matrix = convert_df(idf_df)
        st.download_button(
            label='Download tf_idf_matrix',
            data=tf_idf_matrix,
            file_name='tf_idf_matrix.xlsx')
    # dot product矩阵下载
    if dense_dot.empty == False:
        dense_dot = convert_df(dense_dot)
        st.download_button(
            label='Download dot_product_matrix',
            data=dense_dot,
            file_name='dense dot product.xlsx')

    # cosine similarity矩阵下载
    if cos_dot.empty == False:
        cos_dot = convert_df(cos_dot)
        st.download_button(
            label='Download cosine similarity matrix',
            data=cos_dot,
            file_name='cosine similarity.xlsx')

    # svd矩阵下载

    if svd_button_con == True:
        pres_svd_topic = convert_df(pres_svd_topic)
        herb_svd_weight = convert_df(herb_svd_weight)
        st.download_button(
            label='Download svd topic matrix',
            data=pres_svd_topic,
            file_name='svd topic.xlsx')

        st.download_button(
            label='Download svd weight matrix',
            data=herb_svd_weight,
            file_name='svd herb weight.xlsx')

    # ldia矩阵下载
    if ldia_button_con == True:
        components_herb = convert_df(components_herb)
        components_pres = convert_df(components_pres)
        st.download_button(
            label='Download ldia topic matrix',
            data=components_pres,
            file_name='ldia topic.xlsx')

        st.download_button(
            label='Download ldia herb weight matrix',
            data=components_herb,
            file_name='ldia herb weight.xlsx')
    # pca矩阵下载
    if pca_matrix.empty == False:
        pca_matrix = convert_df(pca_matrix)
        st.download_button(
            label='Download pca topic matrix',
            data=pca_matrix,
            file_name='pca topic.xlsx')
    # word2vec矩阵下载

    # word2vec模型下载




with tab8:
    st.write('Author information:')
    st.write('Name: Zhou Nan')
    st.write('Current situation: PhD student,Universiti Tunku Abdul Rahman(UTAR)')
    st.write('Mail_1:zhounan@1utar.my')
    st.write('Mail_2:zhounan2020@foxmail.com')
    st.write(
        'Due to Streamlit\'s IO capability limitations, this program does not perform well when dealing with larger data sets. If you think this program cannot meet your needs or is always stuck in use, you can contact the author directly, you will get help.')
