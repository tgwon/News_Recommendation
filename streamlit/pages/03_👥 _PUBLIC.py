#라이브러리 import
#필요한 경우 install
import streamlit as st
from streamlit import components
from keybert import KeyBERT
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import seaborn as sns
import numpy as np
from numpy.linalg import norm
from numpy import nan
from numpy import dot
import ast
from PIL import Image
import pandas as pd
import time
import random
from konlpy.tag import Twitter
from konlpy.tag import Okt
from collections import Counter
from wordcloud import WordCloud
import re
import math
from sklearn.preprocessing import normalize
from konlpy.tag import Komoran

#LDA를 위한 패키지
from gensim import corpora
import gensim
import pyLDAvis
import pyLDAvis.gensim

#plotly 시각화 오류시 실행시킬 코드
#import plotly.offline as pyo
#import plotly.graph_objs as go
# 오프라인 모드로 변경하기
#pyo.init_notebook_mode()

#public 페이지를 위한 코드
st.set_page_config(page_title="PUBLIC", page_icon="👥",layout = 'wide')

image = Image.open('images/logo.png')
image2 = Image.open('images/logo2.png')
image3 = Image.open('images/logo3.png')

st.image(image, width=120)
st.sidebar.image(image2, use_column_width=True)
st.sidebar.image(image3, use_column_width=True)

#리스트를 문자열로 인식하는 문제 해결하는 함수
def parse_list(input_str):

    return eval(input_str)


# 화면이 업데이트될 때 마다 변수 할당이 된다면 시간이 오래 걸려서 @st.cache_data 사용(캐싱)
@st.cache_data
def load_client_fv_data():

    #고객 Feature Vector
    client_fv = pd.read_csv("data/client_feature_vector.csv", converters={'feature': parse_list})

    return client_fv

client_fv = load_client_fv_data()


@st.cache_data
def daily_result_load_data():

    #daily news 전처리 및 모델링 결과
    daily_result = pd.read_csv("data/daily_result.csv", converters={'fv': parse_list})

    return daily_result

daily_result = daily_result_load_data()


@st.cache_data
def lda():
    tokenized_doc = daily_result['content_lda'].apply(lambda x: x.split())
    dictionary = corpora.Dictionary(tokenized_doc)
    corpus = [dictionary.doc2bow(text) for text in tokenized_doc]
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 5,
                                           id2word=dictionary,
                                           passes=15)
    topics = ldamodel.print_topics(num_words=4)

    #pyLDAvis.enable_notebook() -> .py 파일에서는 안됨


    #인터넷 연결 안되면 streamlit 상에서 안보임
    vis =  pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)

    #html으로 시각화
    html_string = pyLDAvis.prepared_data_to_html(vis)

    return html_string

html_string = lda()


st.markdown('''
<h2>Daily Report For <span style="color: #6FA8DC;"> EVERYONE 👥</span></h2>
''', unsafe_allow_html=True)
st.text('')
########################################################################################################################################
#st.columns 함수로 화면 레이아웃을 열로 분리
col1, col2, col3 = st.columns([4,0.5,5.5])

with col1:
    #오늘의 뉴스 현황
    st.write('')
    value_counts = daily_result['category'].value_counts()
    fig2=plt.figure(figsize=(10,5))
    plt.rcParams['font.family'] = 'HYdnkM'
    value_counts.plot(kind='bar', color='skyblue')
    plt.title('오늘의 뉴스 현황', fontsize=25)
    plt.rc('xtick', labelsize=14) 
    plt.xlabel('')
    plt.xticks(rotation=0)
    for i, j in zip([0,1,2,3,4,5,6,7], value_counts.values):
        plt.text(i, j, f'{int(j/len(daily_result)*100)}%', ha='center', va='bottom')
    st.pyplot(fig2)

with col2:
    st.write('')

with col3:
    #오늘의 소비자 단어
    cw = pd.read_csv("data/consumer_word.csv")
    num_indices = len(cw)
    # 랜덤으로 인덱스 3개 추출
    random_indices = random.sample(range(num_indices), 3)
    st.subheader("📚 오늘의 소비자 단어")
    st.write('')
    st.write(f'**"{cw.단어[random_indices[0]]}"**')
    st.write(f'**의미 : {cw.뜻[random_indices[0]]}**')

    # linktree 형식
    st.markdown(
        """
    <style>
    .link-card {
        display: flex;
        flex-direction: column;
        padding: 5px;
        margin-bottom: 10px;
        border: 1px solid #E8DDDA;
        border-radius: 15px;
        background-color: white; 
        box-shadow: 0px 0px 5px #F4EDEC;
    }
    .link-card:hover {
    background-color: #FFF6F3;
    }
    a {
        color: black!important;
        text-decoration: none!important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    #하이퍼링크 만드는 함수
    def create_link_card(title, url):
        container = st.container()
        container.markdown(
            f'<div class="link-card"><a href="{url}" target="_blank">{title}</a></div>',
            unsafe_allow_html=True,
        )
        return container
    
    create_link_card(
        "💡네이버 지식백과로 바로가기",
        cw.url[random_indices[0]],
    )
########################################################################################################################################
st.write('')
st.write('')
st.subheader("👀 뉴스 토픽 알아보기")

#LDA 시각화
components.v1.html(html_string, width=1320, height=880, scrolling=True)
########################################################################################################################################
st.write('')
st.subheader('👇 카테고리별 오늘의 뉴스를 확인하세요')

#8개의 탭에 대해서 동일한 코드 적용
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(['**여행** ✈', 
                                                            '**취미**  🏟',
                                                            '**IT_전자** 💻',
                                                            '**생활**  🏪',
                                                            '**패션_뷰티**  👕',
                                                            '**교육**  📖',
                                                            '**의료**  🩺',
                                                            '**외식**  🍣'])
with tab1:
    col1, col2 = st.columns([5.5,4.5])
    with col1:
        st.subheader('**여행** ✈')
        con1 = st.container()
        con1.caption("💡 제목을 클릭해주세요.")
        #index 찾기
        max_first_index = daily_result['fv'].apply(lambda x: x[0])
        max_indices = max_first_index.nlargest(3).index.tolist()
        if st.button(f'"**{daily_result.title[max_indices[0]]}**"'):
            with col2:
#################################################################################################
                st.subheader("👀 시각화")
                okt = Okt()
                nouns = okt.nouns(daily_result.content_copy[max_indices[0]])
                #stop_words = ""
                #stop_words = set(stop_words.split(' '))
                
                words = [n for n in nouns if len(n) > 1] # 단어의 길이가 1개인 것은 제외

                c = Counter(words) # 위에서 얻은 words를 처리하여 단어별 빈도수 형태의 딕셔너리 데이터를 구성

                wordcloud = WordCloud(
                    font_path = 'malgun.ttf',
                    background_color='white', 
                    colormap='Blues' 
                ).generate_from_frequencies(c)

                fig1 = plt.figure()
                #plt.figure(figsize=(10,10))
                plt.imshow(wordcloud,interpolation='bilinear')
                plt.axis('off')
                plt.show()
                st.pyplot(fig1)               
#################################################################################################
                st.subheader("📝 요약")

                text = re.sub(r"\n+", " ", daily_result.content_copy[max_indices[0]])

                sentences = re.split(r'[.!?]+\s*|\n+', daily_result.content_copy[max_indices[0]])

                data = []
                for sentence in sentences:
                    if(sentence == "" or len(sentence) == 0):
                        continue
                    temp_dict = dict()
                    temp_dict['sentence'] = sentence
                    temp_dict['token_list'] = sentence.split() #가장 기초적인 띄어쓰기 단위로 나누자!

                    data.append(temp_dict)

                df = pd.DataFrame(data)

                
                # sentence similarity = len(intersection) / log(len(set_A)) + log(len(set_B))
                similarity_matrix = []
                for i, row_i in df.iterrows():
                    i_row_vec = []
                    for j, row_j in df.iterrows():
                        if i == j:
                            i_row_vec.append(0.0)
                        else:
                            intersection = len(set(row_i['token_list']) & set(row_j['token_list']))
                            log_i = math.log(len(set(row_i['token_list']))) if len(set(row_i['token_list'])) > 0 else 0
                            log_j = math.log(len(set(row_j['token_list']))) if len(set(row_j['token_list'])) > 0 else 0
                            
                            if log_i == 0 or log_j == 0:
                                similarity = 0
                            else:
                                similarity = intersection / (log_i + log_j)
                            
                            i_row_vec.append(similarity)
                    similarity_matrix.append(i_row_vec)

                weightGraph = np.array(similarity_matrix)
                
                def pagerank(x, df=0.85, max_iter=30):
                    assert 0 < df < 1

                    # initialize
                    A = normalize(x, axis=0, norm='l1')
                    R = np.ones(A.shape[0]).reshape(-1,1)
                    bias = (1 - df) * np.ones(A.shape[0]).reshape(-1,1)
                    # iteration
                    for _ in range(max_iter):
                        R = df * (A * R) + bias

                    return R

                
                R = pagerank(weightGraph) # pagerank를 돌려서 rank matrix 반환
                R = R.sum(axis=1) # 반환된 matrix를 row 별로 sum
                indexs = R.argsort()[-3:] # 해당 rank 값을 sort, 값이 높은 3개의 문장 index를 반환


                #rank값이 높은 3개의 문장 출력
                for index in sorted(indexs):
                    st.write(df['sentence'][index])
                
                st.write('')
#################################################################################################
                st.subheader("🔑 키워드")
                kw_model = KeyBERT()
                n=3 #키워드 3개
                keywords_mmr = kw_model.extract_keywords(daily_result.content[max_indices[0]],keyphrase_ngram_range=(1,1),use_mmr = True,top_n = n, diversity = 0.2)
                st.write('#'+keywords_mmr[0][0],' ', '#'+keywords_mmr[1][0],' ', '#'+keywords_mmr[2][0])
                st.write('')
##################################################################################################
                st.subheader('📰 원문 링크')
                st.write(daily_result.url[max_indices[0]])
##################################################################################################
        if st.button(f'"**{daily_result.title[max_indices[1]]}**"'):
            with col2:
#################################################################################################
                st.subheader("👀 시각화")
                okt = Okt()
                nouns = okt.nouns(daily_result.content_copy[max_indices[1]])
                #stop_words = ""
                #stop_words = set(stop_words.split(' '))

                words = [n for n in nouns if len(n) > 1] # 단어의 길이가 1개인 것은 제외

                c = Counter(words) # 위에서 얻은 words를 처리하여 단어별 빈도수 형태의 딕셔너리 데이터를 구성

                wordcloud = WordCloud(
                    font_path = 'malgun.ttf',
                    background_color='white', 
                    colormap='Blues' 
                ).generate_from_frequencies(c)

                fig1 = plt.figure()
                #plt.figure(figsize=(10,10))
                plt.imshow(wordcloud,interpolation='bilinear')
                plt.axis('off')
                plt.show()
                st.pyplot(fig1)               
#################################################################################################
                st.subheader("📝 요약")

                text = re.sub(r"\n+", " ", daily_result.content_copy[max_indices[1]])

                sentences = re.split(r'[.!?]+\s*|\n+', daily_result.content_copy[max_indices[1]])

                data = []
                for sentence in sentences:
                    if(sentence == "" or len(sentence) == 0):
                        continue
                    temp_dict = dict()
                    temp_dict['sentence'] = sentence
                    temp_dict['token_list'] = sentence.split() #가장 기초적인 띄어쓰기 단위로 나누자!

                    data.append(temp_dict)

                df = pd.DataFrame(data)

                
                # sentence similarity = len(intersection) / log(len(set_A)) + log(len(set_B))
                similarity_matrix = []
                for i, row_i in df.iterrows():
                    i_row_vec = []
                    for j, row_j in df.iterrows():
                        if i == j:
                            i_row_vec.append(0.0)
                        else:
                            intersection = len(set(row_i['token_list']) & set(row_j['token_list']))
                            log_i = math.log(len(set(row_i['token_list']))) if len(set(row_i['token_list'])) > 0 else 0
                            log_j = math.log(len(set(row_j['token_list']))) if len(set(row_j['token_list'])) > 0 else 0
                            
                            if log_i == 0 or log_j == 0:
                                similarity = 0
                            else:
                                similarity = intersection / (log_i + log_j)
                            
                            i_row_vec.append(similarity)
                    similarity_matrix.append(i_row_vec)

                weightGraph = np.array(similarity_matrix)
                
                def pagerank(x, df=0.85, max_iter=30):
                    assert 0 < df < 1

                    # initialize
                    A = normalize(x, axis=0, norm='l1')
                    R = np.ones(A.shape[0]).reshape(-1,1)
                    bias = (1 - df) * np.ones(A.shape[0]).reshape(-1,1)
                    # iteration
                    for _ in range(max_iter):
                        R = df * (A * R) + bias

                    return R

                
                R = pagerank(weightGraph) # pagerank를 돌려서 rank matrix 반환
                R = R.sum(axis=1) # 반환된 matrix를 row 별로 sum
                indexs = R.argsort()[-3:] # 해당 rank 값을 sort, 값이 높은 3개의 문장 index를 반환


                #rank값이 높은 3개의 문장 출력
                for index in sorted(indexs):
                    st.write(df['sentence'][index])
                
                st.write('')
#################################################################################################
                st.subheader("🔑 키워드")
                kw_model = KeyBERT()
                n=3 #키워드 3개
                keywords_mmr = kw_model.extract_keywords(daily_result.content[max_indices[1]],keyphrase_ngram_range=(1,1),use_mmr = True,top_n = n, diversity = 0.2)
                st.write('#'+keywords_mmr[0][0],' ', '#'+keywords_mmr[1][0],' ', '#'+keywords_mmr[2][0])
                st.write('')
##################################################################################################
                st.subheader('📰 원문 링크')
                st.write(daily_result.url[max_indices[1]])
##################################################################################################
        if st.button(f'"**{daily_result.title[max_indices[2]]}**"'):
            with col2:
#################################################################################################
                st.subheader("👀 시각화")
                okt = Okt()
                nouns = okt.nouns(daily_result.content_copy[max_indices[2]])
                #stop_words = ""
                #stop_words = set(stop_words.split(' '))

                words = [n for n in nouns if len(n) > 1] # 단어의 길이가 1개인 것은 제외

                c = Counter(words) # 위에서 얻은 words를 처리하여 단어별 빈도수 형태의 딕셔너리 데이터를 구성

                wordcloud = WordCloud(
                    font_path = 'malgun.ttf',
                    background_color='white', 
                    colormap='Blues' 
                ).generate_from_frequencies(c)

                fig1 = plt.figure()
                #plt.figure(figsize=(10,10))
                plt.imshow(wordcloud,interpolation='bilinear')
                plt.axis('off')
                plt.show()
                st.pyplot(fig1)               
#################################################################################################
                st.subheader("📝 요약")

                text = re.sub(r"\n+", " ", daily_result.content_copy[max_indices[2]])

                sentences = re.split(r'[.!?]+\s*|\n+', daily_result.content_copy[max_indices[1]])

                data = []
                for sentence in sentences:
                    if(sentence == "" or len(sentence) == 0):
                        continue
                    temp_dict = dict()
                    temp_dict['sentence'] = sentence
                    temp_dict['token_list'] = sentence.split() #가장 기초적인 띄어쓰기 단위로 나누자!

                    data.append(temp_dict)

                df = pd.DataFrame(data)

                
                # sentence similarity = len(intersection) / log(len(set_A)) + log(len(set_B))
                similarity_matrix = []
                for i, row_i in df.iterrows():
                    i_row_vec = []
                    for j, row_j in df.iterrows():
                        if i == j:
                            i_row_vec.append(0.0)
                        else:
                            intersection = len(set(row_i['token_list']) & set(row_j['token_list']))
                            log_i = math.log(len(set(row_i['token_list']))) if len(set(row_i['token_list'])) > 0 else 0
                            log_j = math.log(len(set(row_j['token_list']))) if len(set(row_j['token_list'])) > 0 else 0
                            
                            if log_i == 0 or log_j == 0:
                                similarity = 0
                            else:
                                similarity = intersection / (log_i + log_j)
                            
                            i_row_vec.append(similarity)
                    similarity_matrix.append(i_row_vec)

                weightGraph = np.array(similarity_matrix)
                
                def pagerank(x, df=0.85, max_iter=30):
                    assert 0 < df < 1

                    # initialize
                    A = normalize(x, axis=0, norm='l1')
                    R = np.ones(A.shape[0]).reshape(-1,1)
                    bias = (1 - df) * np.ones(A.shape[0]).reshape(-1,1)
                    # iteration
                    for _ in range(max_iter):
                        R = df * (A * R) + bias

                    return R

                
                R = pagerank(weightGraph) # pagerank를 돌려서 rank matrix 반환
                R = R.sum(axis=1) # 반환된 matrix를 row 별로 sum
                indexs = R.argsort()[-3:] # 해당 rank 값을 sort, 값이 높은 3개의 문장 index를 반환


                #rank값이 높은 3개의 문장 출력
                for index in sorted(indexs):
                    st.write(df['sentence'][index])
                
                st.write('')
#################################################################################################
                st.subheader("🔑 키워드")
                kw_model = KeyBERT()
                n=3 #키워드 3개
                keywords_mmr = kw_model.extract_keywords(daily_result.content[max_indices[2]],keyphrase_ngram_range=(1,1),use_mmr = True,top_n = n, diversity = 0.2)
                st.write('#'+keywords_mmr[0][0],' ', '#'+keywords_mmr[1][0],' ', '#'+keywords_mmr[2][0])
                st.write('')
##################################################################################################
                st.subheader('📰 원문 링크')
                st.write(daily_result.url[max_indices[2]])
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
with tab2:
    col1, col2 = st.columns([5.5,4.5])
    with col1:
        st.subheader('**취미**  🏟')
        con1 = st.container()
        con1.caption("💡 제목을 클릭해주세요.")
        #index 찾기
        max_first_index = daily_result['fv'].apply(lambda x: x[1])
        max_indices = max_first_index.nlargest(3).index.tolist()
        if st.button(f'"**{daily_result.title[max_indices[0]]}**"'):
            with col2:
#################################################################################################
                st.subheader("👀 시각화")
                okt = Okt()
                nouns = okt.nouns(daily_result.content_copy[max_indices[0]])
                #stop_words = ""
                #stop_words = set(stop_words.split(' '))

                words = [n for n in nouns if len(n) > 1] # 단어의 길이가 1개인 것은 제외

                c = Counter(words) # 위에서 얻은 words를 처리하여 단어별 빈도수 형태의 딕셔너리 데이터를 구성

                wordcloud = WordCloud(
                    font_path = 'malgun.ttf',
                    background_color='white', 
                    colormap='Blues' 
                ).generate_from_frequencies(c)

                fig1 = plt.figure()
                #plt.figure(figsize=(10,10))
                plt.imshow(wordcloud,interpolation='bilinear')
                plt.axis('off')
                plt.show()
                st.pyplot(fig1)               
#################################################################################################
                st.subheader("📝 요약")

                text = re.sub(r"\n+", " ", daily_result.content_copy[max_indices[0]])

                sentences = re.split(r'[.!?]+\s*|\n+', daily_result.content_copy[max_indices[0]])

                data = []
                for sentence in sentences:
                    if(sentence == "" or len(sentence) == 0):
                        continue
                    temp_dict = dict()
                    temp_dict['sentence'] = sentence
                    temp_dict['token_list'] = sentence.split() #가장 기초적인 띄어쓰기 단위로 나누자!

                    data.append(temp_dict)

                df = pd.DataFrame(data)

                
                # sentence similarity = len(intersection) / log(len(set_A)) + log(len(set_B))
                similarity_matrix = []
                for i, row_i in df.iterrows():
                    i_row_vec = []
                    for j, row_j in df.iterrows():
                        if i == j:
                            i_row_vec.append(0.0)
                        else:
                            intersection = len(set(row_i['token_list']) & set(row_j['token_list']))
                            log_i = math.log(len(set(row_i['token_list']))) if len(set(row_i['token_list'])) > 0 else 0
                            log_j = math.log(len(set(row_j['token_list']))) if len(set(row_j['token_list'])) > 0 else 0
                            
                            if log_i == 0 or log_j == 0:
                                similarity = 0
                            else:
                                similarity = intersection / (log_i + log_j)
                            
                            i_row_vec.append(similarity)
                    similarity_matrix.append(i_row_vec)

                weightGraph = np.array(similarity_matrix)
                
                def pagerank(x, df=0.85, max_iter=30):
                    assert 0 < df < 1

                    # initialize
                    A = normalize(x, axis=0, norm='l1')
                    R = np.ones(A.shape[0]).reshape(-1,1)
                    bias = (1 - df) * np.ones(A.shape[0]).reshape(-1,1)
                    # iteration
                    for _ in range(max_iter):
                        R = df * (A * R) + bias

                    return R

                
                R = pagerank(weightGraph) # pagerank를 돌려서 rank matrix 반환
                R = R.sum(axis=1) # 반환된 matrix를 row 별로 sum
                indexs = R.argsort()[-3:] # 해당 rank 값을 sort, 값이 높은 3개의 문장 index를 반환


                #rank값이 높은 3개의 문장 출력
                for index in sorted(indexs):
                    st.write(df['sentence'][index])
                
                st.write('')
#################################################################################################
                st.subheader("🔑 키워드")
                kw_model = KeyBERT()
                n=3 #키워드 3개
                keywords_mmr = kw_model.extract_keywords(daily_result.content[max_indices[0]],keyphrase_ngram_range=(1,1),use_mmr = True,top_n = n, diversity = 0.2)
                st.write('#'+keywords_mmr[0][0],' ', '#'+keywords_mmr[1][0],' ', '#'+keywords_mmr[2][0])
                st.write('')
##################################################################################################
                st.subheader('📰 원문 링크')
                st.write(daily_result.url[max_indices[0]])
##################################################################################################
        if st.button(f'"**{daily_result.title[max_indices[1]]}**"'):
            with col2:
#################################################################################################
                st.subheader("👀 시각화")
                okt = Okt()
                nouns = okt.nouns(daily_result.content_copy[max_indices[1]])
                #stop_words = ""
                #stop_words = set(stop_words.split(' '))

                words = [n for n in nouns if len(n) > 1] # 단어의 길이가 1개인 것은 제외

                c = Counter(words) # 위에서 얻은 words를 처리하여 단어별 빈도수 형태의 딕셔너리 데이터를 구성

                wordcloud = WordCloud(
                    font_path = 'malgun.ttf',
                    background_color='white', 
                    colormap='Blues' 
                ).generate_from_frequencies(c)

                fig1 = plt.figure()
                #plt.figure(figsize=(10,10))
                plt.imshow(wordcloud,interpolation='bilinear')
                plt.axis('off')
                plt.show()
                st.pyplot(fig1)               
#################################################################################################
                st.subheader("📝 요약")

                text = re.sub(r"\n+", " ", daily_result.content_copy[max_indices[1]])

                sentences = re.split(r'[.!?]+\s*|\n+', daily_result.content_copy[max_indices[1]])

                data = []
                for sentence in sentences:
                    if(sentence == "" or len(sentence) == 0):
                        continue
                    temp_dict = dict()
                    temp_dict['sentence'] = sentence
                    temp_dict['token_list'] = sentence.split() #가장 기초적인 띄어쓰기 단위로 나누자!

                    data.append(temp_dict)

                df = pd.DataFrame(data)

                
                # sentence similarity = len(intersection) / log(len(set_A)) + log(len(set_B))
                similarity_matrix = []
                for i, row_i in df.iterrows():
                    i_row_vec = []
                    for j, row_j in df.iterrows():
                        if i == j:
                            i_row_vec.append(0.0)
                        else:
                            intersection = len(set(row_i['token_list']) & set(row_j['token_list']))
                            log_i = math.log(len(set(row_i['token_list']))) if len(set(row_i['token_list'])) > 0 else 0
                            log_j = math.log(len(set(row_j['token_list']))) if len(set(row_j['token_list'])) > 0 else 0
                            
                            if log_i == 0 or log_j == 0:
                                similarity = 0
                            else:
                                similarity = intersection / (log_i + log_j)
                            
                            i_row_vec.append(similarity)
                    similarity_matrix.append(i_row_vec)

                weightGraph = np.array(similarity_matrix)
                
                def pagerank(x, df=0.85, max_iter=30):
                    assert 0 < df < 1

                    # initialize
                    A = normalize(x, axis=0, norm='l1')
                    R = np.ones(A.shape[0]).reshape(-1,1)
                    bias = (1 - df) * np.ones(A.shape[0]).reshape(-1,1)
                    # iteration
                    for _ in range(max_iter):
                        R = df * (A * R) + bias

                    return R

                
                R = pagerank(weightGraph) # pagerank를 돌려서 rank matrix 반환
                R = R.sum(axis=1) # 반환된 matrix를 row 별로 sum
                indexs = R.argsort()[-3:] # 해당 rank 값을 sort, 값이 높은 3개의 문장 index를 반환


                #rank값이 높은 3개의 문장 출력
                for index in sorted(indexs):
                    st.write(df['sentence'][index])
                
                st.write('')
#################################################################################################
                st.subheader("🔑 키워드")
                kw_model = KeyBERT()
                n=3 #키워드 3개
                keywords_mmr = kw_model.extract_keywords(daily_result.content[max_indices[1]],keyphrase_ngram_range=(1,1),use_mmr = True,top_n = n, diversity = 0.2)
                st.write('#'+keywords_mmr[0][0],' ', '#'+keywords_mmr[1][0],' ', '#'+keywords_mmr[2][0])
                st.write('')
##################################################################################################
                st.subheader('📰 원문 링크')
                st.write(daily_result.url[max_indices[1]])
##################################################################################################
        if st.button(f'"**{daily_result.title[max_indices[2]]}**"'):
            with col2:
#################################################################################################
                st.subheader("👀 시각화")
                okt = Okt()
                nouns = okt.nouns(daily_result.content_copy[max_indices[2]])
                #stop_words = ""
                #stop_words = set(stop_words.split(' '))

                words = [n for n in nouns if len(n) > 1] # 단어의 길이가 1개인 것은 제외

                c = Counter(words) # 위에서 얻은 words를 처리하여 단어별 빈도수 형태의 딕셔너리 데이터를 구성

                wordcloud = WordCloud(
                    font_path = 'malgun.ttf',
                    background_color='white', 
                    colormap='Blues' 
                ).generate_from_frequencies(c)

                fig1 = plt.figure()
                #plt.figure(figsize=(10,10))
                plt.imshow(wordcloud,interpolation='bilinear')
                plt.axis('off')
                plt.show()
                st.pyplot(fig1)               
#################################################################################################
                st.subheader("📝 요약")

                text = re.sub(r"\n+", " ", daily_result.content_copy[max_indices[2]])

                sentences = re.split(r'[.!?]+\s*|\n+', daily_result.content_copy[max_indices[1]])

                data = []
                for sentence in sentences:
                    if(sentence == "" or len(sentence) == 0):
                        continue
                    temp_dict = dict()
                    temp_dict['sentence'] = sentence
                    temp_dict['token_list'] = sentence.split() #가장 기초적인 띄어쓰기 단위로 나누자!

                    data.append(temp_dict)

                df = pd.DataFrame(data)

                
                # sentence similarity = len(intersection) / log(len(set_A)) + log(len(set_B))
                similarity_matrix = []
                for i, row_i in df.iterrows():
                    i_row_vec = []
                    for j, row_j in df.iterrows():
                        if i == j:
                            i_row_vec.append(0.0)
                        else:
                            intersection = len(set(row_i['token_list']) & set(row_j['token_list']))
                            log_i = math.log(len(set(row_i['token_list']))) if len(set(row_i['token_list'])) > 0 else 0
                            log_j = math.log(len(set(row_j['token_list']))) if len(set(row_j['token_list'])) > 0 else 0
                            
                            if log_i == 0 or log_j == 0:
                                similarity = 0
                            else:
                                similarity = intersection / (log_i + log_j)
                            
                            i_row_vec.append(similarity)
                    similarity_matrix.append(i_row_vec)

                weightGraph = np.array(similarity_matrix)
                
                def pagerank(x, df=0.85, max_iter=30):
                    assert 0 < df < 1

                    # initialize
                    A = normalize(x, axis=0, norm='l1')
                    R = np.ones(A.shape[0]).reshape(-1,1)
                    bias = (1 - df) * np.ones(A.shape[0]).reshape(-1,1)
                    # iteration
                    for _ in range(max_iter):
                        R = df * (A * R) + bias

                    return R

                
                R = pagerank(weightGraph) # pagerank를 돌려서 rank matrix 반환
                R = R.sum(axis=1) # 반환된 matrix를 row 별로 sum
                indexs = R.argsort()[-3:] # 해당 rank 값을 sort, 값이 높은 3개의 문장 index를 반환


                #rank값이 높은 3개의 문장 출력
                for index in sorted(indexs):
                    st.write(df['sentence'][index])
                
                st.write('')
#################################################################################################
                st.subheader("🔑 키워드")
                kw_model = KeyBERT()
                n=3 #키워드 3개
                keywords_mmr = kw_model.extract_keywords(daily_result.content[max_indices[2]],keyphrase_ngram_range=(1,1),use_mmr = True,top_n = n, diversity = 0.2)
                st.write('#'+keywords_mmr[0][0],' ', '#'+keywords_mmr[1][0],' ', '#'+keywords_mmr[2][0])
                st.write('')
##################################################################################################
                st.subheader('📰 원문 링크')
                st.write(daily_result.url[max_indices[2]])
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
with tab3:
    col1, col2 = st.columns([5.5,4.5])
    with col1:
        st.subheader('**IT_전자** 💻')
        con1 = st.container()
        con1.caption("💡 제목을 클릭해주세요.")
        #index 찾기
        max_first_index = daily_result['fv'].apply(lambda x: x[2])
        max_indices = max_first_index.nlargest(3).index.tolist()
        if st.button(f'"**{daily_result.title[max_indices[0]]}**"'):
            with col2:
#################################################################################################
                st.subheader("👀 시각화")
                okt = Okt()
                nouns = okt.nouns(daily_result.content_copy[max_indices[0]])
                #stop_words = ""
                #stop_words = set(stop_words.split(' '))

                words = [n for n in nouns if len(n) > 1] # 단어의 길이가 1개인 것은 제외

                c = Counter(words) # 위에서 얻은 words를 처리하여 단어별 빈도수 형태의 딕셔너리 데이터를 구성

                wordcloud = WordCloud(
                    font_path = 'malgun.ttf',
                    background_color='white', 
                    colormap='Blues' 
                ).generate_from_frequencies(c)

                fig1 = plt.figure()
                #plt.figure(figsize=(10,10))
                plt.imshow(wordcloud,interpolation='bilinear')
                plt.axis('off')
                plt.show()
                st.pyplot(fig1)               
#################################################################################################
                st.subheader("📝 요약")

                text = re.sub(r"\n+", " ", daily_result.content_copy[max_indices[0]])

                sentences = re.split(r'[.!?]+\s*|\n+', daily_result.content_copy[max_indices[0]])

                data = []
                for sentence in sentences:
                    if(sentence == "" or len(sentence) == 0):
                        continue
                    temp_dict = dict()
                    temp_dict['sentence'] = sentence
                    temp_dict['token_list'] = sentence.split() #가장 기초적인 띄어쓰기 단위로 나누자!

                    data.append(temp_dict)

                df = pd.DataFrame(data)

                
                # sentence similarity = len(intersection) / log(len(set_A)) + log(len(set_B))
                similarity_matrix = []
                for i, row_i in df.iterrows():
                    i_row_vec = []
                    for j, row_j in df.iterrows():
                        if i == j:
                            i_row_vec.append(0.0)
                        else:
                            intersection = len(set(row_i['token_list']) & set(row_j['token_list']))
                            log_i = math.log(len(set(row_i['token_list']))) if len(set(row_i['token_list'])) > 0 else 0
                            log_j = math.log(len(set(row_j['token_list']))) if len(set(row_j['token_list'])) > 0 else 0
                            
                            if log_i == 0 or log_j == 0:
                                similarity = 0
                            else:
                                similarity = intersection / (log_i + log_j)
                            
                            i_row_vec.append(similarity)
                    similarity_matrix.append(i_row_vec)

                weightGraph = np.array(similarity_matrix)
                
                def pagerank(x, df=0.85, max_iter=30):
                    assert 0 < df < 1

                    # initialize
                    A = normalize(x, axis=0, norm='l1')
                    R = np.ones(A.shape[0]).reshape(-1,1)
                    bias = (1 - df) * np.ones(A.shape[0]).reshape(-1,1)
                    # iteration
                    for _ in range(max_iter):
                        R = df * (A * R) + bias

                    return R

                
                R = pagerank(weightGraph) # pagerank를 돌려서 rank matrix 반환
                R = R.sum(axis=1) # 반환된 matrix를 row 별로 sum
                indexs = R.argsort()[-3:] # 해당 rank 값을 sort, 값이 높은 3개의 문장 index를 반환


                #rank값이 높은 3개의 문장 출력
                for index in sorted(indexs):
                    st.write(df['sentence'][index])
                
                st.write('')
#################################################################################################
                st.subheader("🔑 키워드")
                kw_model = KeyBERT()
                n=3 #키워드 3개
                keywords_mmr = kw_model.extract_keywords(daily_result.content[max_indices[0]],keyphrase_ngram_range=(1,1),use_mmr = True,top_n = n, diversity = 0.2)
                st.write('#'+keywords_mmr[0][0],' ', '#'+keywords_mmr[1][0],' ', '#'+keywords_mmr[2][0])
                st.write('')
##################################################################################################
                st.subheader('📰 원문 링크')
                st.write(daily_result.url[max_indices[0]])
##################################################################################################
        if st.button(f'"**{daily_result.title[max_indices[1]]}**"'):
            with col2:
#################################################################################################
                st.subheader("👀 시각화")
                okt = Okt()
                nouns = okt.nouns(daily_result.content_copy[max_indices[1]])
                #stop_words = ""
                #stop_words = set(stop_words.split(' '))

                words = [n for n in nouns if len(n) > 1] # 단어의 길이가 1개인 것은 제외

                c = Counter(words) # 위에서 얻은 words를 처리하여 단어별 빈도수 형태의 딕셔너리 데이터를 구성

                wordcloud = WordCloud(
                    font_path = 'malgun.ttf',
                    background_color='white', 
                    colormap='Blues' 
                ).generate_from_frequencies(c)

                fig1 = plt.figure()
                #plt.figure(figsize=(10,10))
                plt.imshow(wordcloud,interpolation='bilinear')
                plt.axis('off')
                plt.show()
                st.pyplot(fig1)               
#################################################################################################
                st.subheader("📝 요약")

                text = re.sub(r"\n+", " ", daily_result.content_copy[max_indices[1]])

                sentences = re.split(r'[.!?]+\s*|\n+', daily_result.content_copy[max_indices[1]])

                data = []
                for sentence in sentences:
                    if(sentence == "" or len(sentence) == 0):
                        continue
                    temp_dict = dict()
                    temp_dict['sentence'] = sentence
                    temp_dict['token_list'] = sentence.split() #가장 기초적인 띄어쓰기 단위로 나누자!

                    data.append(temp_dict)

                df = pd.DataFrame(data)

                
                # sentence similarity = len(intersection) / log(len(set_A)) + log(len(set_B))
                similarity_matrix = []
                for i, row_i in df.iterrows():
                    i_row_vec = []
                    for j, row_j in df.iterrows():
                        if i == j:
                            i_row_vec.append(0.0)
                        else:
                            intersection = len(set(row_i['token_list']) & set(row_j['token_list']))
                            log_i = math.log(len(set(row_i['token_list']))) if len(set(row_i['token_list'])) > 0 else 0
                            log_j = math.log(len(set(row_j['token_list']))) if len(set(row_j['token_list'])) > 0 else 0
                            
                            if log_i == 0 or log_j == 0:
                                similarity = 0
                            else:
                                similarity = intersection / (log_i + log_j)
                            
                            i_row_vec.append(similarity)
                    similarity_matrix.append(i_row_vec)

                weightGraph = np.array(similarity_matrix)
                
                def pagerank(x, df=0.85, max_iter=30):
                    assert 0 < df < 1

                    # initialize
                    A = normalize(x, axis=0, norm='l1')
                    R = np.ones(A.shape[0]).reshape(-1,1)
                    bias = (1 - df) * np.ones(A.shape[0]).reshape(-1,1)
                    # iteration
                    for _ in range(max_iter):
                        R = df * (A * R) + bias

                    return R

                
                R = pagerank(weightGraph) # pagerank를 돌려서 rank matrix 반환
                R = R.sum(axis=1) # 반환된 matrix를 row 별로 sum
                indexs = R.argsort()[-3:] # 해당 rank 값을 sort, 값이 높은 3개의 문장 index를 반환


                #rank값이 높은 3개의 문장 출력
                for index in sorted(indexs):
                    st.write(df['sentence'][index])
                
                st.write('')
#################################################################################################
                st.subheader("🔑 키워드")
                kw_model = KeyBERT()
                n=3 #키워드 3개
                keywords_mmr = kw_model.extract_keywords(daily_result.content[max_indices[1]],keyphrase_ngram_range=(1,1),use_mmr = True,top_n = n, diversity = 0.2)
                st.write('#'+keywords_mmr[0][0],' ', '#'+keywords_mmr[1][0],' ', '#'+keywords_mmr[2][0])
                st.write('')
##################################################################################################
                st.subheader('📰 원문 링크')
                st.write(daily_result.url[max_indices[1]])
##################################################################################################
        if st.button(f'"**{daily_result.title[max_indices[2]]}**"'):
            with col2:
#################################################################################################
                st.subheader("👀 시각화")
                okt = Okt()
                nouns = okt.nouns(daily_result.content_copy[max_indices[2]])
                #stop_words = ""
                #stop_words = set(stop_words.split(' '))

                words = [n for n in nouns if len(n) > 1] # 단어의 길이가 1개인 것은 제외

                c = Counter(words) # 위에서 얻은 words를 처리하여 단어별 빈도수 형태의 딕셔너리 데이터를 구성

                wordcloud = WordCloud(
                    font_path = 'malgun.ttf',
                    background_color='white', 
                    colormap='Blues' 
                ).generate_from_frequencies(c)

                fig1 = plt.figure()
                #plt.figure(figsize=(10,10))
                plt.imshow(wordcloud,interpolation='bilinear')
                plt.axis('off')
                plt.show()
                st.pyplot(fig1)               
#################################################################################################
                st.subheader("📝 요약")

                text = re.sub(r"\n+", " ", daily_result.content_copy[max_indices[2]])

                sentences = re.split(r'[.!?]+\s*|\n+', daily_result.content_copy[max_indices[1]])

                data = []
                for sentence in sentences:
                    if(sentence == "" or len(sentence) == 0):
                        continue
                    temp_dict = dict()
                    temp_dict['sentence'] = sentence
                    temp_dict['token_list'] = sentence.split() #가장 기초적인 띄어쓰기 단위로 나누자!

                    data.append(temp_dict)

                df = pd.DataFrame(data)

                
                # sentence similarity = len(intersection) / log(len(set_A)) + log(len(set_B))
                similarity_matrix = []
                for i, row_i in df.iterrows():
                    i_row_vec = []
                    for j, row_j in df.iterrows():
                        if i == j:
                            i_row_vec.append(0.0)
                        else:
                            intersection = len(set(row_i['token_list']) & set(row_j['token_list']))
                            log_i = math.log(len(set(row_i['token_list']))) if len(set(row_i['token_list'])) > 0 else 0
                            log_j = math.log(len(set(row_j['token_list']))) if len(set(row_j['token_list'])) > 0 else 0
                            
                            if log_i == 0 or log_j == 0:
                                similarity = 0
                            else:
                                similarity = intersection / (log_i + log_j)
                            
                            i_row_vec.append(similarity)
                    similarity_matrix.append(i_row_vec)

                weightGraph = np.array(similarity_matrix)
                
                def pagerank(x, df=0.85, max_iter=30):
                    assert 0 < df < 1

                    # initialize
                    A = normalize(x, axis=0, norm='l1')
                    R = np.ones(A.shape[0]).reshape(-1,1)
                    bias = (1 - df) * np.ones(A.shape[0]).reshape(-1,1)
                    # iteration
                    for _ in range(max_iter):
                        R = df * (A * R) + bias

                    return R

                
                R = pagerank(weightGraph) # pagerank를 돌려서 rank matrix 반환
                R = R.sum(axis=1) # 반환된 matrix를 row 별로 sum
                indexs = R.argsort()[-3:] # 해당 rank 값을 sort, 값이 높은 3개의 문장 index를 반환


                #rank값이 높은 3개의 문장 출력
                for index in sorted(indexs):
                    st.write(df['sentence'][index])
                
                st.write('')
#################################################################################################
                st.subheader("🔑 키워드")
                kw_model = KeyBERT()
                n=3 #키워드 3개
                keywords_mmr = kw_model.extract_keywords(daily_result.content[max_indices[2]],keyphrase_ngram_range=(1,1),use_mmr = True,top_n = n, diversity = 0.2)
                st.write('#'+keywords_mmr[0][0],' ', '#'+keywords_mmr[1][0],' ', '#'+keywords_mmr[2][0])
                st.write('')
##################################################################################################
                st.subheader('📰 원문 링크')
                st.write(daily_result.url[max_indices[2]])
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################

with tab4:
    col1, col2 = st.columns([5.5,4.5])
    with col1:
        st.subheader('**생활**  🏪')
        con1 = st.container()
        con1.caption("💡 제목을 클릭해주세요.")
        #index 찾기
        max_first_index = daily_result['fv'].apply(lambda x: x[3])
        max_indices = max_first_index.nlargest(3).index.tolist()
        if st.button(f'"**{daily_result.title[max_indices[0]]}**"'):
            with col2:
#################################################################################################
                st.subheader("👀 시각화")
                okt = Okt()
                nouns = okt.nouns(daily_result.content_copy[max_indices[0]])
                #stop_words = ""
                #stop_words = set(stop_words.split(' '))

                words = [n for n in nouns if len(n) > 1] # 단어의 길이가 1개인 것은 제외

                c = Counter(words) # 위에서 얻은 words를 처리하여 단어별 빈도수 형태의 딕셔너리 데이터를 구성

                wordcloud = WordCloud(
                    font_path = 'malgun.ttf',
                    background_color='white', 
                    colormap='Blues' 
                ).generate_from_frequencies(c)

                fig1 = plt.figure()
                #plt.figure(figsize=(10,10))
                plt.imshow(wordcloud,interpolation='bilinear')
                plt.axis('off')
                plt.show()
                st.pyplot(fig1)               
#################################################################################################
                st.subheader("📝 요약")

                text = re.sub(r"\n+", " ", daily_result.content_copy[max_indices[0]])

                sentences = re.split(r'[.!?]+\s*|\n+', daily_result.content_copy[max_indices[0]])

                data = []
                for sentence in sentences:
                    if(sentence == "" or len(sentence) == 0):
                        continue
                    temp_dict = dict()
                    temp_dict['sentence'] = sentence
                    temp_dict['token_list'] = sentence.split() #가장 기초적인 띄어쓰기 단위로 나누자!

                    data.append(temp_dict)

                df = pd.DataFrame(data)

                
                # sentence similarity = len(intersection) / log(len(set_A)) + log(len(set_B))
                similarity_matrix = []
                for i, row_i in df.iterrows():
                    i_row_vec = []
                    for j, row_j in df.iterrows():
                        if i == j:
                            i_row_vec.append(0.0)
                        else:
                            intersection = len(set(row_i['token_list']) & set(row_j['token_list']))
                            log_i = math.log(len(set(row_i['token_list']))) if len(set(row_i['token_list'])) > 0 else 0
                            log_j = math.log(len(set(row_j['token_list']))) if len(set(row_j['token_list'])) > 0 else 0
                            
                            if log_i == 0 or log_j == 0:
                                similarity = 0
                            else:
                                similarity = intersection / (log_i + log_j)
                            
                            i_row_vec.append(similarity)
                    similarity_matrix.append(i_row_vec)

                weightGraph = np.array(similarity_matrix)
                
                def pagerank(x, df=0.85, max_iter=30):
                    assert 0 < df < 1

                    # initialize
                    A = normalize(x, axis=0, norm='l1')
                    R = np.ones(A.shape[0]).reshape(-1,1)
                    bias = (1 - df) * np.ones(A.shape[0]).reshape(-1,1)
                    # iteration
                    for _ in range(max_iter):
                        R = df * (A * R) + bias

                    return R

                
                R = pagerank(weightGraph) # pagerank를 돌려서 rank matrix 반환
                R = R.sum(axis=1) # 반환된 matrix를 row 별로 sum
                indexs = R.argsort()[-3:] # 해당 rank 값을 sort, 값이 높은 3개의 문장 index를 반환


                #rank값이 높은 3개의 문장 출력
                for index in sorted(indexs):
                    st.write(df['sentence'][index])
                
                st.write('')
#################################################################################################
                st.subheader("🔑 키워드")
                kw_model = KeyBERT()
                n=3 #키워드 3개
                keywords_mmr = kw_model.extract_keywords(daily_result.content[max_indices[0]],keyphrase_ngram_range=(1,1),use_mmr = True,top_n = n, diversity = 0.2)
                st.write('#'+keywords_mmr[0][0],' ', '#'+keywords_mmr[1][0],' ', '#'+keywords_mmr[2][0])
                st.write('')
##################################################################################################
                st.subheader('📰 원문 링크')
                st.write(daily_result.url[max_indices[0]])
##################################################################################################
        if st.button(f'"**{daily_result.title[max_indices[1]]}**"'):
            with col2:
#################################################################################################
                st.subheader("👀 시각화")
                okt = Okt()
                nouns = okt.nouns(daily_result.content_copy[max_indices[1]])
                #stop_words = ""
                #stop_words = set(stop_words.split(' '))

                words = [n for n in nouns if len(n) > 1] # 단어의 길이가 1개인 것은 제외

                c = Counter(words) # 위에서 얻은 words를 처리하여 단어별 빈도수 형태의 딕셔너리 데이터를 구성

                wordcloud = WordCloud(
                    font_path = 'malgun.ttf',
                    background_color='white', 
                    colormap='Blues' 
                ).generate_from_frequencies(c)

                fig1 = plt.figure()
                #plt.figure(figsize=(10,10))
                plt.imshow(wordcloud,interpolation='bilinear')
                plt.axis('off')
                plt.show()
                st.pyplot(fig1)               
#################################################################################################
                st.subheader("📝 요약")

                text = re.sub(r"\n+", " ", daily_result.content_copy[max_indices[1]])

                sentences = re.split(r'[.!?]+\s*|\n+', daily_result.content_copy[max_indices[1]])

                data = []
                for sentence in sentences:
                    if(sentence == "" or len(sentence) == 0):
                        continue
                    temp_dict = dict()
                    temp_dict['sentence'] = sentence
                    temp_dict['token_list'] = sentence.split() #가장 기초적인 띄어쓰기 단위로 나누자!

                    data.append(temp_dict)

                df = pd.DataFrame(data)

                
                # sentence similarity = len(intersection) / log(len(set_A)) + log(len(set_B))
                similarity_matrix = []
                for i, row_i in df.iterrows():
                    i_row_vec = []
                    for j, row_j in df.iterrows():
                        if i == j:
                            i_row_vec.append(0.0)
                        else:
                            intersection = len(set(row_i['token_list']) & set(row_j['token_list']))
                            log_i = math.log(len(set(row_i['token_list']))) if len(set(row_i['token_list'])) > 0 else 0
                            log_j = math.log(len(set(row_j['token_list']))) if len(set(row_j['token_list'])) > 0 else 0
                            
                            if log_i == 0 or log_j == 0:
                                similarity = 0
                            else:
                                similarity = intersection / (log_i + log_j)
                            
                            i_row_vec.append(similarity)
                    similarity_matrix.append(i_row_vec)

                weightGraph = np.array(similarity_matrix)
                
                def pagerank(x, df=0.85, max_iter=30):
                    assert 0 < df < 1

                    # initialize
                    A = normalize(x, axis=0, norm='l1')
                    R = np.ones(A.shape[0]).reshape(-1,1)
                    bias = (1 - df) * np.ones(A.shape[0]).reshape(-1,1)
                    # iteration
                    for _ in range(max_iter):
                        R = df * (A * R) + bias

                    return R

                
                R = pagerank(weightGraph) # pagerank를 돌려서 rank matrix 반환
                R = R.sum(axis=1) # 반환된 matrix를 row 별로 sum
                indexs = R.argsort()[-3:] # 해당 rank 값을 sort, 값이 높은 3개의 문장 index를 반환


                #rank값이 높은 3개의 문장 출력
                for index in sorted(indexs):
                    st.write(df['sentence'][index])
                
                st.write('')
#################################################################################################
                st.subheader("🔑 키워드")
                kw_model = KeyBERT()
                n=3 #키워드 3개
                keywords_mmr = kw_model.extract_keywords(daily_result.content[max_indices[1]],keyphrase_ngram_range=(1,1),use_mmr = True,top_n = n, diversity = 0.2)
                st.write('#'+keywords_mmr[0][0],' ', '#'+keywords_mmr[1][0],' ', '#'+keywords_mmr[2][0])
                st.write('')
##################################################################################################
                st.subheader('📰 원문 링크')
                st.write(daily_result.url[max_indices[1]])
##################################################################################################
        if st.button(f'"**{daily_result.title[max_indices[2]]}**"'):
            with col2:
#################################################################################################
                st.subheader("👀 시각화")
                okt = Okt()
                nouns = okt.nouns(daily_result.content_copy[max_indices[2]])
                #stop_words = ""
                #stop_words = set(stop_words.split(' '))

                words = [n for n in nouns if len(n) > 1] # 단어의 길이가 1개인 것은 제외

                c = Counter(words) # 위에서 얻은 words를 처리하여 단어별 빈도수 형태의 딕셔너리 데이터를 구성

                wordcloud = WordCloud(
                    font_path = 'malgun.ttf',
                    background_color='white', 
                    colormap='Blues' 
                ).generate_from_frequencies(c)

                fig1 = plt.figure()
                #plt.figure(figsize=(10,10))
                plt.imshow(wordcloud,interpolation='bilinear')
                plt.axis('off')
                plt.show()
                st.pyplot(fig1)               
#################################################################################################
                st.subheader("📝 요약")

                text = re.sub(r"\n+", " ", daily_result.content_copy[max_indices[2]])

                sentences = re.split(r'[.!?]+\s*|\n+', daily_result.content_copy[max_indices[1]])

                data = []
                for sentence in sentences:
                    if(sentence == "" or len(sentence) == 0):
                        continue
                    temp_dict = dict()
                    temp_dict['sentence'] = sentence
                    temp_dict['token_list'] = sentence.split() #가장 기초적인 띄어쓰기 단위로 나누자!

                    data.append(temp_dict)

                df = pd.DataFrame(data)

                
                # sentence similarity = len(intersection) / log(len(set_A)) + log(len(set_B))
                similarity_matrix = []
                for i, row_i in df.iterrows():
                    i_row_vec = []
                    for j, row_j in df.iterrows():
                        if i == j:
                            i_row_vec.append(0.0)
                        else:
                            intersection = len(set(row_i['token_list']) & set(row_j['token_list']))
                            log_i = math.log(len(set(row_i['token_list']))) if len(set(row_i['token_list'])) > 0 else 0
                            log_j = math.log(len(set(row_j['token_list']))) if len(set(row_j['token_list'])) > 0 else 0
                            
                            if log_i == 0 or log_j == 0:
                                similarity = 0
                            else:
                                similarity = intersection / (log_i + log_j)
                            
                            i_row_vec.append(similarity)
                    similarity_matrix.append(i_row_vec)

                weightGraph = np.array(similarity_matrix)
                
                def pagerank(x, df=0.85, max_iter=30):
                    assert 0 < df < 1

                    # initialize
                    A = normalize(x, axis=0, norm='l1')
                    R = np.ones(A.shape[0]).reshape(-1,1)
                    bias = (1 - df) * np.ones(A.shape[0]).reshape(-1,1)
                    # iteration
                    for _ in range(max_iter):
                        R = df * (A * R) + bias

                    return R

                
                R = pagerank(weightGraph) # pagerank를 돌려서 rank matrix 반환
                R = R.sum(axis=1) # 반환된 matrix를 row 별로 sum
                indexs = R.argsort()[-3:] # 해당 rank 값을 sort, 값이 높은 3개의 문장 index를 반환


                #rank값이 높은 3개의 문장 출력
                for index in sorted(indexs):
                    st.write(df['sentence'][index])
                
                st.write('')
#################################################################################################
                st.subheader("🔑 키워드")
                kw_model = KeyBERT()
                n=3 #키워드 3개
                keywords_mmr = kw_model.extract_keywords(daily_result.content[max_indices[2]],keyphrase_ngram_range=(1,1),use_mmr = True,top_n = n, diversity = 0.2)
                st.write('#'+keywords_mmr[0][0],' ', '#'+keywords_mmr[1][0],' ', '#'+keywords_mmr[2][0])
                st.write('')
##################################################################################################
                st.subheader('📰 원문 링크')
                st.write(daily_result.url[max_indices[2]])
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################

with tab5:
    col1, col2 = st.columns([5.5,4.5])
    with col1:
        st.subheader('**패션_뷰티**  👕')
        con1 = st.container()
        con1.caption("💡 제목을 클릭해주세요.")
        #index 찾기
        max_first_index = daily_result['fv'].apply(lambda x: x[4])
        max_indices = max_first_index.nlargest(3).index.tolist()
        if st.button(f'"**{daily_result.title[max_indices[0]]}**"'):
            with col2:
#################################################################################################
                st.subheader("👀 시각화")
                okt = Okt()
                nouns = okt.nouns(daily_result.content_copy[max_indices[0]])
                #stop_words = ""
                #stop_words = set(stop_words.split(' '))

                words = [n for n in nouns if len(n) > 1] # 단어의 길이가 1개인 것은 제외

                c = Counter(words) # 위에서 얻은 words를 처리하여 단어별 빈도수 형태의 딕셔너리 데이터를 구성

                wordcloud = WordCloud(
                    font_path = 'malgun.ttf',
                    background_color='white', 
                    colormap='Blues' 
                ).generate_from_frequencies(c)

                fig1 = plt.figure()
                #plt.figure(figsize=(10,10))
                plt.imshow(wordcloud,interpolation='bilinear')
                plt.axis('off')
                plt.show()
                st.pyplot(fig1)               
#################################################################################################
                st.subheader("📝 요약")

                text = re.sub(r"\n+", " ", daily_result.content_copy[max_indices[0]])

                sentences = re.split(r'[.!?]+\s*|\n+', daily_result.content_copy[max_indices[0]])

                data = []
                for sentence in sentences:
                    if(sentence == "" or len(sentence) == 0):
                        continue
                    temp_dict = dict()
                    temp_dict['sentence'] = sentence
                    temp_dict['token_list'] = sentence.split() #가장 기초적인 띄어쓰기 단위로 나누자!

                    data.append(temp_dict)

                df = pd.DataFrame(data)

                
                # sentence similarity = len(intersection) / log(len(set_A)) + log(len(set_B))
                similarity_matrix = []
                for i, row_i in df.iterrows():
                    i_row_vec = []
                    for j, row_j in df.iterrows():
                        if i == j:
                            i_row_vec.append(0.0)
                        else:
                            intersection = len(set(row_i['token_list']) & set(row_j['token_list']))
                            log_i = math.log(len(set(row_i['token_list']))) if len(set(row_i['token_list'])) > 0 else 0
                            log_j = math.log(len(set(row_j['token_list']))) if len(set(row_j['token_list'])) > 0 else 0
                            
                            if log_i == 0 or log_j == 0:
                                similarity = 0
                            else:
                                similarity = intersection / (log_i + log_j)
                            
                            i_row_vec.append(similarity)
                    similarity_matrix.append(i_row_vec)

                weightGraph = np.array(similarity_matrix)
                
                def pagerank(x, df=0.85, max_iter=30):
                    assert 0 < df < 1

                    # initialize
                    A = normalize(x, axis=0, norm='l1')
                    R = np.ones(A.shape[0]).reshape(-1,1)
                    bias = (1 - df) * np.ones(A.shape[0]).reshape(-1,1)
                    # iteration
                    for _ in range(max_iter):
                        R = df * (A * R) + bias

                    return R

                
                R = pagerank(weightGraph) # pagerank를 돌려서 rank matrix 반환
                R = R.sum(axis=1) # 반환된 matrix를 row 별로 sum
                indexs = R.argsort()[-3:] # 해당 rank 값을 sort, 값이 높은 3개의 문장 index를 반환


                #rank값이 높은 3개의 문장 출력
                for index in sorted(indexs):
                    st.write(df['sentence'][index])
                
                st.write('')
#################################################################################################
                st.subheader("🔑 키워드")
                kw_model = KeyBERT()
                n=3 #키워드 3개
                keywords_mmr = kw_model.extract_keywords(daily_result.content[max_indices[0]],keyphrase_ngram_range=(1,1),use_mmr = True,top_n = n, diversity = 0.2)
                st.write('#'+keywords_mmr[0][0],' ', '#'+keywords_mmr[1][0],' ', '#'+keywords_mmr[2][0])
                st.write('')
##################################################################################################
                st.subheader('📰 원문 링크')
                st.write(daily_result.url[max_indices[0]])
##################################################################################################
        if st.button(f'"**{daily_result.title[max_indices[1]]}**"'):
            with col2:
#################################################################################################
                st.subheader("👀 시각화")
                okt = Okt()
                nouns = okt.nouns(daily_result.content_copy[max_indices[1]])
                #stop_words = ""
                #stop_words = set(stop_words.split(' '))

                words = [n for n in nouns if len(n) > 1] # 단어의 길이가 1개인 것은 제외

                c = Counter(words) # 위에서 얻은 words를 처리하여 단어별 빈도수 형태의 딕셔너리 데이터를 구성

                wordcloud = WordCloud(
                    font_path = 'malgun.ttf',
                    background_color='white', 
                    colormap='Blues' 
                ).generate_from_frequencies(c)

                fig1 = plt.figure()
                #plt.figure(figsize=(10,10))
                plt.imshow(wordcloud,interpolation='bilinear')
                plt.axis('off')
                plt.show()
                st.pyplot(fig1)               
#################################################################################################
                st.subheader("📝 요약")

                text = re.sub(r"\n+", " ", daily_result.content_copy[max_indices[1]])

                sentences = re.split(r'[.!?]+\s*|\n+', daily_result.content_copy[max_indices[1]])

                data = []
                for sentence in sentences:
                    if(sentence == "" or len(sentence) == 0):
                        continue
                    temp_dict = dict()
                    temp_dict['sentence'] = sentence
                    temp_dict['token_list'] = sentence.split() #가장 기초적인 띄어쓰기 단위로 나누자!

                    data.append(temp_dict)

                df = pd.DataFrame(data)

                
                # sentence similarity = len(intersection) / log(len(set_A)) + log(len(set_B))
                similarity_matrix = []
                for i, row_i in df.iterrows():
                    i_row_vec = []
                    for j, row_j in df.iterrows():
                        if i == j:
                            i_row_vec.append(0.0)
                        else:
                            intersection = len(set(row_i['token_list']) & set(row_j['token_list']))
                            log_i = math.log(len(set(row_i['token_list']))) if len(set(row_i['token_list'])) > 0 else 0
                            log_j = math.log(len(set(row_j['token_list']))) if len(set(row_j['token_list'])) > 0 else 0
                            
                            if log_i == 0 or log_j == 0:
                                similarity = 0
                            else:
                                similarity = intersection / (log_i + log_j)
                            
                            i_row_vec.append(similarity)
                    similarity_matrix.append(i_row_vec)

                weightGraph = np.array(similarity_matrix)
                
                def pagerank(x, df=0.85, max_iter=30):
                    assert 0 < df < 1

                    # initialize
                    A = normalize(x, axis=0, norm='l1')
                    R = np.ones(A.shape[0]).reshape(-1,1)
                    bias = (1 - df) * np.ones(A.shape[0]).reshape(-1,1)
                    # iteration
                    for _ in range(max_iter):
                        R = df * (A * R) + bias

                    return R

                
                R = pagerank(weightGraph) # pagerank를 돌려서 rank matrix 반환
                R = R.sum(axis=1) # 반환된 matrix를 row 별로 sum
                indexs = R.argsort()[-3:] # 해당 rank 값을 sort, 값이 높은 3개의 문장 index를 반환


                #rank값이 높은 3개의 문장 출력
                for index in sorted(indexs):
                    st.write(df['sentence'][index])
                
                st.write('')
#################################################################################################
                st.subheader("🔑 키워드")
                kw_model = KeyBERT()
                n=3 #키워드 3개
                keywords_mmr = kw_model.extract_keywords(daily_result.content[max_indices[1]],keyphrase_ngram_range=(1,1),use_mmr = True,top_n = n, diversity = 0.2)
                st.write('#'+keywords_mmr[0][0],' ', '#'+keywords_mmr[1][0],' ', '#'+keywords_mmr[2][0])
                st.write('')
##################################################################################################
                st.subheader('📰 원문 링크')
                st.write(daily_result.url[max_indices[1]])
##################################################################################################
        if st.button(f'"**{daily_result.title[max_indices[2]]}**"'):
            with col2:
#################################################################################################
                st.subheader("👀 시각화")
                okt = Okt()
                nouns = okt.nouns(daily_result.content_copy[max_indices[2]])
                #stop_words = ""
                #stop_words = set(stop_words.split(' '))

                words = [n for n in nouns if len(n) > 1] # 단어의 길이가 1개인 것은 제외

                c = Counter(words) # 위에서 얻은 words를 처리하여 단어별 빈도수 형태의 딕셔너리 데이터를 구성

                wordcloud = WordCloud(
                    font_path = 'malgun.ttf',
                    background_color='white', 
                    colormap='Blues' 
                ).generate_from_frequencies(c)

                fig1 = plt.figure()
                #plt.figure(figsize=(10,10))
                plt.imshow(wordcloud,interpolation='bilinear')
                plt.axis('off')
                plt.show()
                st.pyplot(fig1)               
#################################################################################################
                st.subheader("📝 요약")

                text = re.sub(r"\n+", " ", daily_result.content_copy[max_indices[2]])

                sentences = re.split(r'[.!?]+\s*|\n+', daily_result.content_copy[max_indices[1]])

                data = []
                for sentence in sentences:
                    if(sentence == "" or len(sentence) == 0):
                        continue
                    temp_dict = dict()
                    temp_dict['sentence'] = sentence
                    temp_dict['token_list'] = sentence.split() #가장 기초적인 띄어쓰기 단위로 나누자!

                    data.append(temp_dict)

                df = pd.DataFrame(data)

                
                # sentence similarity = len(intersection) / log(len(set_A)) + log(len(set_B))
                similarity_matrix = []
                for i, row_i in df.iterrows():
                    i_row_vec = []
                    for j, row_j in df.iterrows():
                        if i == j:
                            i_row_vec.append(0.0)
                        else:
                            intersection = len(set(row_i['token_list']) & set(row_j['token_list']))
                            log_i = math.log(len(set(row_i['token_list']))) if len(set(row_i['token_list'])) > 0 else 0
                            log_j = math.log(len(set(row_j['token_list']))) if len(set(row_j['token_list'])) > 0 else 0
                            
                            if log_i == 0 or log_j == 0:
                                similarity = 0
                            else:
                                similarity = intersection / (log_i + log_j)
                            
                            i_row_vec.append(similarity)
                    similarity_matrix.append(i_row_vec)

                weightGraph = np.array(similarity_matrix)
                
                def pagerank(x, df=0.85, max_iter=30):
                    assert 0 < df < 1

                    # initialize
                    A = normalize(x, axis=0, norm='l1')
                    R = np.ones(A.shape[0]).reshape(-1,1)
                    bias = (1 - df) * np.ones(A.shape[0]).reshape(-1,1)
                    # iteration
                    for _ in range(max_iter):
                        R = df * (A * R) + bias

                    return R

                
                R = pagerank(weightGraph) # pagerank를 돌려서 rank matrix 반환
                R = R.sum(axis=1) # 반환된 matrix를 row 별로 sum
                indexs = R.argsort()[-3:] # 해당 rank 값을 sort, 값이 높은 3개의 문장 index를 반환


                #rank값이 높은 3개의 문장 출력
                for index in sorted(indexs):
                    st.write(df['sentence'][index])
                
                st.write('')
#################################################################################################
                st.subheader("🔑 키워드")
                kw_model = KeyBERT()
                n=3 #키워드 3개
                keywords_mmr = kw_model.extract_keywords(daily_result.content[max_indices[2]],keyphrase_ngram_range=(1,1),use_mmr = True,top_n = n, diversity = 0.2)
                st.write('#'+keywords_mmr[0][0],' ', '#'+keywords_mmr[1][0],' ', '#'+keywords_mmr[2][0])
                st.write('')
##################################################################################################
                st.subheader('📰 원문 링크')
                st.write(daily_result.url[max_indices[2]])
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################

with tab6:
    col1, col2 = st.columns([5.5,4.5])
    with col1:
        st.subheader('**교육**  📖')
        con1 = st.container()
        con1.caption("💡 제목을 클릭해주세요.")
        #index 찾기
        max_first_index = daily_result['fv'].apply(lambda x: x[5])
        max_indices = max_first_index.nlargest(3).index.tolist()
        if st.button(f'"**{daily_result.title[max_indices[0]]}**"'):
            with col2:
#################################################################################################
                st.subheader("👀 시각화")
                okt = Okt()
                nouns = okt.nouns(daily_result.content_copy[max_indices[0]])
                #stop_words = ""
                #stop_words = set(stop_words.split(' '))

                words = [n for n in nouns if len(n) > 1] # 단어의 길이가 1개인 것은 제외

                c = Counter(words) # 위에서 얻은 words를 처리하여 단어별 빈도수 형태의 딕셔너리 데이터를 구성

                wordcloud = WordCloud(
                    font_path = 'malgun.ttf',
                    background_color='white', 
                    colormap='Blues' 
                ).generate_from_frequencies(c)

                fig1 = plt.figure()
                #plt.figure(figsize=(10,10))
                plt.imshow(wordcloud,interpolation='bilinear')
                plt.axis('off')
                plt.show()
                st.pyplot(fig1)               
#################################################################################################
                st.subheader("📝 요약")

                text = re.sub(r"\n+", " ", daily_result.content_copy[max_indices[0]])

                sentences = re.split(r'[.!?]+\s*|\n+', daily_result.content_copy[max_indices[0]])

                data = []
                for sentence in sentences:
                    if(sentence == "" or len(sentence) == 0):
                        continue
                    temp_dict = dict()
                    temp_dict['sentence'] = sentence
                    temp_dict['token_list'] = sentence.split() #가장 기초적인 띄어쓰기 단위로 나누자!

                    data.append(temp_dict)

                df = pd.DataFrame(data)

                
                # sentence similarity = len(intersection) / log(len(set_A)) + log(len(set_B))
                similarity_matrix = []
                for i, row_i in df.iterrows():
                    i_row_vec = []
                    for j, row_j in df.iterrows():
                        if i == j:
                            i_row_vec.append(0.0)
                        else:
                            intersection = len(set(row_i['token_list']) & set(row_j['token_list']))
                            log_i = math.log(len(set(row_i['token_list']))) if len(set(row_i['token_list'])) > 0 else 0
                            log_j = math.log(len(set(row_j['token_list']))) if len(set(row_j['token_list'])) > 0 else 0
                            
                            if log_i == 0 or log_j == 0:
                                similarity = 0
                            else:
                                similarity = intersection / (log_i + log_j)
                            
                            i_row_vec.append(similarity)
                    similarity_matrix.append(i_row_vec)

                weightGraph = np.array(similarity_matrix)
                
                def pagerank(x, df=0.85, max_iter=30):
                    assert 0 < df < 1

                    # initialize
                    A = normalize(x, axis=0, norm='l1')
                    R = np.ones(A.shape[0]).reshape(-1,1)
                    bias = (1 - df) * np.ones(A.shape[0]).reshape(-1,1)
                    # iteration
                    for _ in range(max_iter):
                        R = df * (A * R) + bias

                    return R

                
                R = pagerank(weightGraph) # pagerank를 돌려서 rank matrix 반환
                R = R.sum(axis=1) # 반환된 matrix를 row 별로 sum
                indexs = R.argsort()[-3:] # 해당 rank 값을 sort, 값이 높은 3개의 문장 index를 반환


                #rank값이 높은 3개의 문장 출력
                for index in sorted(indexs):
                    st.write(df['sentence'][index])
                
                st.write('')
#################################################################################################
                st.subheader("🔑 키워드")
                kw_model = KeyBERT()
                n=3 #키워드 3개
                keywords_mmr = kw_model.extract_keywords(daily_result.content[max_indices[0]],keyphrase_ngram_range=(1,1),use_mmr = True,top_n = n, diversity = 0.2)
                st.write('#'+keywords_mmr[0][0],' ', '#'+keywords_mmr[1][0],' ', '#'+keywords_mmr[2][0])
                st.write('')
##################################################################################################
                st.subheader('📰 원문 링크')
                st.write(daily_result.url[max_indices[0]])
##################################################################################################
        if st.button(f'"**{daily_result.title[max_indices[1]]}**"'):
            with col2:
#################################################################################################
                st.subheader("👀 시각화")
                okt = Okt()
                nouns = okt.nouns(daily_result.content_copy[max_indices[1]])
                #stop_words = ""
                #stop_words = set(stop_words.split(' '))

                words = [n for n in nouns if len(n) > 1] # 단어의 길이가 1개인 것은 제외

                c = Counter(words) # 위에서 얻은 words를 처리하여 단어별 빈도수 형태의 딕셔너리 데이터를 구성

                wordcloud = WordCloud(
                    font_path = 'malgun.ttf',
                    background_color='white', 
                    colormap='Blues' 
                ).generate_from_frequencies(c)

                fig1 = plt.figure()
                #plt.figure(figsize=(10,10))
                plt.imshow(wordcloud,interpolation='bilinear')
                plt.axis('off')
                plt.show()
                st.pyplot(fig1)               
#################################################################################################
                st.subheader("📝 요약")

                text = re.sub(r"\n+", " ", daily_result.content_copy[max_indices[1]])

                sentences = re.split(r'[.!?]+\s*|\n+', daily_result.content_copy[max_indices[1]])

                data = []
                for sentence in sentences:
                    if(sentence == "" or len(sentence) == 0):
                        continue
                    temp_dict = dict()
                    temp_dict['sentence'] = sentence
                    temp_dict['token_list'] = sentence.split() #가장 기초적인 띄어쓰기 단위로 나누자!

                    data.append(temp_dict)

                df = pd.DataFrame(data)

                
                # sentence similarity = len(intersection) / log(len(set_A)) + log(len(set_B))
                similarity_matrix = []
                for i, row_i in df.iterrows():
                    i_row_vec = []
                    for j, row_j in df.iterrows():
                        if i == j:
                            i_row_vec.append(0.0)
                        else:
                            intersection = len(set(row_i['token_list']) & set(row_j['token_list']))
                            log_i = math.log(len(set(row_i['token_list']))) if len(set(row_i['token_list'])) > 0 else 0
                            log_j = math.log(len(set(row_j['token_list']))) if len(set(row_j['token_list'])) > 0 else 0
                            
                            if log_i == 0 or log_j == 0:
                                similarity = 0
                            else:
                                similarity = intersection / (log_i + log_j)
                            
                            i_row_vec.append(similarity)
                    similarity_matrix.append(i_row_vec)

                weightGraph = np.array(similarity_matrix)
                
                def pagerank(x, df=0.85, max_iter=30):
                    assert 0 < df < 1

                    # initialize
                    A = normalize(x, axis=0, norm='l1')
                    R = np.ones(A.shape[0]).reshape(-1,1)
                    bias = (1 - df) * np.ones(A.shape[0]).reshape(-1,1)
                    # iteration
                    for _ in range(max_iter):
                        R = df * (A * R) + bias

                    return R

                
                R = pagerank(weightGraph) # pagerank를 돌려서 rank matrix 반환
                R = R.sum(axis=1) # 반환된 matrix를 row 별로 sum
                indexs = R.argsort()[-3:] # 해당 rank 값을 sort, 값이 높은 3개의 문장 index를 반환


                #rank값이 높은 3개의 문장 출력
                for index in sorted(indexs):
                    st.write(df['sentence'][index])
                
                st.write('')
#################################################################################################
                st.subheader("🔑 키워드")
                kw_model = KeyBERT()
                n=3 #키워드 3개
                keywords_mmr = kw_model.extract_keywords(daily_result.content[max_indices[1]],keyphrase_ngram_range=(1,1),use_mmr = True,top_n = n, diversity = 0.2)
                st.write('#'+keywords_mmr[0][0],' ', '#'+keywords_mmr[1][0],' ', '#'+keywords_mmr[2][0])
                st.write('')
##################################################################################################
                st.subheader('📰 원문 링크')
                st.write(daily_result.url[max_indices[1]])
##################################################################################################
        if st.button(f'"**{daily_result.title[max_indices[2]]}**"'):
            with col2:
#################################################################################################
                st.subheader("👀 시각화")
                okt = Okt()
                nouns = okt.nouns(daily_result.content_copy[max_indices[2]])
                #stop_words = ""
                #stop_words = set(stop_words.split(' '))

                words = [n for n in nouns if len(n) > 1] # 단어의 길이가 1개인 것은 제외

                c = Counter(words) # 위에서 얻은 words를 처리하여 단어별 빈도수 형태의 딕셔너리 데이터를 구성

                wordcloud = WordCloud(
                    font_path = 'malgun.ttf',
                    background_color='white', 
                    colormap='Blues' 
                ).generate_from_frequencies(c)

                fig1 = plt.figure()
                #plt.figure(figsize=(10,10))
                plt.imshow(wordcloud,interpolation='bilinear')
                plt.axis('off')
                plt.show()
                st.pyplot(fig1)               
#################################################################################################
                st.subheader("📝 요약")

                text = re.sub(r"\n+", " ", daily_result.content_copy[max_indices[2]])

                sentences = re.split(r'[.!?]+\s*|\n+', daily_result.content_copy[max_indices[1]])

                data = []
                for sentence in sentences:
                    if(sentence == "" or len(sentence) == 0):
                        continue
                    temp_dict = dict()
                    temp_dict['sentence'] = sentence
                    temp_dict['token_list'] = sentence.split() #가장 기초적인 띄어쓰기 단위로 나누자!

                    data.append(temp_dict)

                df = pd.DataFrame(data)

                
                # sentence similarity = len(intersection) / log(len(set_A)) + log(len(set_B))
                similarity_matrix = []
                for i, row_i in df.iterrows():
                    i_row_vec = []
                    for j, row_j in df.iterrows():
                        if i == j:
                            i_row_vec.append(0.0)
                        else:
                            intersection = len(set(row_i['token_list']) & set(row_j['token_list']))
                            log_i = math.log(len(set(row_i['token_list']))) if len(set(row_i['token_list'])) > 0 else 0
                            log_j = math.log(len(set(row_j['token_list']))) if len(set(row_j['token_list'])) > 0 else 0
                            
                            if log_i == 0 or log_j == 0:
                                similarity = 0
                            else:
                                similarity = intersection / (log_i + log_j)
                            
                            i_row_vec.append(similarity)
                    similarity_matrix.append(i_row_vec)

                weightGraph = np.array(similarity_matrix)
                
                def pagerank(x, df=0.85, max_iter=30):
                    assert 0 < df < 1

                    # initialize
                    A = normalize(x, axis=0, norm='l1')
                    R = np.ones(A.shape[0]).reshape(-1,1)
                    bias = (1 - df) * np.ones(A.shape[0]).reshape(-1,1)
                    # iteration
                    for _ in range(max_iter):
                        R = df * (A * R) + bias

                    return R

                
                R = pagerank(weightGraph) # pagerank를 돌려서 rank matrix 반환
                R = R.sum(axis=1) # 반환된 matrix를 row 별로 sum
                indexs = R.argsort()[-3:] # 해당 rank 값을 sort, 값이 높은 3개의 문장 index를 반환


                #rank값이 높은 3개의 문장 출력
                for index in sorted(indexs):
                    st.write(df['sentence'][index])
                
                st.write('')
#################################################################################################
                st.subheader("🔑 키워드")
                kw_model = KeyBERT()
                n=3 #키워드 3개
                keywords_mmr = kw_model.extract_keywords(daily_result.content[max_indices[2]],keyphrase_ngram_range=(1,1),use_mmr = True,top_n = n, diversity = 0.2)
                st.write('#'+keywords_mmr[0][0],' ', '#'+keywords_mmr[1][0],' ', '#'+keywords_mmr[2][0])
                st.write('')
##################################################################################################
                st.subheader('📰 원문 링크')
                st.write(daily_result.url[max_indices[2]])
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################

with tab7:
    col1, col2 = st.columns([5.5,4.5])
    with col1:
        st.subheader('**의료**  🩺')
        con1 = st.container()
        con1.caption("💡 제목을 클릭해주세요.")
        #index 찾기
        max_first_index = daily_result['fv'].apply(lambda x: x[6])
        max_indices = max_first_index.nlargest(3).index.tolist()
        if st.button(f'"**{daily_result.title[max_indices[0]]}**"'):
            with col2:
#################################################################################################
                st.subheader("👀 시각화")
                okt = Okt()
                nouns = okt.nouns(daily_result.content_copy[max_indices[0]])
                #stop_words = ""
                #stop_words = set(stop_words.split(' '))

                words = [n for n in nouns if len(n) > 1] # 단어의 길이가 1개인 것은 제외

                c = Counter(words) # 위에서 얻은 words를 처리하여 단어별 빈도수 형태의 딕셔너리 데이터를 구성

                wordcloud = WordCloud(
                    font_path = 'malgun.ttf',
                    background_color='white', 
                    colormap='Blues' 
                ).generate_from_frequencies(c)

                fig1 = plt.figure()
                #plt.figure(figsize=(10,10))
                plt.imshow(wordcloud,interpolation='bilinear')
                plt.axis('off')
                plt.show()
                st.pyplot(fig1)               
#################################################################################################
                st.subheader("📝 요약")

                text = re.sub(r"\n+", " ", daily_result.content_copy[max_indices[0]])

                sentences = re.split(r'[.!?]+\s*|\n+', daily_result.content_copy[max_indices[0]])

                data = []
                for sentence in sentences:
                    if(sentence == "" or len(sentence) == 0):
                        continue
                    temp_dict = dict()
                    temp_dict['sentence'] = sentence
                    temp_dict['token_list'] = sentence.split() #가장 기초적인 띄어쓰기 단위로 나누자!

                    data.append(temp_dict)

                df = pd.DataFrame(data)

                
                # sentence similarity = len(intersection) / log(len(set_A)) + log(len(set_B))
                similarity_matrix = []
                for i, row_i in df.iterrows():
                    i_row_vec = []
                    for j, row_j in df.iterrows():
                        if i == j:
                            i_row_vec.append(0.0)
                        else:
                            intersection = len(set(row_i['token_list']) & set(row_j['token_list']))
                            log_i = math.log(len(set(row_i['token_list']))) if len(set(row_i['token_list'])) > 0 else 0
                            log_j = math.log(len(set(row_j['token_list']))) if len(set(row_j['token_list'])) > 0 else 0
                            
                            if log_i == 0 or log_j == 0:
                                similarity = 0
                            else:
                                similarity = intersection / (log_i + log_j)
                            
                            i_row_vec.append(similarity)
                    similarity_matrix.append(i_row_vec)

                weightGraph = np.array(similarity_matrix)
                
                def pagerank(x, df=0.85, max_iter=30):
                    assert 0 < df < 1

                    # initialize
                    A = normalize(x, axis=0, norm='l1')
                    R = np.ones(A.shape[0]).reshape(-1,1)
                    bias = (1 - df) * np.ones(A.shape[0]).reshape(-1,1)
                    # iteration
                    for _ in range(max_iter):
                        R = df * (A * R) + bias

                    return R

                
                R = pagerank(weightGraph) # pagerank를 돌려서 rank matrix 반환
                R = R.sum(axis=1) # 반환된 matrix를 row 별로 sum
                indexs = R.argsort()[-3:] # 해당 rank 값을 sort, 값이 높은 3개의 문장 index를 반환


                #rank값이 높은 3개의 문장 출력
                for index in sorted(indexs):
                    st.write(df['sentence'][index])
                
                st.write('')
#################################################################################################
                st.subheader("🔑 키워드")
                kw_model = KeyBERT()
                n=3 #키워드 3개
                keywords_mmr = kw_model.extract_keywords(daily_result.content[max_indices[0]],keyphrase_ngram_range=(1,1),use_mmr = True,top_n = n, diversity = 0.2)
                st.write('#'+keywords_mmr[0][0],' ', '#'+keywords_mmr[1][0],' ', '#'+keywords_mmr[2][0])
                st.write('')
##################################################################################################
                st.subheader('📰 원문 링크')
                st.write(daily_result.url[max_indices[0]])
##################################################################################################
        if st.button(f'"**{daily_result.title[max_indices[1]]}**"'):
            with col2:
#################################################################################################
                st.subheader("👀 시각화")
                okt = Okt()
                nouns = okt.nouns(daily_result.content_copy[max_indices[1]])
                #stop_words = ""
                #stop_words = set(stop_words.split(' '))

                words = [n for n in nouns if len(n) > 1] # 단어의 길이가 1개인 것은 제외

                c = Counter(words) # 위에서 얻은 words를 처리하여 단어별 빈도수 형태의 딕셔너리 데이터를 구성

                wordcloud = WordCloud(
                    font_path = 'malgun.ttf',
                    background_color='white', 
                    colormap='Blues' 
                ).generate_from_frequencies(c)

                fig1 = plt.figure()
                #plt.figure(figsize=(10,10))
                plt.imshow(wordcloud,interpolation='bilinear')
                plt.axis('off')
                plt.show()
                st.pyplot(fig1)               
#################################################################################################
                st.subheader("📝 요약")

                text = re.sub(r"\n+", " ", daily_result.content_copy[max_indices[1]])

                sentences = re.split(r'[.!?]+\s*|\n+', daily_result.content_copy[max_indices[1]])

                data = []
                for sentence in sentences:
                    if(sentence == "" or len(sentence) == 0):
                        continue
                    temp_dict = dict()
                    temp_dict['sentence'] = sentence
                    temp_dict['token_list'] = sentence.split() #가장 기초적인 띄어쓰기 단위로 나누자!

                    data.append(temp_dict)

                df = pd.DataFrame(data)

                
                # sentence similarity = len(intersection) / log(len(set_A)) + log(len(set_B))
                similarity_matrix = []
                for i, row_i in df.iterrows():
                    i_row_vec = []
                    for j, row_j in df.iterrows():
                        if i == j:
                            i_row_vec.append(0.0)
                        else:
                            intersection = len(set(row_i['token_list']) & set(row_j['token_list']))
                            log_i = math.log(len(set(row_i['token_list']))) if len(set(row_i['token_list'])) > 0 else 0
                            log_j = math.log(len(set(row_j['token_list']))) if len(set(row_j['token_list'])) > 0 else 0
                            
                            if log_i == 0 or log_j == 0:
                                similarity = 0
                            else:
                                similarity = intersection / (log_i + log_j)
                            
                            i_row_vec.append(similarity)
                    similarity_matrix.append(i_row_vec)

                weightGraph = np.array(similarity_matrix)
                
                def pagerank(x, df=0.85, max_iter=30):
                    assert 0 < df < 1

                    # initialize
                    A = normalize(x, axis=0, norm='l1')
                    R = np.ones(A.shape[0]).reshape(-1,1)
                    bias = (1 - df) * np.ones(A.shape[0]).reshape(-1,1)
                    # iteration
                    for _ in range(max_iter):
                        R = df * (A * R) + bias

                    return R

                
                R = pagerank(weightGraph) # pagerank를 돌려서 rank matrix 반환
                R = R.sum(axis=1) # 반환된 matrix를 row 별로 sum
                indexs = R.argsort()[-3:] # 해당 rank 값을 sort, 값이 높은 3개의 문장 index를 반환


                #rank값이 높은 3개의 문장 출력
                for index in sorted(indexs):
                    st.write(df['sentence'][index])
                
                st.write('')
#################################################################################################
                st.subheader("🔑 키워드")
                kw_model = KeyBERT()
                n=3 #키워드 3개
                keywords_mmr = kw_model.extract_keywords(daily_result.content[max_indices[1]],keyphrase_ngram_range=(1,1),use_mmr = True,top_n = n, diversity = 0.2)
                st.write('#'+keywords_mmr[0][0],' ', '#'+keywords_mmr[1][0],' ', '#'+keywords_mmr[2][0])
                st.write('')
##################################################################################################
                st.subheader('📰 원문 링크')
                st.write(daily_result.url[max_indices[1]])
##################################################################################################
        if st.button(f'"**{daily_result.title[max_indices[2]]}**"'):
            with col2:
#################################################################################################
                st.subheader("👀 시각화")
                okt = Okt()
                nouns = okt.nouns(daily_result.content_copy[max_indices[2]])
                #stop_words = ""
                #stop_words = set(stop_words.split(' '))

                words = [n for n in nouns if len(n) > 1] # 단어의 길이가 1개인 것은 제외

                c = Counter(words) # 위에서 얻은 words를 처리하여 단어별 빈도수 형태의 딕셔너리 데이터를 구성

                wordcloud = WordCloud(
                    font_path = 'malgun.ttf',
                    background_color='white', 
                    colormap='Blues' 
                ).generate_from_frequencies(c)

                fig1 = plt.figure()
                #plt.figure(figsize=(10,10))
                plt.imshow(wordcloud,interpolation='bilinear')
                plt.axis('off')
                plt.show()
                st.pyplot(fig1)               
#################################################################################################
                st.subheader("📝 요약")

                text = re.sub(r"\n+", " ", daily_result.content_copy[max_indices[2]])

                sentences = re.split(r'[.!?]+\s*|\n+', daily_result.content_copy[max_indices[1]])

                data = []
                for sentence in sentences:
                    if(sentence == "" or len(sentence) == 0):
                        continue
                    temp_dict = dict()
                    temp_dict['sentence'] = sentence
                    temp_dict['token_list'] = sentence.split() #가장 기초적인 띄어쓰기 단위로 나누자!

                    data.append(temp_dict)

                df = pd.DataFrame(data)

                
                # sentence similarity = len(intersection) / log(len(set_A)) + log(len(set_B))
                similarity_matrix = []
                for i, row_i in df.iterrows():
                    i_row_vec = []
                    for j, row_j in df.iterrows():
                        if i == j:
                            i_row_vec.append(0.0)
                        else:
                            intersection = len(set(row_i['token_list']) & set(row_j['token_list']))
                            log_i = math.log(len(set(row_i['token_list']))) if len(set(row_i['token_list'])) > 0 else 0
                            log_j = math.log(len(set(row_j['token_list']))) if len(set(row_j['token_list'])) > 0 else 0
                            
                            if log_i == 0 or log_j == 0:
                                similarity = 0
                            else:
                                similarity = intersection / (log_i + log_j)
                            
                            i_row_vec.append(similarity)
                    similarity_matrix.append(i_row_vec)

                weightGraph = np.array(similarity_matrix)
                
                def pagerank(x, df=0.85, max_iter=30):
                    assert 0 < df < 1

                    # initialize
                    A = normalize(x, axis=0, norm='l1')
                    R = np.ones(A.shape[0]).reshape(-1,1)
                    bias = (1 - df) * np.ones(A.shape[0]).reshape(-1,1)
                    # iteration
                    for _ in range(max_iter):
                        R = df * (A * R) + bias

                    return R

                
                R = pagerank(weightGraph) # pagerank를 돌려서 rank matrix 반환
                R = R.sum(axis=1) # 반환된 matrix를 row 별로 sum
                indexs = R.argsort()[-3:] # 해당 rank 값을 sort, 값이 높은 3개의 문장 index를 반환


                #rank값이 높은 3개의 문장 출력
                for index in sorted(indexs):
                    st.write(df['sentence'][index])
                
                st.write('')
#################################################################################################
                st.subheader("🔑 키워드")
                kw_model = KeyBERT()
                n=3 #키워드 3개
                keywords_mmr = kw_model.extract_keywords(daily_result.content[max_indices[2]],keyphrase_ngram_range=(1,1),use_mmr = True,top_n = n, diversity = 0.2)
                st.write('#'+keywords_mmr[0][0],' ', '#'+keywords_mmr[1][0],' ', '#'+keywords_mmr[2][0])
                st.write('')
##################################################################################################
                st.subheader('📰 원문 링크')
                st.write(daily_result.url[max_indices[2]])
##################################################################################################
##################################################################################################
##################################################################################################
################################################################################################## 

with tab8:
    col1, col2 = st.columns([5.5,4.5])
    with col1:
        st.subheader('**외식**  🍣')
        con1 = st.container()
        con1.caption("💡 제목을 클릭해주세요.")
        #index 찾기
        max_first_index = daily_result['fv'].apply(lambda x: x[7])
        max_indices = max_first_index.nlargest(3).index.tolist()
        if st.button(f'"**{daily_result.title[max_indices[0]]}**"'):
            with col2:
#################################################################################################
                st.subheader("👀 시각화")
                okt = Okt()
                nouns = okt.nouns(daily_result.content_copy[max_indices[0]])
                #stop_words = ""
                #stop_words = set(stop_words.split(' '))

                words = [n for n in nouns if len(n) > 1] # 단어의 길이가 1개인 것은 제외

                c = Counter(words) # 위에서 얻은 words를 처리하여 단어별 빈도수 형태의 딕셔너리 데이터를 구성

                wordcloud = WordCloud(
                    font_path = 'malgun.ttf',
                    background_color='white', 
                    colormap='Blues' 
                ).generate_from_frequencies(c)

                fig1 = plt.figure()
                #plt.figure(figsize=(10,10))
                plt.imshow(wordcloud,interpolation='bilinear')
                plt.axis('off')
                plt.show()
                st.pyplot(fig1)               
#################################################################################################
                st.subheader("📝 요약")

                text = re.sub(r"\n+", " ", daily_result.content_copy[max_indices[0]])

                sentences = re.split(r'[.!?]+\s*|\n+', daily_result.content_copy[max_indices[0]])

                data = []
                for sentence in sentences:
                    if(sentence == "" or len(sentence) == 0):
                        continue
                    temp_dict = dict()
                    temp_dict['sentence'] = sentence
                    temp_dict['token_list'] = sentence.split() #가장 기초적인 띄어쓰기 단위로 나누자!

                    data.append(temp_dict)

                df = pd.DataFrame(data)

                
                # sentence similarity = len(intersection) / log(len(set_A)) + log(len(set_B))
                similarity_matrix = []
                for i, row_i in df.iterrows():
                    i_row_vec = []
                    for j, row_j in df.iterrows():
                        if i == j:
                            i_row_vec.append(0.0)
                        else:
                            intersection = len(set(row_i['token_list']) & set(row_j['token_list']))
                            log_i = math.log(len(set(row_i['token_list']))) if len(set(row_i['token_list'])) > 0 else 0
                            log_j = math.log(len(set(row_j['token_list']))) if len(set(row_j['token_list'])) > 0 else 0
                            
                            if log_i == 0 or log_j == 0:
                                similarity = 0
                            else:
                                similarity = intersection / (log_i + log_j)
                            
                            i_row_vec.append(similarity)
                    similarity_matrix.append(i_row_vec)

                weightGraph = np.array(similarity_matrix)
                
                def pagerank(x, df=0.85, max_iter=30):
                    assert 0 < df < 1

                    # initialize
                    A = normalize(x, axis=0, norm='l1')
                    R = np.ones(A.shape[0]).reshape(-1,1)
                    bias = (1 - df) * np.ones(A.shape[0]).reshape(-1,1)
                    # iteration
                    for _ in range(max_iter):
                        R = df * (A * R) + bias

                    return R

                
                R = pagerank(weightGraph) # pagerank를 돌려서 rank matrix 반환
                R = R.sum(axis=1) # 반환된 matrix를 row 별로 sum
                indexs = R.argsort()[-3:] # 해당 rank 값을 sort, 값이 높은 3개의 문장 index를 반환


                #rank값이 높은 3개의 문장 출력
                for index in sorted(indexs):
                    st.write(df['sentence'][index])
                
                st.write('')
#################################################################################################
                st.subheader("🔑 키워드")
                kw_model = KeyBERT()
                n=3 #키워드 3개
                keywords_mmr = kw_model.extract_keywords(daily_result.content[max_indices[0]],keyphrase_ngram_range=(1,1),use_mmr = True,top_n = n, diversity = 0.2)
                st.write('#'+keywords_mmr[0][0],' ', '#'+keywords_mmr[1][0],' ', '#'+keywords_mmr[2][0])
                st.write('')
##################################################################################################
                st.subheader('📰 원문 링크')
                st.write(daily_result.url[max_indices[0]])
##################################################################################################
        if st.button(f'"**{daily_result.title[max_indices[1]]}**"'):
            with col2:
#################################################################################################
                st.subheader("👀 시각화")
                okt = Okt()
                nouns = okt.nouns(daily_result.content_copy[max_indices[1]])
                #stop_words = ""
                #stop_words = set(stop_words.split(' '))

                words = [n for n in nouns if len(n) > 1] # 단어의 길이가 1개인 것은 제외

                c = Counter(words) # 위에서 얻은 words를 처리하여 단어별 빈도수 형태의 딕셔너리 데이터를 구성

                wordcloud = WordCloud(
                    font_path = 'malgun.ttf',
                    background_color='white', 
                    colormap='Blues' 
                ).generate_from_frequencies(c)

                fig1 = plt.figure()
                #plt.figure(figsize=(10,10))
                plt.imshow(wordcloud,interpolation='bilinear')
                plt.axis('off')
                plt.show()
                st.pyplot(fig1)               
#################################################################################################
                st.subheader("📝 요약")

                text = re.sub(r"\n+", " ", daily_result.content_copy[max_indices[1]])

                sentences = re.split(r'[.!?]+\s*|\n+', daily_result.content_copy[max_indices[1]])

                data = []
                for sentence in sentences:
                    if(sentence == "" or len(sentence) == 0):
                        continue
                    temp_dict = dict()
                    temp_dict['sentence'] = sentence
                    temp_dict['token_list'] = sentence.split() #가장 기초적인 띄어쓰기 단위로 나누자!

                    data.append(temp_dict)

                df = pd.DataFrame(data)

                
                # sentence similarity = len(intersection) / log(len(set_A)) + log(len(set_B))
                similarity_matrix = []
                for i, row_i in df.iterrows():
                    i_row_vec = []
                    for j, row_j in df.iterrows():
                        if i == j:
                            i_row_vec.append(0.0)
                        else:
                            intersection = len(set(row_i['token_list']) & set(row_j['token_list']))
                            log_i = math.log(len(set(row_i['token_list']))) if len(set(row_i['token_list'])) > 0 else 0
                            log_j = math.log(len(set(row_j['token_list']))) if len(set(row_j['token_list'])) > 0 else 0
                            
                            if log_i == 0 or log_j == 0:
                                similarity = 0
                            else:
                                similarity = intersection / (log_i + log_j)
                            
                            i_row_vec.append(similarity)
                    similarity_matrix.append(i_row_vec)

                weightGraph = np.array(similarity_matrix)
                
                def pagerank(x, df=0.85, max_iter=30):
                    assert 0 < df < 1

                    # initialize
                    A = normalize(x, axis=0, norm='l1')
                    R = np.ones(A.shape[0]).reshape(-1,1)
                    bias = (1 - df) * np.ones(A.shape[0]).reshape(-1,1)
                    # iteration
                    for _ in range(max_iter):
                        R = df * (A * R) + bias

                    return R

                
                R = pagerank(weightGraph) # pagerank를 돌려서 rank matrix 반환
                R = R.sum(axis=1) # 반환된 matrix를 row 별로 sum
                indexs = R.argsort()[-3:] # 해당 rank 값을 sort, 값이 높은 3개의 문장 index를 반환


                #rank값이 높은 3개의 문장 출력
                for index in sorted(indexs):
                    st.write(df['sentence'][index])
                
                st.write('')
#################################################################################################
                st.subheader("🔑 키워드")
                kw_model = KeyBERT()
                n=3 #키워드 3개
                keywords_mmr = kw_model.extract_keywords(daily_result.content[max_indices[1]],keyphrase_ngram_range=(1,1),use_mmr = True,top_n = n, diversity = 0.2)
                st.write('#'+keywords_mmr[0][0],' ', '#'+keywords_mmr[1][0],' ', '#'+keywords_mmr[2][0])
                st.write('')
##################################################################################################
                st.subheader('📰 원문 링크')
                st.write(daily_result.url[max_indices[1]])
##################################################################################################
        if st.button(f'"**{daily_result.title[max_indices[2]]}**"'):
            with col2:
#################################################################################################
                st.subheader("👀 시각화")
                okt = Okt()
                nouns = okt.nouns(daily_result.content_copy[max_indices[2]])
                #stop_words = ""
                #stop_words = set(stop_words.split(' '))

                words = [n for n in nouns if len(n) > 1] # 단어의 길이가 1개인 것은 제외

                c = Counter(words) # 위에서 얻은 words를 처리하여 단어별 빈도수 형태의 딕셔너리 데이터를 구성

                wordcloud = WordCloud(
                    font_path = 'malgun.ttf',
                    background_color='white', 
                    colormap='Blues' 
                ).generate_from_frequencies(c)

                fig1 = plt.figure()
                #plt.figure(figsize=(10,10))
                plt.imshow(wordcloud,interpolation='bilinear')
                plt.axis('off')
                plt.show()
                st.pyplot(fig1)               
#################################################################################################
                st.subheader("📝 요약")

                text = re.sub(r"\n+", " ", daily_result.content_copy[max_indices[2]])

                sentences = re.split(r'[.!?]+\s*|\n+', daily_result.content_copy[max_indices[1]])

                data = []
                for sentence in sentences:
                    if(sentence == "" or len(sentence) == 0):
                        continue
                    temp_dict = dict()
                    temp_dict['sentence'] = sentence
                    temp_dict['token_list'] = sentence.split() #가장 기초적인 띄어쓰기 단위로 나누자!

                    data.append(temp_dict)

                df = pd.DataFrame(data)

                
                # sentence similarity = len(intersection) / log(len(set_A)) + log(len(set_B))
                similarity_matrix = []
                for i, row_i in df.iterrows():
                    i_row_vec = []
                    for j, row_j in df.iterrows():
                        if i == j:
                            i_row_vec.append(0.0)
                        else:
                            intersection = len(set(row_i['token_list']) & set(row_j['token_list']))
                            log_i = math.log(len(set(row_i['token_list']))) if len(set(row_i['token_list'])) > 0 else 0
                            log_j = math.log(len(set(row_j['token_list']))) if len(set(row_j['token_list'])) > 0 else 0
                            
                            if log_i == 0 or log_j == 0:
                                similarity = 0
                            else:
                                similarity = intersection / (log_i + log_j)
                            
                            i_row_vec.append(similarity)
                    similarity_matrix.append(i_row_vec)

                weightGraph = np.array(similarity_matrix)
                
                def pagerank(x, df=0.85, max_iter=30):
                    assert 0 < df < 1

                    # initialize
                    A = normalize(x, axis=0, norm='l1')
                    R = np.ones(A.shape[0]).reshape(-1,1)
                    bias = (1 - df) * np.ones(A.shape[0]).reshape(-1,1)
                    # iteration
                    for _ in range(max_iter):
                        R = df * (A * R) + bias

                    return R

                
                R = pagerank(weightGraph) # pagerank를 돌려서 rank matrix 반환
                R = R.sum(axis=1) # 반환된 matrix를 row 별로 sum
                indexs = R.argsort()[-3:] # 해당 rank 값을 sort, 값이 높은 3개의 문장 index를 반환


                #rank값이 높은 3개의 문장 출력
                for index in sorted(indexs):
                    st.write(df['sentence'][index])
                
                st.write('')
#################################################################################################
                st.subheader("🔑 키워드")
                kw_model = KeyBERT()
                n=3 #키워드 3개
                keywords_mmr = kw_model.extract_keywords(daily_result.content[max_indices[2]],keyphrase_ngram_range=(1,1),use_mmr = True,top_n = n, diversity = 0.2)
                st.write('#'+keywords_mmr[0][0],' ', '#'+keywords_mmr[1][0],' ', '#'+keywords_mmr[2][0])
                st.write('')
##################################################################################################
                st.subheader('📰 원문 링크')
                st.write(daily_result.url[max_indices[2]])
##################################################################################################
##################################################################################################
##################################################################################################
################################################################################################## 
