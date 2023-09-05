#라이브러리 import
#필요한 경우 install
import streamlit as st
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
from konlpy.tag import Twitter
from konlpy.tag import Okt
from collections import Counter
from wordcloud import WordCloud
import re
import math
from sklearn.preprocessing import normalize

# plotly 시각화 오류시 실행시킬 코드
#import plotly.offline as pyo
#import plotly.graph_objs as go
# 오프라인 모드로 변경하기
#pyo.init_notebook_mode()

#private 페이지를 위한 코드
st.set_page_config(page_title="PRIVATE", page_icon="👤",layout = 'wide')

image = Image.open('images/logo.png')
image2 = Image.open('images/logo2.png')
image3 = Image.open('images/logo3.png')

st.image(image, width=120)
st.sidebar.image(image2, use_column_width=True)
st.sidebar.image(image3, use_column_width=True)

#벡터가 문자열로 인식되는 문제 해결하는 함수
def parse_list(input_str):

    return eval(input_str)


# 화면이 업데이트될 때 마다 변수 할당이 된다면 시간이 오래 걸려서 @st.cache_data 사용(캐싱)
@st.cache_data
def load_client_fv_data():

    client_fv = pd.read_csv("data/client_feature_vector.csv", converters={'feature': parse_list})

    return client_fv

client_fv = load_client_fv_data()


@st.cache_data
def daily_result_load_data():

    daily_result = pd.read_csv("data/daily_result.csv", converters={'fv': parse_list})

    return daily_result

daily_result = daily_result_load_data()


#코사인유사도를 위한 함수 정의
def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))


#하이퍼링크 만드는 함수
def create_link_card(title, url):
    container = st.container()
    container.markdown(
    f'<div class="link-card"><a href="{url}" target="_blank">{title}</a></div>',
    unsafe_allow_html=True,
    )
    return container


st.markdown('''
<h2>Daily Report For <span style="color: #6FA8DC;"> YOU 👤</span></h2>
''', unsafe_allow_html=True)
st.text('')


#value 파라미터로 디폴트 값 지정 가능
#페이지가 열리면 value 값이 자동으로 input_user_name에 할당됨
input_user_name = st.text_input(label="**고객 ID를 먼저 입력하신 뒤 아래 버튼들을 눌러주세요.**", value = "") 


#client_fv에 들어있는 고객ID인지 판단
if input_user_name == '':
    time.sleep(1)
else:
    is_included = int(input_user_name) in client_fv['고객ID'].values
########################################################################################################################################
col1, col2 = st.columns([5.5,4.5])

with col1:
    if st.button("📰 뉴스 추천 받기"):
        if is_included:
            con1 = st.container()
            con1.caption("Result")
            st.info(f'안녕하세요🙂 {str(input_user_name)} 님')
            st.info(f"{str(input_user_name)} 님을 위한 뉴스 추천 서비스입니다.")

            # linktree
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

            # 코사인 유사도 계산
            cosine_similarities = [cos_sim(client_fv[client_fv.고객ID == int(input_user_name)]['feature'].iloc[0], fv) for fv in daily_result.fv]

            # 상위 1개 값의 인덱스 찾기
            top_indices = sorted(range(len(daily_result)), key=lambda i: cosine_similarities[i], reverse=True)[:1]
########################################################################################################################################
            st.subheader("📰 제목")
            st.write(f'**"{daily_result.title[top_indices].iloc[0]}"**')
            st.write('')
########################################################################################################################################     
            st.subheader("👀 시각화")

            #워드클라우드에 사용하기 위해 명사만 추출
            okt = Okt()
            nouns = okt.nouns(daily_result.content_copy[top_indices].iloc[0])

            #필요하다면 불용어 지정 가능
            #stop_words = ""
            #stop_words = set(stop_words.split(' '))
            
            # 단어의 길이가 1개인 것은 제외
            words = [n for n in nouns if len(n) > 1] 

            # 위에서 얻은 words를 처리하여 단어별 빈도수 형태의 딕셔너리 데이터를 구성
            c = Counter(words) 

            # wordcloud
            wordcloud = WordCloud(
                font_path = 'malgun.ttf',
                background_color='white', 
                colormap='Blues' 
            ).generate_from_frequencies(c)

            fig1 = plt.figure()
            plt.imshow(wordcloud,interpolation='bilinear')
            plt.axis('off')
            plt.show()
            st.pyplot(fig1)
            st.write('')
########################################################################################################################################
            st.subheader("📝 요약")

            #하나 이상의 개행문자를 빈 공백으로 대체
            text = re.sub(r"\n+", " ", daily_result.content_copy[top_indices[0]])

            #마침표(.), 물음표(?), 느낌표(!), 개행문자(\n), 공백을 기준으로 문장으로 나눈다
            #sentences = sent_tokenize(daily_result.content_copy[top_indices[0]]) -> sent_tokenize로 문장을 나누니까 잘 안나뉨
            sentences = re.split(r'[.!?]+\s*|\n+', text)

            data = []
            for sentence in sentences:
                if(sentence == "" or len(sentence) == 0):
                    continue
                temp_dict = dict()
                temp_dict['sentence'] = sentence
                #띄어쓰기 단위로 토큰화
                temp_dict['token_list'] = sentence.split() 

                data.append(temp_dict)

            df = pd.DataFrame(data)

            # Text Rank 유사도 수식 구현
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
            

            # Text Rank, Rank 값 구하는 함수 정의
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

            # pagerank를 돌려서 rank matrix 반환
            R = pagerank(weightGraph)

            # 반환된 matrix를 row 별로 sum  
            R = R.sum(axis=1)        

            # 해당 rank 값을 sort, 값이 높은 3개의 문장 index를 반환   
            indexs = R.argsort()[-3:]   

            #rank값이 높은 3개의 문장 출력
            for index in sorted(indexs):
                st.write(df['sentence'][index])
            
            st.write('')
#########################################################################################################################################
            st.subheader("🔑 키워드")
            kw_model = KeyBERT()

            #키워드 3개
            n=3 

            keywords_mmr = kw_model.extract_keywords(daily_result.content[top_indices[0]],
                                                              keyphrase_ngram_range=(1,1),
                                                              use_mmr = False,
                                                              top_n = n,
                                                              diversity = 0.2,
                                                              stop_words = [''])

            st.write('#'+keywords_mmr[0][0],' ', '#'+keywords_mmr[1][0],' ', '#'+keywords_mmr[2][0])
            st.write('')
########################################################################################################################################
            create_link_card(
                "✅ 클릭하시면 뉴스 원문을 볼 수 있습니다.",
                daily_result.url[top_indices].iloc[0])
            st.write('')
        else:
            st.warning(f"{input_user_name}는 올바른 고객ID가 아닙니다. 다시 입력해주세요.")
########################################################################################################################################
with col2:
    if st.button("📒 상세 레포트 보기"):
        if is_included:
            con2 = st.container()
            con2.caption("Result")
            st.info(f"결제데이터를 통해 파악한 {input_user_name} 님의 소비성향입니다. 👇")
            con2.write("")

            # 코사인 유사도 계산
            cosine_similarities = [cos_sim(client_fv[client_fv.고객ID == int(input_user_name)]['feature'].iloc[0], fv) for fv in daily_result.fv]

            # 상위 1개 값의 인덱스 찾기
            top_indices = sorted(range(len(daily_result)), key=lambda i: cosine_similarities[i], reverse=True)[:1]
########################################################################################################################################
            value_a = client_fv[client_fv.고객ID == int(input_user_name)]['feature'].iloc[0]
            data = {'category' : ['여행', '취미', 'IT_전자', '생활', '패션_뷰티', '교육', '의료', '외식'],
                    'value' : value_a }
            df1 = pd.DataFrame(data)

            fig1 = px.pie(df1, names='category', values='value',width=600, height=400)
            fig1.update_layout(
                legend_yanchor="top",
                legend_y=1,
                legend_xanchor="left",
                legend_x=-0.1
            )
            st.plotly_chart(fig1)
########################################################################################################################################            
            st.info(f"{str(input_user_name)} 님에게 추천된 뉴스의 성향입니다. 👇")

            value_b = daily_result.fv[top_indices[0]]
            data = {'category' : ['여행', '취미', 'IT_전자', '생활', '패션_뷰티', '교육', '의료', '외식'],
                    'value' : value_b }
            df2 = pd.DataFrame(data)

            fig2 = px.pie(df2, names='category', values='value',width=600, height=400)
            fig2.update_layout(
                legend_yanchor="top",
                legend_y=1,
                legend_xanchor="left",
                legend_x=-0.1
            )
            st.plotly_chart(fig2)
########################################################################################################################################
            st.info(f"{str(input_user_name)} 님의 소비 성향과 뉴스의 유사도는❓")
            sim = round(cos_sim(value_a, value_b) * 100, 1)

            col1, col2 , col3 = st.columns([4,2,4])

            with col1:
                st.text('')
            with col2:
                st.subheader(f"  {sim}%")
            with col3:
                st.text('')

        # 예외처리
        else:
            st.warning(f"{input_user_name}는 올바른 고객ID가 아닙니다. 다시 입력해주세요.")

