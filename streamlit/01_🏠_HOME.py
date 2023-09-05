#라이브러리 import
#필요한 경우 install
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image

#########################################중요###########################################
# 터미널에서 명령어(streamlit run 01_🏠_HOME.py)를 실행 시켜주어야 스트림릿이 작동함
#######################################################################################

image = Image.open('images/logo.png')
image2 = Image.open('images/logo2.png')
image3 = Image.open('images/logo3.png')

#페이지를 위한 코드
#layout = wide : 화면 설정 디폴트값을 와이드로
st.set_page_config(page_title="HOME", page_icon="🏠",layout = 'wide')

#메뉴 탭 하단 사이드바에 이미지 넣기
st.sidebar.image(image2, use_column_width=True)
st.sidebar.image(image3, use_column_width=True)

#최상단에 이미지 넣기
st.image(image, width=120)

st.markdown('''
<h2>Daily News Service by <span style="color: #6FA8DC;"> CAUsumer</span></h2>
''', unsafe_allow_html=True)
st.text('')

st.info("뉴스를 추천받고 싶다면 좌측 메뉴 탭의 **PRIVATE** 을 누르세요!")
st.info("오늘의 뉴스 트렌드가 궁금하면 좌측 메뉴 탭의 **PUBLIC** 을 누르세요!")
st.text('')

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

# 링크
create_link_card(
    "💡소스 코드가 궁금하면 클릭해주세요.  Github으로 이동합니다.",
    "https://github.com/ytg000629/daegu_bigdata",
)

st.write('')
st.write('👇 아래 소개를 참고해주세요')

# st.tab 함수를 통해 하단에 tab 메뉴 생성
tab1, tab2, tab3 = st.tabs(['**Who**❓',  '**PRIVATE**  👤', '**PUBLIC**  👥'])

with tab1:
    st.header('Team  Introduction')
    image = Image.open('images/face.png')
    st.image(image, width=200)
    st.write("안녕하세요, 저희는 **CAUsumer** 입니다.")
    st.write("팀 **CAUsumer**는 중앙대학교 응용통계학과 학생들로 이루어져있습니다.")
    st.text('')
    st.write("빅데이터 분석 공모전 **소비자** 부문으로 참가하며 **소비자**들에게 도움을 줄 수 있는 서비스를 고민했고, 고민 끝에  이 서비스를 기획하게 되었습니다.")
    st.write("**CAUsumer**는 **소비자들**의 결제 데이터와 **뉴스**를 이용한 다양한 서비스를 제공합니다.")

with tab2:
    st.header('Private Service')
    st.write("💡**오늘의 뉴스 현황**을 알려드립니다.")
    st.write('8가지 업종 카테고리 중 어떤 업종의 뉴스가 많이 나왔는지 확인해보세요.')
    st.text('')
    st.write('💡**오늘의 소비자 단어**를 알려드립니다.')
    st.write('소비자들이 알면 좋을 단어를 매일 받아보시고 현명한 소비자가 되어보세요.')
    st.text('')
    st.write('💡**오늘의 뉴스 트렌드**를 알려드립니다.')
    st.write('오늘의 뉴스 키워드 중 어떤 단어가 가장 핫한지 확인해보세요.')
    st.text('')
    st.write('💡여러 업종에 관심이 많은 여러분에게 **업종별 대표 뉴스**를 3개씩 알려드립니다.')

with tab3:
    st.header('Public Service')
    st.write("💡**당신의 고객ID**를 입력하세요. 당신의 **소비 패턴에 적합한 뉴스를 추천**해드립니다. ")
    st.text('')
    st.write('💡긴 뉴스를 선별하여 볼 시간이 없는 바쁜 당신에게')
    st.write('뉴스 제목 및 **요약문**, **핵심 키워드**를 출력해드립니다. ')
    st.text('')
    st.write('💡다방면으로 소비 활동을 펼치는 당신에게.')
    st.write('당신의 소비 패턴 및 추천 뉴스와의 **유사도**를 알려드립니다.')
    st.text('')
    st.write('💡당신을 위한 맞춤형 뉴스를 받아 합리적이고 현명한 소비생활을 즐기세요!')