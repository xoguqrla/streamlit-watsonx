import streamlit as st

st.set_page_config(page_title="팀 소개", page_icon="👥", layout="wide")

st.title("👥 팀 소개")

st.write("""
Watsonx AI Assistant 프로젝트 팀은 AI 및 머신러닝 전문가들로 구성되어 있습니다.

🛠 **팀 멤버**:
- **김태협** - 프로젝트 매니저 & AI 엔지니어
- **데이터 사이언티스트** - 모델 최적화 및 데이터 분석 담당
- **프론트엔드 개발자** - UI/UX 및 사용자 인터페이스 설계
- **백엔드 개발자** - 서버 및 API 개발

우리는 AI 기술을 활용하여 사용자들에게 더 나은 정보를 제공하는 것을 목표로 합니다.
""")

# 버튼 스타일 적용 (Streamlit Markdown 활용)
st.markdown("""
    <style>
        .button-container {
            display: flex;
            justify-content: center;
            gap: 20px;
        }
        .button {
            background-color: #007BFF;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            text-align: center;
            cursor: pointer;
            font-size: 18px;
            transition: 0.3s;
        }
        .button:hover {
            background-color: #0056b3;
        }
    </style>
""", unsafe_allow_html=True)

# 버튼 컨테이너
st.markdown("<div class='button-container'>", unsafe_allow_html=True)

if st.button("🤖 AI 챗봇 사용하기"):
    st.switch_page("app")

if st.button("🚀 비전 보기"):
    st.switch_page("vision")

st.markdown("</div>", unsafe_allow_html=True)
