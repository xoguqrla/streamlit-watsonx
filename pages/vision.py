import streamlit as st

st.set_page_config(page_title="비전", page_icon="🚀", layout="wide")

st.title("🚀 비전")

st.write("""
Watsonx AI Assistant는 IBM Watsonx 기반으로 동작하는 AI 챗봇입니다. 
이 서비스는 다양한 정보를 제공하며, 특히 지원 제도 및 AI 관련 컨설팅을 제공합니다.

💡 **비전**:
- AI 기술을 활용하여 사용자 맞춤형 정보를 제공
- Watsonx AI를 기반으로 더욱 향상된 챗봇 경험 제공
- 기업 및 개인 사용자를 위한 혁신적인 AI 솔루션 개발

궁금한 점이 있으면 AI 챗봇을 사용해 보세요!
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
            background-color: #28A745;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            text-align: center;
            cursor: pointer;
            font-size: 18px;
            transition: 0.3s;
        }
        .button:hover {
            background-color: #1e7e34;
        }
    </style>
""", unsafe_allow_html=True)

# 버튼 컨테이너
st.markdown("<div class='button-container'>", unsafe_allow_html=True)

if st.button("🤖 AI 챗봇 사용하기"):
    st.switch_page("app")

if st.button("👥 팀 소개 보기"):
    st.switch_page("team")

st.markdown("</div>", unsafe_allow_html=True)
