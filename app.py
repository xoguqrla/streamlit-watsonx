import os
import logging
import streamlit as st
from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods

# 📌 환경 변수 로드
load_dotenv()
API_KEY = os.getenv("API_KEY")
PROJECT_ID = os.getenv("PROJECT_ID")
API_URL = os.getenv("URL")

# 📌 로깅 설정
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(filename=f"{LOG_DIR}/chatbot.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

st.set_page_config(page_title="Watsonx AI 챗봇", layout="wide")

# 🎨 UI 스타일 개선
st.markdown(
    """
    <style>
    .stChatMessage { 
        border-radius: 10px;
        padding: 10px;
    }
    .stTextInput {
        border-radius: 5px;
    }
    .stWarning {
        color: orange;
    }
    .stSuccess {
        color: green;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🏡📱💰 Watsonx AI 지원 정보 챗봇")

# 📌 Watsonx LLM 파라미터 설정
parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY.value,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 500,
    GenParams.STOP_SEQUENCES: ["<|endoftext|>"]
}

# 📌 모델 생성 함수
@st.cache_resource
def create_llm():
    """IBM Watsonx AI 모델 생성"""
    if not API_KEY or not PROJECT_ID or not API_URL:
        st.error("🚨 API 키 또는 프로젝트 ID가 설정되지 않았습니다! `.env` 파일을 확인하세요.")
        st.stop()

    try:
        credentials = Credentials(url=API_URL, api_key=API_KEY)
        model = ModelInference(
            model_id="ibm/granite-3-8b-instruct",
            params=parameters,
            credentials=credentials,
            project_id=PROJECT_ID
        )
        return model
    except Exception as e:
        logging.error(f"❌ 모델 생성 실패: {str(e)}")
        st.error("❌ Watsonx 모델을 생성하는 중 오류가 발생했습니다.")
        return None

# 📌 모델 로드
if "model" not in st.session_state:
    st.session_state.model = create_llm()

# 📌 채팅 메시지 관리
if "messages" not in st.session_state:
    st.session_state.messages = []

# 📌 사용자 입력값 저장 공간
if "user_inputs" not in st.session_state:
    st.session_state.user_inputs = {"성별": None, "나이": None, "카테고리": None, "지역": None}

# 📌 Watsonx AI 호출 함수
def watsonx_ai_api(prompt):
    """Watsonx AI로부터 응답을 가져오는 함수"""
    if not prompt.strip():
        return "⚠️ 입력된 메시지가 없습니다. 내용을 입력해주세요."

    if st.session_state.model is None:
        return "❌ 모델이 준비되지 않았습니다. API 키를 확인하세요."

    try:
        response = st.session_state.model.generate(prompt=prompt)['results'][0]['generated_text'].strip()
        logging.info(f"📨 사용자 입력: {prompt}  | 📩 모델 응답: {response}")
        return response
    except Exception as e:
        logging.error(f"❌ Watsonx 응답 오류: {str(e)}")
        return "🚨 Watsonx AI와 통신 중 오류가 발생했습니다."

# 📌 성별 선택
st.sidebar.header("🔹 성별 선택")
gender = st.sidebar.radio("성별을 선택하세요:", ["남자", "여자"])

# 📌 나이 선택
st.sidebar.header("🔹 나이 선택")
age = st.sidebar.selectbox("나이를 선택하세요:", list(range(10, 70, 5)))

# 📌 카테고리 선택
st.sidebar.header("🔹 카테고리 선택")
category = st.sidebar.selectbox("관심 있는 정보를 선택하세요:", ["주거", "일자리", "금융", "보험", "핸드폰", "지원 제도"])

# 📌 지역 선택
st.sidebar.header("🔹 지역 선택")
region = st.sidebar.text_input("거주 지역을 입력하세요 (예: 서울, 부산, 대구)")

# 📌 선택 완료 버튼
if st.sidebar.button("✅ 설정 완료"):
    if not region:
        st.sidebar.warning("🚨 지역을 입력하세요!")
    else:
        st.session_state.user_inputs["성별"] = gender
        st.session_state.user_inputs["나이"] = age
        st.session_state.user_inputs["카테고리"] = category
        st.session_state.user_inputs["지역"] = region
        st.sidebar.success(f"✅ {region}에 거주하는 {age}세 {gender}님의 {category} 정보를 제공합니다!")

# 📌 기존 채팅 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 📌 사용자 입력 받기
if prompt := st.chat_input("💬 추가 질문이 있나요?"):
    # 📌 Watsonx 프롬프트에 사용자 정보 추가
    user_info = f"사용자 정보: {st.session_state.user_inputs['지역']} 거주 {st.session_state.user_inputs['나이']}세 {st.session_state.user_inputs['성별']}."
    full_prompt = f"{user_info}\n\n{st.session_state.user_inputs['카테고리']} 관련 질문입니다: {prompt}"

    st.session_state.messages.append({"role": "user", "content": full_prompt})

    with st.chat_message("user"):
        st.markdown(full_prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤖 Watsonx AI가 답변을 생성 중..."):
            response = watsonx_ai_api(full_prompt)
            # 📌 응답 최적화 (전화번호, 사이트 포함)
            response += "\n\n📞 관련 기관 문의: 123-456-7890\n🌐 공식 홈페이지: www.example.com"
            st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
