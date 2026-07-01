#=======================================================================
# 이미지 설명 + 해시태그 생성기
# 
# --> Vision : 이미지를 Base64 로 인코딩해서 LLM에 전달
# --> Prompt Engineering : 톤/언어/개수를 프롬프트로 제어
# --> Streamlit : 파일 업로드 + 결과 복사 버튼 UI
#========================================================================
from __future__ import annotations
import base64
import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

APP_TITLE = '이미지 설명 + 해시태그 생성기'
DEFAULT_MODEL = 'gpt-4o-mini' # Vision 지원 모델

# ── 유틸 함수 ──────────────────────────────────────────────────────────────

def get_client() -> OpenAI:
    """OpenAI 클라이언트 반환. API 키 없으면 즉시 오류."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 없습니다. .env 파일을 확인하세요.")
    return OpenAI(api_key=api_key)


def image_to_base64(file_bytes: bytes, mime_type: str) -> str:
    """
    이미지 바이트 → Base64 Data URL 변환

    [수업 포인트]
    OpenAI Vision API는 이미지를 직접 받지 않습니다.
    Base64로 인코딩한 뒤 "data:image/jpeg;base64,..." 형태의
    Data URL로 전달해야 합니다. (영수증 분석기와 동일한 패턴!)
    """
    encoded = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"

def analyze_image(
    data_url: str,
    tone: str,
    language: str,
    tag_count: int,
) -> dict[str, str]:
    client = get_client()

    system_prompt = f'''당신은 SNS 마케팅 카피라이터입니다.
    이미지를 보고 {tone} 스타일로 설명문과 해시태그를 {language}로 작성합니다.
    반드시 아래 형식으로만 출력하세요. 다른 말은 하지 마세요.

    [설명]
    (2~3문장 설명)

    [해시태그]
    (해시태그 {tag_count}개를 #으로 시작하여 스페이스로 구분해서 나열)'''

    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        temperature=0.7,
        max_tokens=500,
        messages=[
            {'role':'system', 'content': system_prompt},
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': '이 이미지를 분석해주세요.'},
                    {'type':'image_url', 'image_url': {'url':data_url}},
                ],
            },
        ],
    )
    raw = response.choices[0].message.content or ''

    # 응답 파싱 - 이미지설명과 해시태그로 텍스트를 분리
    description, hashtags = '', ''
    if '[설명]' in raw and '[해시태그]' in raw:
        parts = raw.split('[해시태그]')
        description = parts[0].replace('[설명]', '').strip()
        hashtags = parts[1].strip()
    else:
        description = raw  # 실패하면 원본 전체를 설명란에 표시

    return {'description':description, 'hashtags':hashtags}




# ── Streamlit UI ───────────────────────────────────────────────────────────

st.set_page_config(page_title=APP_TITLE, page_icon="📸", layout="wide")
st.title(APP_TITLE)
st.caption("이미지를 업로드하면 SNS용 설명문과 해시태그를 자동으로 생성합니다.")

# ── 사이드바: 옵션 설정 ──────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 생성 옵션")

    tone = st.selectbox(
        "설명 톤",
        options=[
            "친근하고 유머러스하게",
            "전문적이고 신뢰감 있게",
            "감성적이고 따뜻하게",
            "짧고 임팩트 있게",
        ],
    )

    language = st.radio(
        "출력 언어",
        options=["한국어", "English"],
        horizontal=True,
    )

    tag_count = st.slider(
        "해시태그 개수",
        min_value=3,
        max_value=15,
        value=8,
    )

    st.markdown("---")
    st.subheader("📚 수업 포인트")
    st.markdown(
        """
- **Vision**: 이미지 → Base64 → LLM
- **Prompt 제어**: 톤·언어·개수를 변수로 주입
- **파싱**: 구분자로 결과 분리
"""
    )

# ── 메인: 파일 업로드 + 결과 ─────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. 이미지 업로드")
    uploaded = st.file_uploader(
        "JPG / PNG / WEBP 이미지를 올려주세요",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded:
        # 업로드된 이미지 미리보기
        st.image(uploaded, caption=uploaded.name, width='stretch')

with col2:
    st.subheader("2. 결과")

    if uploaded is None:
        st.info("왼쪽에서 이미지를 업로드하세요.")
    else:
        if st.button("✨ 설명 + 해시태그 생성", type="primary", width='stretch'):
            # MIME 타입 결정 (확장자 기반)
            ext = uploaded.name.rsplit(".", 1)[-1].lower()
            mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                        "png": "image/png", "webp": "image/webp"}
            mime_type = mime_map.get(ext, "image/jpeg")

            try:
                with st.spinner("이미지 분석 중..."):
                    file_bytes = uploaded.getvalue()
                    data_url = image_to_base64(file_bytes, mime_type)
                    result = analyze_image(data_url, tone, language, tag_count)

                # ── 결과 출력 ─────────────────────────────────────────────
                st.markdown("#### 📝 설명문")
                st.write(result["description"])

                # 클립보드 복사용 text_area
                st.text_area(
                    "복사용",
                    value=result["description"],
                    height=100,
                    label_visibility="collapsed",
                )

                st.markdown("#### #️⃣ 해시태그")
                st.write(result["hashtags"])
                st.text_area(
                    "복사용",
                    value=result["hashtags"],
                    height=80,
                    label_visibility="collapsed",
                )

            except Exception as e:
                st.error("오류가 발생했습니다.")
                st.exception(e)
