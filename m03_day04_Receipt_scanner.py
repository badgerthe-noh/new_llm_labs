#=================================================
#1. 라이브러리 불러오기
#=================================================
import os
import json
import base64
from pathlib import Path
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

#=================================================
#2. 환경 설정 (api키 불러오기, 상수 설정(이름정하기), 웹페이지 설정)
#=================================================
load_dotenv()

APP_TITLE = 'Invoice / receipt Analyzer'
MODEL = 'gpt-4o-mini'
OUTPUT_DIR = Path('outputs') # ouputs라는 폴더 경로 지정
OUTPUT_DIR.mkdir(exist_ok=True) # 폴더를 만든다; 이미 있으면 그냥 넘어간다

st.set_page_config(page_title=APP_TITLE, page_icon='🧾', layout='wide')

#=================================================
#3. 함수 정의 - openai api key
#=================================================
def get_client() -> OpenAI:
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError('api 키가 없습니다! .env 파일을 확인하세요.')
    return OpenAI(api_key=api_key)

#=================================================
#4. 함수 정의 - 이미지를 불러와서 컴퓨터가 읽을 수 있도록 인코딩
#             streamlit이 업로드한 파일을 Base64 Data URL로 변환
#         OpenAI Vision API는 이미지를 URL 또는 Base64형식으로 받는다
# 변환 흐름:
# 업로드 파일 -> 바이트(Bytes) -> Base64 인코딩 -> Data URL 문자열
#
# 예시:
#   "data:image/png;base64,ivdsdfaefef...."
#=================================================
def image_to_data_url(uploaded_file) -> str:
    raw = uploaded_file.getvalue() # 원시 데이터 저장
    b64 = base64.b64encode(raw).decode('utf-8')
    mime = uploaded_file.type or 'image/png'
    return f'data:{mime};base64, {b64}'


#=================================================
#5. 함수 정의 - 영수증 분석
#   이미지를 GPT-4o-mini Vision으로 분석하고, JSON 형식에 맞춰
#   구조화된 데이터를 반환한다
#
#=================================================
def analyze_invoice_image(uploaded_file) -> dict:
    client = get_client()   # 함수 호출 - OpenAI 모델 불러오는 것
    data_url = image_to_data_url(uploaded_file) # 함수 호출 - 이미지를 데이터url로 변환
# ── JSON Schema 정의 ───────────────────────────────────
    # LLM이 반드시 이 형식으로만 응답하도록 강제합니다.
    # additionalProperties: False → 스키마에 없는 키는 포함 불가
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "invoice_receipt_result",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "document_type": {
                        "type": "string",
                        "description": "invoice or receipt or unknown"
                    },
                    "vendor":          {"type": "string"},   # 판매처/업체명
                    "invoice_number":  {"type": "string"},   # 인보이스/영수증 번호
                    "invoice_date":    {"type": "string"},   # 발행일
                    "due_date":        {"type": "string"},   # 납부기한 (없으면 빈 문자열)
                    "currency":        {"type": "string"},   # 통화 (KRW, USD 등)
                    "subtotal":        {"type": "number"},   # 공급가액
                    "tax":             {"type": "number"},   # 세금
                    "total":           {"type": "number"},   # 합계금액
                    "payment_method":  {"type": "string"},   # 결제수단
                    "items": {
                        # 항목 배열 (품목 목록)
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "description": {"type": "string"},  # 품목명
                                "quantity":    {"type": "number"},  # 수량
                                "unit_price":  {"type": "number"},  # 단가
                                "amount":      {"type": "number"},  # 소계
                            },
                            "required": ["description", "quantity", "unit_price", "amount"]
                        }
                    },
                    "notes": {"type": "string"}  # 비고 / 기타 정보
                },
                "required": [
                    "document_type", "vendor", "invoice_number", "invoice_date",
                    "due_date", "currency", "subtotal", "tax", "total",
                    "payment_method", "items", "notes"
                ]
            }
        }
    }
    # ---- API 호출: Vision + Structured Output --------------
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                'role':'user',
                'content': [
                    # 텍스트 지시문
                    {
                        'type':'text',
                        'text':'이미지 속 영수증 또는 인보이스를 읽고 정확한 JSON으로 추출해줘.'
                               '보이지 않는 값은 빈 문자열 또는 0으로 해줘.'
                    },
                    # 이미지 (Base64 Data URL)
                    {
                        'type':'image_url',
                        'image_url':{'url':data_url}
                    },

                ],
            }
        ],
        response_format=schema, # Strurctured Output
        temperature=0, # 항상 같은 결과 (정확하게)
        max_tokens=1600,
        
    )
    # 응답 문자열을 파이썬 딕셔너리로 파싱하여 변환
    return json.loads(response.choices[0].message.content)

#=================================================
#6. 스트림릿 UI(User Interface)
#
#=================================================
st.title(APP_TITLE)
st.caption('영수증/인보이스 이미지를 JSON과 CSV로 정리하는 멀티모달 구조화 출력 프로젝트입니다!')

#----------사이드바-----------
with st.sidebar:
    st.header('핵심 개념')
    st.info('Vision + Structured Output + Pandas 저장')
    st.write(f'LLM:{MODEL}')

#---------파일 업로더 -----------
uploaded = st.file_uploader('영수증 또는 인보이스 이미지 업로드', type=['png','jpg'])

if uploaded: # 업로드 성공했다면(데이터가 있다면, 잘 불러와졌다면)
    col1, col2 = st.columns([1,1])
    
    #왼쪽 컬럼 - 업로드 된 이미지 미리보기
    with col1:
        st.subheader('업로드 이미지')
        st.image(uploaded, use_container_width=True)

    # 오른쪽 컬럼 - 분석 버튼 및 결과 표시
    with col2:
        st.subheader('분석')
        if st.button('JSON 추출하기', type='primary'):
            try:
                with st.spinner('이미지를 분석하는 중입니다...'):
                    result = analyze_invoice_image(uploaded) # 분석 함수 호출
                st.success('분석 완료!')
                st.json(result) # 원본 JSON 출력

                # 판다스 데이터프레임으로 변환
                summary_df = pd.DataFrame([{
                    'document_type': result.get('document_type'),
                    'vendor': result.get('vendor'),
                    'invoice_number': result.get('invoice_number'),
                    "invoice_date":   result.get("invoice_date"),
                    "due_date":       result.get("due_date"),
                    "currency":       result.get("currency"),
                    "subtotal":       result.get("subtotal"),
                    "tax":            result.get("tax"),
                    "total":          result.get("total"),
                    "payment_method": result.get("payment_method"),
                }])

                # 품목 정보 (items 배열 - 여러 행)
                items_df = pd.DataFrame(result.get('items', []))

                # 경로 지정
                summary_path = OUTPUT_DIR / 'invoice_summary.csv'
                items_path = OUTPUT_DIR / 'invoice_items.csv'

                # csv 저장
                summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
                items_df.to_csv(items_path, index=False, encoding='utf-8-sig')

                #---표 출력 (스트림릿)-----------
                st.subheader('요약 표')
                st.dataframe(summary_df, use_container_width=True)

                st.subheader('항목 표')
                st.dataframe(items_df, use_container_width=True)

                st.caption(f'CSV 저장 완료: {summary_path}, {items_path}')

            except Exception as e:
                st.error('이미지 분석 중 오류가 발생했습니다!')
                st.exception(e)

else: 
    st.info('샘플 이미지는 석쌤 노션의 sample_invoice.png를 사용하세요.')

