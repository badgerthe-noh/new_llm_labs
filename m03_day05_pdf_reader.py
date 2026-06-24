#===========================================================================================
# AI 튜터 ver 2
# - PDF 파싱 (pdf를 쉽게 읽겠다.) ---> 텍스트 추출 + 페이지 이미지 렌더링
# - 임베딩 모델을 쓰지 않고 로컬 키워드 검색
# - Vision + RAG --> 텍스트 문맥 파악 + 이미지와 함께 LLM에 전달
# - Structured Output(구조화된 출력) --> json 스키마로 형식 고정
#===========================================================================================

#===========================================================================================
# 1. 라이브러리 불러오기
#===========================================================================================
import os
import json
import base64 # 이미지 읽을 때
import tempfile
from pathlib import Path
from typing import List, Dict, Any # 자료형 힌트!
import fitz # PyMuPDF 불러올 때; PDF 파싱 라이브러리
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image

load_dotenv() # .env 파일에서 OPENAI_API_KEY를 불러온다.

#===========================================================================================
# 2. 상수를 설정 
#===========================================================================================
APP_TITLE = '멀티모달 AI 튜텨 Ver.2'
LLM_MODEL = 'gpt-4o-mini' # Vision + Structured Output 모두 지원하는 최소 모델

#===========================================================================================
# 3. 스트림릿 웹페이지 설정
#===========================================================================================
st.set_page_config(page_title=APP_TITLE, page_icon='📖')

#===========================================================================================
# 4. OpenAI 클라이언트 생성
#===========================================================================================
def get_client() -> OpenAI:
    """환경변수에서 API키를 읽어 OpenAI 클라이언트를 반환한다."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 없습니다. .env파일을 확인하세요.")
    return OpenAI(api_key=api_key)

#===========================================================================================
# 5. 업로드 파일 -> 임시 파일로 저장
#===========================================================================================
def save_uploaded_file(upload_file, suffix: str) -> str:
    """
    스트림릿 업로드 위젯을 통한 파일 업로드를 운영체제 임시 폴더에 저장하고 경로를 반환한다.

    PyMuPDF 라이브러리는 파일 경로(str)가 필요하다.

    매개변수:
        suffix: 확장자 (ex. pdf file)
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload_file.getvalue())
        return tmp.name

#===========================================================================================
# 6. PDF 텍스트 추출 ---> [{str:Any}]
#===========================================================================================
def extract_pdf_pages(pdf_path: str) -> List[Dict[str, Any]]: # key는 str, 값은 아무거나
    """
    PyMuPDF로 PDF 전체 페이지의 텍스트를 추출한다.
    키워드 검색은 작동하지 않으며, Vision으로만 분석한다.

    반환 형식:
        [{"page":1, "text": "어쩌구 저쩌구...."}, {"page":2, "text": "dfdafda"}, ...]
    
    """
    doc = fitz.open(pdf_path)
    pages = [] # 결과
    for i, page in enumerate(doc, start=1): # 페이지 번호를 1부터 시작
        text = page.get_text('text').strip() # .strip() : 양쪽 공백 없앤다.
        pages.append({"page":i, "text":text})
    doc.close()
    return pages

#===========================================================================================
# 7. PDF 페이지를 이미지 렌더링
#===========================================================================================
def render_pdf_page_images(pdf_path: str, max_pages: int=3) -> list[str]:
    """
    PDF 앞부분의 페이지를 PNG이미지로 렌더링한다.

    스트림릿 미리보기(잘 불러와졌다고 확인시켜준다.)
    Vision 입력(LLM이 슬라이드 이미지를 시각적으로 분석)

    Matrix(1.5, 1.5) : PDF를 1.5배 해상도로 렌더링

    alpha=False: 투명도 없이 렌더링 (PNG 파일의 크기 감소)

    """
    doc = fitz.open(pdf_path)
    image_paths = [] #파일 경로 리스트 (결과)
    for i in range(min(max_pages, len(doc))):
        page = doc[i]
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5,1.5), alpha=False)
        # 동시 접속자 파일명 충돌 방지 - 프로세스 ID (ex: lecture_page_01_04.png)
        out_path = Path(tempfile.gettempdir()) / f'lecture_page_{os.getpid()}_{i+1}.png'
        pix.save(str(out_path))
        image_paths.append(str(out_path))
    doc.close()
    return image_paths

#===========================================================================================
# 8. 로컬 키워드 검색 (Embedding API 없음)
#===========================================================================================
def keyword_search(pages: List[Dict[str,Any]], query: str, top_k: int=3) -> List[Dict[str, Any]]:
    """
    임베딩 API 없이 단순 키워드 빈도로 관련 페이지를 검색한다.

    동작 방식:
    1. 질문을 공백/구두점으로 분리 -> 2글자 이상 단어만 추출
    2. 각 페이지 텍스트에서 각 단어가 몇 번 등장하는지 합산 -> 점수
    3. 실제 프로덕션에서는 BM25 또는 벡터 검색으로 교체 가능

    """
    # 질문을 단어로 분리(물음표/쉼표 제거, 2글자 이상만 추출)
    query_terms = [
        t.strip().lower()
        for t in query.replace('?',' ').replace(',',' ').split()
        if len(t.strip()) >= 2
    ] # [결과 for if]의 형태

    scored = []
    for p in pages:
        text = p['text'] or ''
        lower = text.lower() # 소문자로 
        # 각 키워드가 페이지에 몇 번 등장하는지 합산
        score = sum(lower.count(term) for term in query_terms)
        # 텍스트는 있지만 키워드가 미매칭 (틀리다!) -> 최소 점수 부여
        if score == 0 and text:
            score = 0.01 # 완전히 배제되지 않도록 한다.
        scored.append({**p, 'score': score}) # 딕셔너리 묶어서 score 리스트에 추가

    # 점수를 내림차순 정렬 -> 상위 top_k 반환
    scored.sort(key=lambda x: x['score'], reverse=True)
    return scored[:top_k]

#===========================================================================================
# 9. 스트림릿 UI
#===========================================================================================
st.title(APP_TITLE)
st.caption('PDF 텍스트 + 슬라이드/캡쳐 이미지를 함께 분석하는 예제입니다!')

# -------- 사이드 바 --------------------
with st.sidebar:
    st.header('수업실습용 설정')
    st.write(f'LLM: {LLM_MODEL}')
    st.info(
        '**비용 제로 검색**\n'
        '- 임베딩 API 미사용\n'
        '- 로컬 키워드 빈도로 관련 페이지 검색\n'
    )
    st.markdown('---')

# -------- 파일 업로더 -------------------
pdf_file = st.file_uploader('PDF 자료 업로드', type = ['pdf'])
image_files = st.file_uploader(
    '추가 이미지 업로드 : 슬라이드 캡쳐, 표, 그림 등',
    type = ['png', 'jpg', 'gif'],
    accept_multiple_files=True # 여러 파일 동시 업로드 허용
)

# --------PDF 업로드 후 메인 UI ------------
if pdf_file:
    pdf_path = save_uploaded_file(pdf_file, '.pdf') # 함수 호출: 임시 경로 저장

    with st.spinner('PDF 텍스트와 미리보기 이미지를 추출하는 중입니다...'):
        pages = extract_pdf_pages(pdf_path) # 함수 호출: 텍스트 추출
        rendered_images = render_pdf_page_images(pdf_path, max_pages=3) # 함수 호출: 이미지 렌더링
        
    st.success(f'PDF 로딩 완료: 총 {len(pages)}페이지')

    #----미리보기 영역-----
    col1, col2 = st.columns([1,1]) #1대1비율
    with col1: #왼쪽
        st.subheader('PDF 앞부분 미리보기')
        for img_path in rendered_images: 
            st.image(img_path, width='stretch')
    with col2: #오른쪽
        st.subheader('추가 이미지 미리보기')
        if image_files:
            for img in image_files[:3]:
                st.image(img, caption=img.name, width='stretch')
        else:
            st.caption('추가 이미지를 올리면 Vision 모델이 함께 참고합니다.')

    #-----vision 용 Data URL 준비 --------
    image_data_urls = [image_to_data_url(p) for p in rendered_images]
    #-----추가 업로드 이미지 -> Data URL
    if image_files:
        image_data_urls.extend(uploaded_image_to_data_url(img) for img in image_files)

    st.markdown('---')
    


#===========================================================================================
# 10. 
#===========================================================================================