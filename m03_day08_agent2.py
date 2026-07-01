# =======================================================================
# AI Research Agent
#
# --> 실시간 웹검색 API 없이도 동작하는 워크플로형 Agent 입문 예제
#
# Agent 단계 (순서대로 실행)
# 1. 조사 계획 생성 -> 2. 자료 요약 -> 3. 최종 보고서 -> 4. 발표 스크립트 ->
# 5. 예상 QnA
#
# 복잡한 랭체인, 랭그래프 없이도 Agent식 사고방식을 구현해보자!
# 각 단계(노드)가 독립된 함수로 분리되어 있어 유지보수가 쉽다.
# =========================================================================

from __future__ import annotations
import json 
import os
from datetime import datetime
from typing import Any # 어떤 타입이든 가능 (타입 힌트 중 Any)
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# .env 파일 로드
load_dotenv()

APP_TITLE = 'AI Research Agent'
DEFAULT_MODEL = 'gpt-4o-mini'

# --------------------------------------------------------
# 1. 기본 유틸 함수
# --------------------------------------------------------
def get_client() -> OpenAI | None: # 리턴되는 자료형이 OpenAI 객체 또는 None
    """api를 읽어와서 OpenAI 객체 생성"""
    api_key = os.getenv('OPENAI_API_KEY') # 환경변수에서 키 읽기
    if not api_key:
        return None
    return OpenAI(api_key=api_key) # 클라이언트(객체) 생성 후 반환

def call_llm(system_prompt: str, user_prompt: str, temperature: float=0.2) -> str:
    """OpenAI LLM 호출하는 함수

    매개변수:
        system_prompt: AI 역할, 행동 지침
        user_prompt: 실제 작업 지시

    반환값:
        모델이 생성한 텍스트 (str)
    """
    client = get_client() # 함수 호출
    if client is None:
        raise RuntimeError('OPENAI_API_KEY가 없습니다. .env파일을 확인하세요!')
    
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        temperature=temperature,
        messsages=[
            {'role':'system', 'content': system_prompt},
            {'role':'user', 'content': user_prompt},
        ],
    )

    return response.choices[0].message.content or ''

def safe_json_loads(text: str) -> dict[str, Any]:
    """모델 응답해서 JSON 파싱
    마크다운 코드 블록을 감싸서 주는 경우가 많다.
    이 함수는 코드블록 기호를 먼저 제거한 뒤 파싱한다.
    """
    cleaned = text.strip() # 양쪽 공백 제거

    # 마크다운 코드 브록 제거 ('''json 먼저 --->''')
    if cleaned.startswith("'''json"):
        cleaned = cleaned.removeprefix("'''json").strip() # 앞에 붙은 '''jpson 제거
    if cleaned.startswith("'''"):
        cleaned = cleaned.removeprefix("'''").strip() # 앞에 붙은 ''' 제거
    if cleaned.endswith("'''"):
        cleaned = cleaned.removesuffix("'''").strip() # 끝에 붙은 ''' 제거

    try:
        return json.loads(cleaned) # 정상 파싱 성공!
    except json.JSONDecodeError:
        # 파싱 실패 시 원문을 "raw_response"키에 그대로 담아 반환
        return {'raw_response': text}

def make_download_text(topic: str, report: str, script: str, qa: str) -> str: 
    """
    보고서, 스크립트, Q&A를 하나의 마크다운 파일로 합친다.
    현재 날짜와 시간도 기록
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M") # 예시) 2026-06-29 20:08
    return f'''# AI Research Agent 보고서

- 주제: {topic}
- 생성 시각: {now}

---

## 1. 최종 보고서

{report}

---

## 2. 발표 스크립트

{script}

---

## 3. 예상 질문과 답변

{qa}

'''

# ---------------------------------------------------------------
# 2. 프롬프트 빌더 함수
# 
# - 각 단계마다 독립된 함수로 프롬프트를 만들어 (system, user) 튜플로 반환
# - 문제가 생길 때 프롬프트만 단독으로 수정, 테스트 가능
# ---------------------------------------------------------------

def build_plan_prompt(topic: str, audience: str, goal: str) -> tuple[str, str]:
    """
    조사 계획 생성 프롬프트 함수
    - 보고서를 쓰기 전, 무엇을 어떻게 조사할지 계획을 세운다.
    - Agent 관점: 계획(Planning) 단계에 해당한다.
    """
    system = '''당신은 AI/IT 수업용 리서치 코치입니다.
    학생이 조사 주제를 정하면, 실제 보고서 작성을 위한 조사 계획을 명확하고 실용적으로
    제안합니다.
    출력은 반드시 한국어로 작성하세요.'''

    user = f'''
조사 주제: {topic}
대상 독자/청중: {audience}
보고서 목적: {goal}

아래 형식으로 조사 계획을 작성해 주세요.
1. 핵심 질문 5개
2. 찾아야 할 자료 유형 5개
3. 보고서 목차 초안
4. 좋은 자료인지 판단하는 기준
5. 학생이 바로 검색할 수 있는 검색어 8개

'''
    return system, user 

def build_summary_prompt(topic:str, pasted_sources: str) -> tuple[str,str]:
    """
    붙여놓은 자료를 요약하는 프롬프트 함수
    --> 사용자가 직접 붙여넣은 자료만 분석하여 핵심을 추출한다.
    Agent관점: '정보 수집, 분석(Observation)'단계에 해당된다.
    """
    system = '''당신은 자료 분석가입니다.
    사용자가 붙여넣은 자료만 근거로 핵심 내용을 요약합니다.
    자료에 없는 내용은 추측하지 말고, "제공 자료에서 확인 되지 않음"이라고 표시하세요.
    출력은 한국어로 작성하세요.'''

    user = f'''
주제: {topic}

[붙여넣은 자료]
{pasted_sources}

아래 형식으로 정리해주세요.
1. 핵심 요약 5줄
2. 중요한 사실/수치/근거
3. 서로 다른 관점 또는 쟁점
4. 보고서에 반드시 넣을 포인트
5. 추가로 확인하면 좋은 내용
'''
    return system, user

def build_report_prompt(topic: str, audience: str, goal: str, pasted_sources: str) -> tuple[str, str]: 
    """
    최종 보고서 생성 프롬프트 함수
    --> 수집한 자료를 바탕으로 포트폴리오용 완성 보고서를 작성한다.
    Agent 관점 : '실행(Action)' 단계
    """
    system = '''당신은 수업용 보고서를 작성하는 AI 리서치 에이전트입니다.
    사용자가 제공한 자료를 중심으로, 포트폴리오에 넣을 수 있는 깔끔한 보고서를 작성합니다.
    근거없는 과장은 피하고, 자료에 없는 내용은 명확히 구분하세요.
    출력은 한국어 Markdown으로 작성하세요.'''

    user = f'''
조사 주제: {topic}
대상 독자/청중: {audience}
보고서 목적: {goal}

[붙여넣은 자료]
{pasted_sources}

다음 구조로 보고서를 작성해주세요.
# 제목
## 1. 한 줄 요약
## 2. 배경
## 3. 핵심 내용
## 4. 활용 사례
## 5. 한계와 주의점
## 6. 결론
## 7. 포트폴리오 확장 아이디어
'''
    return system, user

def build_script_prompt(topic: str, report: str) -> tuple[str, str]:
    """
    발표 스크립트 생성하는 프롬프트 함수
    --> 작성된 보고서를 3분 발표 스크립트로 변환한다.
    Agent 관점: '변환(transform)' 단계
    """
    system = '''당신은 발표 코치입니다.
    보고서를 3분 발표용 스크립트로 바꾸고, 발표자가 자연스럽게 말할 수 있게 작성합니다.
    출력은 한국어로 작성하세요.'''

    user = f'''
주제: {topic}

[보고서]
{report}

아래 형식으로 작성해주세요.
1. 30초 오프닝
2. 2분 핵심 발표 스크립트
3. 30초 마무리
4. 발표 슬라이드 제목 5개
'''
    return system, user

def build_qa_prompt(topic: str, report: str) -> tuple[str, str]:
    """
    예상 Q&A 생성하는 프롬프트 함수
    --> 발표 후 나올 수 있는 예상 질문과 모범 답변을 준비한다.
    Agent 관점: '검증(Verification)' 단계
    """
    system = '''당신은 발표 후 질의 응답을 준비하는 코치입니다.
    예상 질문과 답변을 현실적으로 만듭니다.
    출력은 한국어로 작성하세요.'''

    user = f'''
주제: {topic}

[보고서]
{report}

예상 질문 7개와 모범 답변을 만들어주세요.
질문은 쉬운 질문, 비판적 질문, 기술적 질문이 섞이게 작성하세요.
'''
    return system, user

# ---------------------------------------------------------------
# 3. Streamlit UI
#
#
# ---------------------------------------------------------------

# 3-1. 페이지 기본 설정
st.set_page_config(
    page_title=APP_TITLE,
    page_icon='👨🏻‍💼',
    layout='wide',
)

st.title('👨🏻‍💼 AI Research Agent')
st.caption('실시간 웹검색 없이, 붙여넣은 자료 기반으로 보고서를 생성합니다.')

# 3-2. 사이드 바
with st.sidebar:
    st.header('실행 상태')

    #API 키 유무에 따라 초록색 성공 / 빨간색 에러 표시
    if os.getenv('OPENAI_API_KEY'):
        st.success('OPENAI_API_KEY 감지됨')
    else:
        st.error('OPENAI_API_KEY 없음')

    st.markdown('---')
    st.subheader('프로젝트 포인트')
    st.markdown(
        '''
- Agent를 꼭 복잡한 프레임워크로 만들 필요는 없다.
- '계획 -> 자료 분석 -> 보고서 -> 발표 -> Q&A'ç 처럼 단계를 나누면 워크플로우형 Agent 가 됩니다.
- 안정성을 위해 웹검색 API를 쓰지 않았습니다.
'''
    )

# 3-3. 레이아웃 
col1, col2 = st.columns([1,1]) # 1:1 비율

# 3-4. 입력 영역
with col1:
    st.subheader('1. 조사 설정')

    # value = : 웹을 처음 열었을 때 보여줄 기본 값
    topic = st.text_input('조사 주제', value='생성형 AI 가 교육 분야에 미치는 영향')
    audience = st.text_input('대상 독자/청중', value= 'AI 입문 수강생')
    goal = st.text_input('보고서 목적', value='수업 발표와 포트폴리오 정리')

    st.subheader('2. 자료 붙여넣기')
    pasted_sources = st.text_area(
        '뉴스, 블로그, 논문 초록, 회사 자료 등을 붙여넣으세요.',
        height=300,
        placeholder='여기에 조사 자료를 붙여넣으세요. /n예: 기사 요약, 공식 문서 일부, 통계 자료 등'
    )

# 3-5. 버튼 영역 (col2)
with col2:
    st.subheader('e. Agent 실행')
    st.info('처음에는 "조사 계획 생성"만 눌러도 실행이 가능합니다.')

    # width='stretch' --> 버튼이 컬럼 너비를 꽉 채운다.
    plan_btn = st.button('1. 조사 계획 생성', width='stretch')
    summary_btn = st.button('2. 붙여넣은 자료 요약', width='stretch')
    report_btn = st.button('3. 최종 보고서 생성', width='stretch')
    full_btn = st.button('4. 보고서 + 발표 + Q&A 한번에 생성', width='stretch')

#====================세션 상태 초기화 =================================================
# 스트림릿은 버튼을 누를 때 마다 스크립트 전체를 위->아래로 재실행한다.
# 일반 변수는 재실행 될 때마다 초기화된다.
# st.session_state에 저장한 값은 재실행 후에도 유지된다.
if 'plan' not in st.session_state:
    st.session_state.plan = ''
if 'summary' not in st.session_state:
    st.session_state.summary = ''
if 'report' not in st.session_state:
    st.session_state.report = ''
if 'script' not in st.session_state:
    st.session_state.script = ''
if 'qa' not in st.session_state:
    st.session_state.qa = ''

#===============버튼 클릭 처리=========================================================
# try~except 예외처리 --> 오류가 나도 프로그램이 비정상 종료 되지 않는다.
try:
    # 1. 조사 계획 생성
    if plan_btn: # 버튼이 눌러졌다면
        with st.spinner('조사 계획 생성 중...'):
            system, user = build_plan_prompt(topic, audience, goal) # 계획 함수 호출
            # 결과를 session_state 에 저장 -> 재실행 후에도 결과 유지
            st.session_state.plan = call_llm(system, user) # llm 함수 호출

    # 2. 자료 요약
    if summary_btn:
        # 자료 없이 요약버튼을 누르면 경고만 표시하고 API는 호출하지 않는다.
        if not pasted_sources.strip():
            st.warning('먼저 자료를 붙여넣어 주세요.')
        else:
            with st.spinner('자료 요약 중...'):
                system, user = build_summary_prompt(topic, pasted_sources) # 요약 함수 호출
                st.session_state.summary = call_llm(system, user) # llm 함수 호출
    
    # 3. 최종 보고서 생성
    if report_btn:
        if not pasted_sources.strip(): # 2번과 동일
            st.warning('먼저 자료를 붙여넣어 주세요.')
        else:
            with st.spinner('보고서 생성 중...'):
                system, user = build_report_prompt(topic, audience, goal, pasted_sources) # 보고서 함수 호출
                st.session_state.report = call_llm(system, user) # llm 함수 호출

    # 4. 보고서 + 발표 스크립트 + QnA 한번에 생성
    if full_btn:
        if not pasted_sources.strip():
            st.warning('먼저 자료를 붙여넣어 주세요.')
        else: 
            # 보고서 -> 스크립트 -> QnA (단계별)
            with st.spinner('보고서 생성 중...'):
                system, user = build_report_prompt(topic, audience, goal, pasted_sources)
                st.session_state.report = call_llm(system, user)
            
            # 이전 단계 결과(보고서)를 입력으로 사용
            with st.spinner('발표 스크립트 생성 중...'):
                system, user = build_script_prompt(topic, st.session_state.report)
                st.session_state.script = call_llm(system, user)

            # 이전 단계 결과(보고서)를 입력으로 사용
            with st.spinner('예상 QnA 생성 중...'):
                system, user = build_qa_prompt(topic, st.session_state.report)
                st.session_state.qa = call_llm(system, user)

except Exception as e: 
    st.error('실행 중 오류가 발생했습니다.')
    st.exception(e) # 디버깅에 유용

# ============= 결과 출력 탭 ===============================================================
st.markdown('---')
st.subheader('결과')

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ['조사 계획', '자료 요약', '최종 보고서', '발표 스크립트', '예상 QnA']
)

with tab1:
    if st.session_state.plan:
        st.markdown(st.session_state.plan)
    else: 
        st.caption('조사 계획이 아직 없습니다.') # 회색 작은 글자

with tab2:
    if st.session_state.summary:
        st.markdown(st.session_state.summary)
    else:
        st.caption('자료 요약이 아직 없습니다.')

with tab3:
    if st.session_state.report:
        st.markdown(st.session_state.report)
    else:
        st.caption('최종 보고서가 아직 없습니다.')

with tab4:
    if st.session_state.script:
        st.markdown(st.session_state.script)
    else:
        st.caption('발표 스크립트가 아직 없습니다.')

with tab5:
    if st.session_state.qa:
        st.markdown(st.session_state.qa)
    else:
        st.caption('예상 QnA가 아직 없습니다.')

# ======= 다운로드 버튼 표시 ==================================
# 보고서, 발표 스크립트, QnA 중 하나라도 있으면 다운로드 버튼을 표시한다.
if st.session_state.report or st.session_state.script or st.session_state.qa:
    download_text = make_download_text(
        topic = topic, 
        report = st.session_state.report,
        script= st.session_state.script,
        qa = st.session_state.qa,
    )
    st.download_button(
        label = 'Markdown 보고서 다운로드 ⬇️',
        data = download_text,
        file_name ='research_agent_report.md',
        mime = 'text/markdown', # 브라우저에 파일 타입 알려준다.
        width = 'stretch'   
    )

#========설명 접기/펼치기 ==========================
# st.expander: 클릭하면 펼쳐지는 섹션, 부가 설명을 숨겨두기 좋다.
with st.expander('이게 왜 Agent 인가요?'):
    st.markdown(
        '''
### 단순 챗봇 vs. 워크플로우형 Agent

| 구분 | 단순 챗봇 | 이 예제 (워크플로우형 Agent) |
|------|-----------|---------------------------|
| 처리 방식 | 질문 1개 → 답변 1개 | 목표를 여러 단계로 분해하여 순서대로 실행 |
| 상태 관리 | 없음 | `st.session_state`로 단계별 결과 보존 |
| 출력 체이닝 | 없음 | 보고서 → 스크립트 → Q&A 로 연결 |

### 이 앱의 Agent 흐름

```
사용자 입력 (주제, 자료)
        │
        ▼
① 조사 계획 생성    ← Planning 단계
        │
        ▼
② 자료 요약         ← Observation 단계
        │
        ▼
③ 보고서 작성       ← Action 단계 (핵심 산출물)
        │
        ▼
④ 발표 스크립트     ← Transform 단계
        │
        ▼
⑤ 예상 Q&A         ← Verification 단계
```

> 💡 **핵심**: 복잡한 프레임워크 없이도 "단계 분리 + 결과 체이닝"만으로
> 워크플로우형 Agent를 구현할 수 있습니다.
> LangGraph를 배우면 이 흐름을 그래프로 시각화하고 자동화할 수 있습니다.


'''
    )