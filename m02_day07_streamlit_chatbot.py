import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# 왼쪽 사이드 바
with st.sidebar:
    openai_api_key = os.getenv('OPENAI_API_KEY')
    '[Type your OpenAI API Key](https://platform.openai.com/api-keys)'

# 오른쪽(메인) 화면
st.title('🤖 ChatBot')

# 초기 질문 설정 (한번도 아직 대화를 안했을 때)
if 'messages' not in st.session_state: #첫 화면엔 당연히 메시지가 없을 것
    st.session_state['messages'] = [{'role': 'assistant', 'content': 'How can I help you?'}]

# 대화 기록을 출력
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

# 사용자의 입력을 받아 대화 기록에 추가하고 AI가 응답을 생성
if prompt := st.chat_input():
    if not openai_api_key:
        st.info('Plz add your OpenAI API key to continue.')
        st.stop()

    client = OpenAI(api_key=openai_api_key)
    st.session_state.messages.append({'role':'user', 'content':prompt})
    st.chat_message('user').write(prompt) # 화면에 보여짐
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=st.session_state.messages
    )
    msg=response.choices[0].message.content
    st.session_state.messages.append({'role':'assistant', 'content':msg})
    st.chat_message('assistant').write(msg)