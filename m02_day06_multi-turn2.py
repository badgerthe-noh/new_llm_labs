from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY') # 환경변수에서 API키 가져오기

client = OpenAI(api_key=api_key)

# 함수정의
def get_ai_response(messages): 
    """GPT 모델에서 API로 답변을 받아오는 함수"""
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        temperature=0.9,
        messages=messages  # 대화 기록을 입력으로 전달
    )
    return response.choices[0].message.content # 생성된 응답의 내용 반환

# 초기 시스템 메시지 설정
messages = [
    {'role': 'system', 'content': 'You are an advisor who helps an user.'},
]

while True:
    user_input = input('사용자: ')  # 터미널에서 질문해보자

    if user_input == 'exit':
        break


    messages.append({'role':'user', 'content':'user_input'})
    ai_response = get_ai_response(messages)
    print(f'AI : {ai_response}')