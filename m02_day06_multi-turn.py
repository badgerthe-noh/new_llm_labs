from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY') # 환경변수에서 API키 가져오기

client = OpenAI(api_key=api_key)

while True:
    user_input = input('사용자: ')  # 터미널에서 질문해보자

    if user_input == 'exit':
        break

    response = client.chat.completions.create(
        model='gpt-4o-mini',
        temperature=0.9,
        messages=[
            {'role':'system', 'content':'You are an advisor who helps an user.'},
            {'role':'user','content':user_input}
        ]
    )
    print(f'AI : {response.choices[0].message.content}')
