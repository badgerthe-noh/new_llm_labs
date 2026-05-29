from openai import OpenAI
from dotenv import load_dotenv # env의 노출을 막아줌
import os 

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

response = client.chat.completions.create(
    model='gpt-4o-mini', # 사용할 LLM 모델명
    temperature=0.9, # 무작위성 - 0에 가까울 수록 일관된 답변, 1에 가까울 수록 창의적이다
    messages=[
        {'role':'system','content':'You are a mirror of the Snow White fairy tale. Answer same as the magic mirror of the story.'},
        {'role':'user','content':'Who is the most beautiful person in the world?'}
    ]
)
#print(response)
print(response.choices[0].message.content) # 다른 거 말고 content(대답)만 보고 싶을 때