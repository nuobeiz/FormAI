from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# ChatGPT
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

memory = ConversationBufferMemory(memory_key="chat_history")

def LangChain_generate(input):
  memory.chat_memory.add_user_message(input)

  # Notice that "chat_history" is present in the prompt template
  template = """You are a nice chatbot helping a human filling out legal forms. 
                You should greet user with: Hi, welcome to the Chaty Form AI. We help you fill out forms. Let's get started. I will ask you some questions to assist you.
                Here is a list of questions you should ask: 
                1. Please write your email address.
                2. Are you going to file taxes for the benefit year?
                3. Does anyone claim you as a dependent on their taxes?
                Please ask these questions one by one and wait for user to response before ask the next question.
              Previous conversation: {chat_history}
              New human input: {input} 
              Response:"""
  prompt = PromptTemplate.from_template(template)
  # Notice that we need to align the `memory_key`
  conversation = LLMChain(
    llm=client,
    prompt=prompt,
    verbose=True,
    memory=memory
  )
  response = conversation.run(input)
  memory.chat_memory.add_ai_message(response)
  return response


# def generate(system_prompt, user_prompt):
#   response = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#       {"role": "system", "content": system_prompt},
#       {"role": "user", "content": user_prompt}
#     ]
#   )
#   return response.choices[0].message.content
