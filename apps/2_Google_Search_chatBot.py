from dotenv import load_dotenv
load_dotenv()

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_mistralai import ChatMistralAI
from langchain.agents import create_agent

llm = ChatMistralAI(
    model = "mistral-small-2506"
)
search = GoogleSerperAPIWrapper()

agent = create_agent(
    model=llm,
    tools=[search.run],
    system_prompt="You are a helpful assistant that can answer questions using Google Search. Use the provided tool to search for information and provide accurate answers.",
)

while True:
    ques = input("User: ")
    if ques.lower() in ["exit" , "quit" , "bye"]:
        print("Exiting the chatbot. Goodbye!")
        break

    response = agent.invoke({"messages":[{"role":"user","content":ques}]})

    print("ChatBot: ", response["messages"][-1].content)