# from dotenv import load_dotenv
# load_dotenv()

# from langchain_google_genai import ChatGoogleGenerativeAI

# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

# while True:
#     user_input = input("user : ")
#     if user_input.lower() in ["exit","quit","bye"]:
#         print("GoodBye! 👋")
#         break
#     res = llm.invoke(user_input)
#     print("Bot :", res.content , "\n")


from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


def initialize_llm() -> ChatGoogleGenerativeAI:
    """
    Load environment variables and initialize the Gemini LLM.
    """
    load_dotenv()
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")


def chat_loop(llm: ChatGoogleGenerativeAI) -> None:
    """
    Run an interactive command-line chat loop.
    """
    print("🤖 Gemini Chatbot initialized. Type 'exit', 'quit', or 'bye' to stop.\n")

    while True:
        user_input = input("User: ").strip()

        if user_input.lower() in {"exit", "quit", "bye"}:
            print("\nGoodbye! 👋")
            break

        try:
            response = llm.invoke(user_input)
            print(f"Bot: {response.content}\n")
        except Exception as error:
            print("⚠️  An error occurred while generating a response.")
            print(f"Details: {error}\n")


def main() -> None:
    """
    Application entry point.
    """
    llm = initialize_llm()
    chat_loop(llm)


if __name__ == "__main__":
    main()
