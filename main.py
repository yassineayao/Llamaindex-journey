import argparse
import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type=int, required=True, help="Day number")
    args = parser.parse_args()

    if args.d == 0:
        from day0.main import RAG

        rag = RAG()
        response = rag.run("What did the author do growing up?")
        print(response)

    elif args.d == 1:
        from day1.main import RAG

        streaming = True
        rag = RAG(similarity_top_k=1, streaming=streaming)
        response = rag.run("What did the author do growing up?")
        if streaming:
            response.print_response_stream()  # type: ignore
        else:
            print(response)

    elif args.d == 2:
        from day2.main import Chatbot

        chatbot = Chatbot(similarity_top_k=3)
        chatbot.run()