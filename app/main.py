from app.graph import ask
from app.vector_store import seed_chroma


def main():
    try:
        seed_chroma()

        print("\nTrade Compliance Assistant")
        print("Type 'exit' to stop.\n")

        while True:
            question = input("You: ").strip()

            if question.lower() == "exit":
                break

            if not question:
                continue

            answer = ask(question)
            print("\nAssistant:", answer, "\n")

    except KeyboardInterrupt:
        print("\nStopped.")

    except Exception as exc:
        print(
            "\nApplication failed:",
            type(exc).__name__,
            str(exc),
        )


if __name__ == "__main__":
    main()
