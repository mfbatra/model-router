"""Basic routing example using the built-in DI container."""

from model_router.core.container import DIContainer


def main() -> None:
    router = DIContainer.create_router(openai_key="sk-demo")

    response = router.complete(
        "Summarize the differences between SQL and NoSQL databases.",
        max_cost=0.05,
    )
    print("Model:", response.model_used)
    print("Cost:", response.cost)
    print("Response:", response.content)


if __name__ == "__main__":
    main()
