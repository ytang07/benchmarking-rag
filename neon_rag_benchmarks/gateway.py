"""OpenAI-compatible Databricks AI Gateway chat calls."""


def answer(question: str, context: str, model: str, base_url: str, token: str) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the full extra for Databricks answers") from exc
    response = OpenAI(base_url=base_url, api_key=token).chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Answer the question using only the retrieved context. Cite every factual claim "
                    "with the exact stable identifier in square brackets, including page when shown "
                    "(for example [doc#chunk-0] or [doc#chunk-0|page=2]). Do not invent identifiers. "
                    f"\n\nRetrieved context:\n{context}\n\nQuestion: {question}"
                ),
            }
        ],
    )
    return response.choices[0].message.content or ""
