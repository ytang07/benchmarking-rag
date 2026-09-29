"""OpenAI-compatible Databricks AI Gateway chat and judge calls."""

import json


def _content(response) -> str:
    return response.choices[0].message.content or ""


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
    return _content(response)


def judge_answer(
    question: str, context: str, generated_answer: str, model: str, base_url: str, token: str
) -> dict:
    """Return validated structured relevance data; malformed model output raises ValueError."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the full extra for the LLM judge") from exc
    response = OpenAI(base_url=base_url, api_key=token).chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an answer relevance judge. Return only JSON with exactly these keys: "
                    "score (number from 0 to 1), label (one of irrelevant, partial, relevant), "
                    "and rationale (short string). Score how well the answer addresses the question "
                    "using the retrieved context. Do not reward unsupported claims."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\nRetrieved context:\n{context}\n\n"
                    f"Generated answer:\n{generated_answer}"
                ),
            },
        ],
    )
    raw = _content(response).strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("judge returned invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("judge output must be a JSON object")
    score = parsed.get("score")
    label = parsed.get("label")
    rationale = parsed.get("rationale")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1:
        raise ValueError("judge score must be a number between 0 and 1")
    if label not in {"irrelevant", "partial", "relevant"}:
        raise ValueError("judge label is invalid")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("judge rationale must be a non-empty string")
    return {"score": float(score), "label": label, "rationale": rationale.strip()}
