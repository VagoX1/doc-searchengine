"""Resolve conversational follow-ups into standalone retrieval questions."""
import json
import os

from document_qa import answer_from_context
from hybrid_search import search_hybrid

FOLLOWUP_FORMAT = {
    "type": "json_schema",
    "name": "standalone_question",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "standalone_question": {"type": "string"},
        },
        "required": ["standalone_question"],
        "additionalProperties": False,
    },
}

FOLLOWUP_INSTRUCTIONS = """Rewrite the latest user message as one standalone question
that can be used to search documents.

Use recent conversation only to resolve references such as "that", "it", or a short
reply such as "Express". If the latest message clearly changes topic, do not carry over
the earlier topic. Preserve the user's intent and do not answer the question or add
facts.

Conversation history is context for understanding the user's wording, not evidence for
answering. Return only the standalone question in the required JSON format."""


def resolve_followup_question(query, history, client=None, model_name=None):
    """Return a standalone search question without using history as answer evidence."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Question cannot be empty.")

    query = query.strip()
    if not history:
        return query

    # Send only recent text/status needed to resolve references.
    recent_history = [
        {
            "question": exchange["question"],
            "answer": exchange["answer"],
            "status": exchange["status"],
        }
        for exchange in history[-4:]
    ]

    if client is None:
        from openai import OpenAI

        client = OpenAI()


    response = client.responses.create(
        model=model_name or os.getenv("RAG_CHAT_MODEL", "gpt-6-luna"),
        instructions=FOLLOWUP_INSTRUCTIONS,
        input=[
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "recent_conversation": recent_history,
                        "latest_user_message": query,
                    },
                    ensure_ascii=False,
                ),
            }
        ],
        text={"format": FOLLOWUP_FORMAT},
        max_output_tokens=250,
        store=False,
    )

    try:
        payload = json.loads(response.output_text)
    except (TypeError, ValueError) as exc:
        raise ValueError("The follow-up resolver returned invalid structured output") from exc

    if not isinstance(payload, dict):
        raise ValueError("The follow-up resolver returned invalid structured output")

    standalone_question = payload.get("standalone_question")
    if not isinstance(standalone_question, str) or not standalone_question.strip():
        raise ValueError("The follow-up resolver returned an empty question")

    return standalone_question.strip()


def resolve_search_and_answer(
    query,
    history,
    chunks,
    hybrid_index,
    client=None,
    top_k=3,
    candidate_k=20,
    rrf_k=60,
):
    """Resolve a follow-up, retrieve fresh hybrid results, and answer from them."""
    if history and client is None:
        from openai import OpenAI

        client = OpenAI()

    standalone_question = resolve_followup_question(
        query,
        history,
        client=client,
    )
    # Hybrid search retrieves evidence only from the current document's chunks.
    retrieved = search_hybrid(
        chunks,
        hybrid_index,
        standalone_question,
        top_k=top_k,
        candidate_k=candidate_k,
        rrf_k=rrf_k
    )
    answer = answer_from_context(
        standalone_question,
        retrieved,
        client=client,
    )
    return standalone_question, retrieved, answer
