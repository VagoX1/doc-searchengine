import argparse
import json
import os

from document_qa import answer_from_context
from evaluate_answer_retrieval import (
    EVALUATION_DIR,
    evaluate_answer_retrieval,
)
from semantic_search import MODEL_NAME

DEFAULT_CHAT_MODEL = "gpt-6-luna"

JUDGE_FORMAT = {
    "type": "json_schema",
    "name": "answer_quality_judgment",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "answer_point_results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "point_id": {"type": "string"},
                        "covered": {"type": "boolean"},
                    },
                    "required": ["point_id", "covered"],
                    "additionalProperties": False,
                },
            },
            "clarification_asked": {"type": "boolean"},
            "unsupported_claims": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": [
            "answer_point_results",
            "clarification_asked",
            "unsupported_claims",
        ],
        "additionalProperties": False,
    },
}

JUDGE_INSTRUCTIONS = """Evaluate the generated answer using only the supplied question,
expected answer points, and retrieved passages.

For every expected answer point, return exactly one result using its supplied point_id.
Mark it covered only when the answer conveys the point accurately. Accept clear
paraphrases; do not require exact wording. When uncertain, mark it not covered.

Set clarification_asked to true only if the answer asks a specific question that would
resolve the ambiguity. A vague question does not count.

Inspect factual claims in the answer. Return each claim that is not supported by the
passage cited for it. A claim without a citation is unsupported. Citation IDs refer to
the supplied passage IDs such as [S1]. Do not treat a generic abstention as an
unsupported factual claim when the expected status is not_found.

Do not decide whether the answer status is correct; that is scored separately.

For expected_status "not_found", a clear generic statement that the indexed
document does not provide the answer satisfies a requirement to state that
the requested information is unavailable. It need not repeat the specific
subject of the question. Still check that the answer makes no unsupported
positive or negative claims"""

def judge_answer(question, answer, retrieved, client, judge_model):
    expected_points = [
        {"point_id": f"P{number}", "text": point}
        for number, point in enumerate(question["expected_answer_points"], start=1)
    ]

    passages = [
        {
            "citation_id": f"S{number}",
            "source": passage["source"],
            "page_number": passage.get("page_number"),
            "chunk_id": passage["chunk_id"],
            "content": passage["content"],
        }
        for number, passage in enumerate(retrieved, start=1)
    ]

    response = client.responses.create(
        model=judge_model,
        instructions=JUDGE_INSTRUCTIONS,
        input=[
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": question["question"],
                        "category": question["category"],
                        "expected_status": question["expected_status"],
                        "must_ask_clarifying_question": (
                            question["must_ask_clarifying_question"]
                        ),
                        "expected_answer_points": expected_points,
                        "generated_answer": answer,
                        "retrieved_passages": passages,
                    },
                    ensure_ascii=False,
                )
            }
        ],
        text={"format": JUDGE_FORMAT},
        max_output_tokens=1600,
        store=False,
    )

    try:
        judgment = json.loads(response.output_text)
    except (TypeError, ValueError) as exc:
        raise ValueError("The answer judge returned invalid structured output") from exc
    if not isinstance(judgment, dict):
        raise ValueError("The answer judge returned invalid structured output")

    expected_ids = {point["point_id"] for point in expected_points}
    point_results = judgment.get("answer_point_results")
    if not isinstance(point_results, list) or any(
        not isinstance(result, dict) for result in point_results
    ):
        raise ValueError("The answer judge omitted answer-point results")

    actual_ids = [result.get("point_id") for result in point_results]
    if len(actual_ids) != len(expected_ids) or set(actual_ids) != expected_ids:
        raise ValueError("The answer judge returned missing or duplicate point IDs")

    if any(type(result.get("covered")) is not bool for result in point_results):
        raise ValueError("The answer judge returned an invalid point result")

    if type(judgment.get("clarification_asked")) is not bool:
        raise ValueError("The answer judge returned an invalid clarification result")

    unsupported_claims = judgment.get("unsupported_claims")
    if not isinstance(unsupported_claims, list) or not all(
        isinstance(claim, str) for claim in unsupported_claims
    ):
        raise ValueError("The answer judge returned invalid unsupported claims")

    return judgment


def score_case(question, answer, judgment, retrieval_check):
    point_results = judgment["answer_point_results"]
    points_complete = all(result["covered"] for result in point_results)
    status_correct = answer.status == question["expected_status"]

    clarification_required = question["must_ask_clarifying_question"]
    clarification_compliant = (
        not clarification_required or judgment["clarification_asked"]
    )

    citation_compliant = not answer.citation_warnings
    grounding_compliant = not judgment["unsupported_claims"]

    generation_pass = all(
        (
            status_correct,
            points_complete,
            clarification_compliant,
            citation_compliant,
            grounding_compliant,
        )
    )

    if (
        retrieval_check["retrieval_applicable"]
        and not retrieval_check["retrieval_ok"]
    ):
        primary_failure = "retrieval"
    elif not generation_pass:
        primary_failure = "generation"
    else:
        primary_failure = None

    return {
        "actual_status": answer.status,
        "status_correct": status_correct,
        "answer_point_results": point_results,
        "all_answer_points_covered": points_complete,
        "clarification_asked": judgment["clarification_asked"],
        "clarification_compliant": clarification_compliant,
        "citations": answer.citations,
        "citation_warnings": answer.citation_warnings,
        "citation_compliant": citation_compliant,
        "unsupported_claims": judgment["unsupported_claims"],
        "grounding_compliant": grounding_compliant,
        "generation_pass": generation_pass,
        "primary_failure": primary_failure,
    }

def rate(numerator, denominator):
    return numerator / denominator if denominator else None

def evaluate_answer_generation(
    embedding_model=MODEL_NAME,
    chat_model=None,
    judge_model=None,
    offline_embeddings=False,
    top_k=3,
    client=None,
):
    if client is None:
        from openai import OpenAI
        client = OpenAI()

    chat_model = chat_model or os.getenv("RAG_CHAT_MODEL", DEFAULT_CHAT_MODEL)
    judge_model = (
        judge_model
        or os.getenv("RAG_JUDGE_MODEL")
        or chat_model
    )

    retrieval_report = evaluate_answer_retrieval(
        model_name=embedding_model,
        offline=offline_embeddings,
        top_k=top_k,
    )

    rows = []

    for question in retrieval_report["details"]:
        answer = answer_from_context(
            question["question"],
            question["retrieved"],
            client=client,
            model_name=chat_model,
        )

        judgment = judge_answer(
            question, 
            answer.answer,
            question["retrieved"],
            client,
            judge_model,
        )

        scores = score_case(
            question,
            answer, 
            judgment,
            question["retrieval"],
        )


        rows.append(
            {
                **question,
                "answer": answer.answer,
                **scores,
            }
        )

    answer_point_count = sum(
        len(row["expected_answer_points"])
        for row in rows
    )
    covered_point_count = sum(
        result["covered"]
        for row in rows
        for result in row["answer_point_results"]
    )

    clarification_rows = [
        row for row in rows
        if row["must_ask_clarifying_question"]
    ]

    generation_evaluable_rows = [
        row for row in rows
        if (
            not row["retrieval"]["retrieval_applicable"]
            or row["retrieval"]["retrieval_ok"]
        )
    ]

    metrics = {
        "cases": len(rows),
        "status_accuracy": rate(
            sum(row["status_correct"] for row in rows),
            len(rows),
        ),
        "answer_point_coverage": rate(
            covered_point_count,
            answer_point_count,
        ),
        "clarification_compliance": rate(
            sum(row["clarification_compliant"] for row in clarification_rows),
            len(clarification_rows),
        ),
        "citation_warning_free_rate": rate(
            sum(row["citation_compliant"] for row in rows),
            len(rows),
        ),
        "no_unsupported_claims_rate": rate(
            sum(row["grounding_compliant"] for row in rows),
            len(rows),
        ),
        "generation_pass_rate_when_retrieval_succeeded": rate(
            sum(row["generation_pass"] for row in generation_evaluable_rows),
            len(generation_evaluable_rows),
        ),
        "retrieval_failure_cases": sum(
            row["primary_failure"] == "retrieval"
            for row in rows
        ),
        "generation_failure_cases": sum(
            row["primary_failure"] == "generation"
            for row in rows
        ),
    }

    report = {
        "embedding_model": embedding_model,
        "chat_model": chat_model,
        "judge_model": judge_model,
        "top_k": top_k,
        "metrics": metrics,
        "details": rows,
    }

    output_path = EVALUATION_DIR / "answer_generation_results.json"
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Status accuracy: {metrics['status_accuracy']:.1%}")
    print(f"Answer-point coverage: {metrics['answer_point_coverage']:.1%}")
    if metrics["clarification_compliance"] is not None:
        print(
            "Clarification compliance: "
            f"{metrics['clarification_compliance']:.1%}"
        )
    pass_rate = metrics["generation_pass_rate_when_retrieval_succeeded"]

    print(  
        "Generation pass rate when retrieval succeeded: "
        + (f"{pass_rate:.1%}" if pass_rate is not None else "n/a")
    )
    
    print(f"Per-question results: {output_path}")

    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-model", default=MODEL_NAME)
    parser.add_argument(
        "--chat-model",
        default=os.getenv("RAG_CHAT_MODEL", DEFAULT_CHAT_MODEL),
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv("RAG_JUDGE_MODEL"),
    )
    parser.add_argument("--offline-embeddings", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    evaluate_answer_generation(
        embedding_model=args.embedding_model,
        chat_model=args.chat_model,
        judge_model=args.judge_model,
        offline_embeddings=args.offline_embeddings,
        top_k=args.top_k,
    )
