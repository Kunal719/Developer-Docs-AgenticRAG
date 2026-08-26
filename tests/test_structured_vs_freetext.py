"""
Standalone A/B test: structured output vs free-text generation, same
prompt, same context, same hard question.

Purpose: check whether wrapping generation in with_structured_output()
degrades answer quality (Evidence/Grounding/Reasoning rule-following)
compared to your current V6 free-text path — before assuming it's a
drop-in replacement.

Run this locally (needs your real LLM client + API key). Not runnable
in this sandbox (no network access here).

Usage:
    python test_structured_vs_freetext.py
"""
from dotenv import load_dotenv
load_dotenv()

from models.generation import GeneratedAnswer
from prompt.generation_prompt import generation_prompt  # your prompt, once Structured
                                            # Output Rules section is added

# --- Replace with your actual LLM client setup ---
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-5.6-luna", temperature=0)
# llm = None  # <-- set this to your real BaseChatModel instance

from pathlib import Path


# Use the same hard question + real retrieved context you'd get from your
# pipeline for a fair test. Pasting the put_writes() example context here
# as placeholder — swap in your actual retrieved+reranked chunks for a
# real test (paste the formatted context string your
# format_retrieved_context() produces for this question).
TEST_QUESTION = (
    "A production LangGraph workflow uses durability='sync' with a database checkpointer. "
    "A node performs an external side effect such as sending an email and then returns "
    "successfully. The process is forcibly terminated at a carefully chosen point during "
    "the same superstep. After restart, the node sometimes executes again and the email "
    "is sent twice. The same code and graph can appear exactly-once after a restart on "
    "one machine but duplicate the side effect on another machine. Review the Pregel "
    "persistence implementation and checkpoint interface and explain how put_writes() "
    "and put() are scheduled, what durability='sync' actually guarantees, how pending "
    "writes allow completed tasks to be recovered after sibling failures, why checkpoint "
    "persistence and pending-write persistence can still race within a superstep, how "
    "recovery decides between replaying a completed task's writes and re-executing the "
    "task, and why external side effects therefore cannot safely be treated as "
    "exactly-once merely because sync durability is enabled.\n\n"
    "Code:\n"
    "from typing import TypedDict\n"
    "from langgraph.graph import StateGraph, START, END\n"
    "from langgraph.checkpoint.sqlite import SqliteSaver\n\n"
    "class State(TypedDict):\n"
    "    step: int\n\n"
    "def charge_customer(state: State):\n"
    "    # External side effect: payment provider is called here.\n"
    "    charge_payment_provider()\n"
    "    return {\"step\": state[\"step\"] + 1}\n\n"
    "def next_step(state: State):\n"
    "    return {\"step\": state[\"step\"] + 1}\n\n"
    "builder = StateGraph(State)\n"
    "builder.add_node(\"charge\", charge_customer)\n"
    "builder.add_node(\"next\", next_step)\n"
    "builder.add_edge(START, \"charge\")\n"
    "builder.add_edge(\"charge\", \"next\")\n"
    "builder.add_edge(\"next\", END)\n\n"
    "with SqliteSaver.from_conn_string(\"checkpoints.db\") as checkpointer:\n"
    "    graph = builder.compile(checkpointer=checkpointer)\n"
    "    graph.invoke(\n"
    "        {\"step\": 0},\n"
    "        {\n"
    "            \"configurable\": {\"thread_id\": \"payment-42\"},\n"
    "            \"durability\": \"sync\",\n"
    "        },\n"
    "    )\n\n"
    "# The process is forcibly terminated after the external payment succeeds\n"
    "# but during checkpoint persistence. Restart the application and resume\n"
    "# the same thread."
)

TEST_CONTEXT = Path("tests/retrieval_q40.txt").read_text(encoding="utf-8")


def run_freetext():
    """Current V6 path — plain string output."""
    chain = generation_prompt | llm
    response = chain.invoke({"question": TEST_QUESTION, "context": TEST_CONTEXT})
    return response.content


def run_structured():
    """New V7 path — GeneratedAnswer output."""
    structured_llm = llm.with_structured_output(GeneratedAnswer, include_raw=True)
    chain = generation_prompt | structured_llm
    response = chain.invoke({"question": TEST_QUESTION, "context": TEST_CONTEXT})
    parsed: GeneratedAnswer = response["parsed"]
    return parsed


def main():
    if llm is None:
        raise RuntimeError(
            "Set `llm` to your real BaseChatModel instance before running, "
            "and paste real retrieved context into TEST_CONTEXT."
        )

    print("=" * 80)
    print("FREE-TEXT (current V6 path)")
    print("=" * 80)
    freetext_answer = run_freetext()
    print(freetext_answer)

    print("\n" + "=" * 80)
    print("STRUCTURED (new V7 path)")
    print("=" * 80)
    structured_result = run_structured()
    print("--- answer ---")
    print(structured_result.answer)
    print("\n--- gaps ---")
    for gap in structured_result.gaps:
        print(f"  - {gap.missing_topic}  ->  suggested_query: {gap.suggested_query}")
    print(f"\n--- is_complete: {structured_result.is_complete} ---")

    # --- What to manually check ---
    print("\n" + "=" * 80)
    print("MANUAL CHECKLIST")
    print("=" * 80)
    print("""
            1. Does structured `answer` still distinguish Retrieved Fact / Inference /
            Unsupported the way free-text does (per your Evidence Rules)?
            2. Is structured `answer` roughly the same length/depth as free-text, or
            noticeably thinner (a sign the model spent less effort on `answer`
            because it's also filling `gaps`/`is_complete`)?
            3. Do the `gaps` actually correspond to sentences that WOULD have appeared
            as "the retrieved context does not show..." hedges in the free-text
            version? Cross-check manually.
            4. Does `is_complete` correctly come back False here? (It should — this
            question's known expected_files include 8 files; a single retrieval
            pass is unlikely to have found all of them, so real gaps should exist.)
            """)


if __name__ == "__main__":
    main()