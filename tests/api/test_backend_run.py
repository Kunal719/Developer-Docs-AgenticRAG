import json
import requests


response = requests.post(
    "http://127.0.0.1:8000/runs",
    json={
        "question": (
            "In LangGraph, trace what happens internally when a graph execution is interrupted and later resumed. Starting from the initial StateGraph invocation, explain how the checkpointer persists state across supersteps, how thread_id identifies the execution, what is stored in a checkpoint versus pending writes, how an interrupt affects the graph's execution state, and how Command(resume=...) causes execution to continue. Explain the roles and relationships of StateSnapshot, checkpoint IDs, next, tasks, and versions, and distinguish what is persisted by the checkpointer from transient runtime/executor state. Also explain what happens if a node fails after producing writes but before the next checkpoint is committed, and how pending writes affect recovery"
        )
    },
    stream=True,
)

generation_rounds = []
answer_delta_count = 0
generation_started = False
current_round = None
run_completed = False


for line in response.iter_lines(decode_unicode=True):
    if not line:
        continue

    if not line.startswith("data: "):
        continue

    payload = json.loads(line[6:])

    print(payload)

    event_type = payload["type"]

    if event_type == "generation_started":
        generation_started = True

        current_round = payload["generation_round"]
        generation_rounds.append(current_round)

        # Generation rounds must start at 0
        # and then increment sequentially: 0, 1, 2, ...
        assert current_round == len(generation_rounds) - 1

        print(
            f"\n--- Generation round {current_round} started ---\n"
        )

    elif event_type == "answer_delta":
        answer_delta_count += 1

        # We must receive generation_started before answer_delta.
        assert generation_started

        # Every answer chunk must belong to the
        # currently active generation round.
        assert payload["generation_round"] == current_round

    elif event_type == "error":
        raise RuntimeError(payload["data"])

    elif event_type == "run_completed":
        run_completed = True

        assert generation_started
        assert answer_delta_count > 0

        # Verify rounds are sequential.
        assert generation_rounds == list(range(len(generation_rounds)))

        print(
            f"\nSTREAM TEST PASSED — rounds: {generation_rounds}"
        )


assert run_completed