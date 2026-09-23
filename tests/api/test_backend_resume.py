import json
import requests


RUN_ID = "test-resume-api"

response = requests.post(
    f"http://127.0.0.1:8000/runs/{RUN_ID}/resume",
    stream=True,
)

response.raise_for_status()

run_resumed = False
generation_started = False
answer_delta_count = 0
current_round = None
run_completed = False


for line in response.iter_lines(decode_unicode=True):
    if not line:
        continue

    if not line.startswith("data: "):
        continue

    payload = json.loads(line[6:])

    print(payload)

    # Resume must continue the same logical run.
    assert payload["run_id"] == RUN_ID

    event_type = payload["type"]

    if event_type == "run_resumed":
        run_resumed = True

        assert payload["generation_round"] is None

        print("\n--- Run resumed ---")

    elif event_type == "generation_started":
        assert run_resumed

        generation_started = True

        current_round = payload["generation_round"]

        print(
            f"\n--- Generation round {current_round} started ---\n"
        )

    elif event_type == "answer_delta":
        answer_delta_count += 1

        assert run_resumed
        assert generation_started
        assert payload["generation_round"] == current_round

    elif event_type == "error":
        raise RuntimeError(payload["data"])

    elif event_type == "run_completed":
        run_completed = True

        assert run_resumed
        assert generation_started
        assert answer_delta_count > 0

        print(
            f"\nRESUME STREAM TEST PASSED — "
            f"generation round: {current_round}"
        )


assert run_completed