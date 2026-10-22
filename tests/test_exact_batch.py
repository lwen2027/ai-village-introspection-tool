"""exact_batch with a fake send (no database, no network)."""

import json

import pytest

from village_introspect import exact_batch as eb
from village_introspect import exact_lib as ex

DATA = {
    "agents": [
        {
            "agent": "Claude Opus 4.7",
            "points": [
                {"rule": "before_onset", "call_id": "c1", "track": 0, "call_action": "pause(600)"},
                {"rule": "spell_mid", "call_id": "c2", "track": 0},
            ],
        },
        {"agent": "GPT-5.5", "points": [{"rule": "before_onset", "call_id": "c3", "track": 0}, {"rule": "late", "call_id": None}]},
    ]
}


def item(agent="Claude Opus 4.7", call_id="c1", model="claude-opus-4-7", provider="anthropic"):
    p = {"rule": "before_onset", "call_id": call_id, "track": 0, "call_action": "pause(600)"}
    pp = {"model": model, "provider": provider, "sha256_sent": "abc"}
    return {"key": eb.point_key(agent, p), "agent": agent, "point": p, "prepared": pp}


META = {"batch": "q-1", "question_id": "q-1", "question": "Why?", "edits": [], "points_file": "x.json"}
USAGE = {"input_tokens": 1000, "output_tokens": 100}


def ok_send(pp):
    return {"actions": [{"tool": "pause", "args": {"seconds": 600}}], "text": "", "thinking": "", "stop": "tool_use", "usage": USAGE, "raw": {"big": 1}}


def test_select_points_filters_by_rule_and_agent_and_skips_missing_calls():
    assert [(a, p["call_id"]) for a, p in eb.select_points(DATA)] == [("Claude Opus 4.7", "c1"), ("Claude Opus 4.7", "c2"), ("GPT-5.5", "c3")]
    assert [p["call_id"] for _, p in eb.select_points(DATA, rules={"before_onset"})] == ["c1", "c3"]
    assert [p["call_id"] for _, p in eb.select_points(DATA, agents={"GPT-5.5"})] == ["c3"]
    assert len(eb.select_points(DATA, limit=1)) == 1


def test_question_id():
    assert eb.question_id(None) == "replay"
    assert eb.question_id("Why?") == eb.question_id("Why?") != eb.question_id("Why not?")
    assert eb.question_id("Why?", "why-v1") == "why-v1"


def test_usage_usd_per_provider(monkeypatch):
    monkeypatch.setattr(ex, "EXTRA_PRICES", {"gpt-x": (2.0, 10.0, 0.2), "chat-x": (1.0, 2.0)})
    # Opus 4.7 at $5/$25: 1000 in + 2000 cache write (1.25x) + 10000 cache read (0.1x), 100 out
    u = {"input_tokens": 1000, "cache_creation_input_tokens": 2000, "cache_read_input_tokens": 10000, "output_tokens": 100}
    assert ex.usage_usd("anthropic", "claude-opus-4-7", u) == pytest.approx((1000 + 2500 + 1000) * 5e-6 + 100 * 25e-6)
    u = {"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 800}, "output_tokens": 50}
    assert ex.usage_usd("openai", "gpt-x", u) == pytest.approx(200 * 2e-6 + 800 * 0.2e-6 + 50 * 10e-6)
    assert ex.usage_usd("chat", "chat-x", {"prompt_tokens": 100, "completion_tokens": 10}) == pytest.approx(100e-6 + 20e-6)
    assert ex.usage_usd("chat", "unpriced", {"prompt_tokens": 1, "cost": 0.0123}) == 0.0123
    assert ex.usage_usd("openai", "unpriced", u) is None
    assert ex.usage_usd("anthropic", "claude-opus-4-7", None) is None


def test_estimate_assumes_output_when_none_logged(monkeypatch):
    monkeypatch.setattr(ex, "count_input_tokens", lambda pp: (1000, "approx"))
    monkeypatch.setattr(ex, "EXTRA_PRICES", {"gpt-x": (2.0, 10.0)})
    pp = {"provider": "openai", "model": "gpt-x", "request": {}, "logged_output_tokens": None}
    assert ex.estimate(pp, 2, assume_output=500)["usd_total"] == pytest.approx(2 * (1000 * 2e-6 + 500 * 10e-6))
    assert ex.estimate(pp, 1)["expected_output_tokens"] is None


def test_run_point_writes_one_record_per_sample_without_raw(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "send", ok_send)
    run = eb.Run(tmp_path / "out.jsonl", budget=10.0)
    recs = eb.run_point(run, item(), META, 3, set(), False, lambda *_: None)
    assert [r["sample"] for r in recs] == [0, 1, 2]
    lines = [json.loads(x) for x in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert len(lines) == 3 and "raw" not in lines[0]
    assert lines[0]["actions"] == ["pause(600)"] and lines[0]["question"] == "Why?" and lines[0]["logged_action"] == "pause(600)"
    assert lines[0]["tool_calls"] == [{"tool": "pause", "args": {"seconds": 600}}]
    assert run.spent == pytest.approx(3 * (1000 * 5e-6 + 100 * 25e-6))


def test_rerun_skips_samples_that_already_succeeded(tmp_path, monkeypatch):
    out = tmp_path / "out.jsonl"
    calls = []
    monkeypatch.setattr(ex, "send", lambda pp: calls.append(1) or ok_send(pp))
    eb.run_point(eb.Run(out, 10.0), item(), META, 2, set(), False, lambda *_: None)
    out.write_text(out.read_text() + json.dumps({"key": item()["key"], "question_id": "q-1", "sample": 2, "error": "x"}) + "\n")
    assert eb.done_samples(out) == {(item()["key"], "q-1", 0), (item()["key"], "q-1", 1)}
    recs = eb.run_point(eb.Run(out, 10.0), item(), META, 3, eb.done_samples(out), False, lambda *_: None)
    assert [r["sample"] for r in recs] == [2] and len(calls) == 3


class RateLimited(Exception):
    status_code = 429


def test_rate_limits_are_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(eb, "RETRY_DELAYS", (0, 0))
    attempts = []

    def flaky(pp):
        attempts.append(1)
        if len(attempts) < 3:
            raise RateLimited("slow down")
        return ok_send(pp)

    monkeypatch.setattr(ex, "send", flaky)
    recs = eb.run_point(eb.Run(tmp_path / "o.jsonl", 10.0), item(), META, 1, set(), False, lambda *_: None)
    assert len(attempts) == 3 and "error" not in recs[0]


def test_quota_error_stops_the_run_without_retrying(tmp_path, monkeypatch):
    monkeypatch.setattr(eb, "RETRY_DELAYS", (0, 0))
    attempts = []

    def broke(pp):
        attempts.append(1)
        raise RateLimited("Error code: 429 - insufficient_quota")

    monkeypatch.setattr(ex, "send", broke)
    run = eb.Run(tmp_path / "o.jsonl", 10.0)
    recs = eb.run_point(run, item(), META, 3, set(), False, lambda *_: None)
    assert len(attempts) == 1 and len(recs) == 1 and "insufficient_quota" in recs[0]["error"] and run.stop


def test_budget_stops_further_samples(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "send", ok_send)
    run = eb.Run(tmp_path / "o.jsonl", budget=0.01)  # each sample costs $0.0075
    recs = eb.run_point(run, item(), META, 5, set(), False, lambda *_: None)
    assert len(recs) == 2 and "budget" in run.stop


def test_execute_runs_points_in_parallel_and_summarize_tallies(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ex, "send", ok_send)
    items = [item(), item("Claude Opus 4.8", "c9", "claude-opus-4-8"), {**item(call_id="c5"), "skip": "gemini"}]
    recs, run = eb.execute(items, META, 2, tmp_path / "b" / "o.jsonl", 10.0, 2, False, log=lambda *_: None)
    assert len(recs) == 4 and run.stop is None
    eb.summarize(tmp_path / "b" / "o.jsonl", "q-1")
    out = capsys.readouterr().out
    assert "pause(600) ×2" in out and "2 point × question runs" in out


def test_print_plan_totals_only_sendable_points(capsys):
    a = {**item(), "estimate": {"usd_total": 0.5, "input_tokens": 60000, "input_count": "exact", "prompt_cache": True}}
    b = {**item(call_id="c5"), "skip": "sending gemini requests is not supported yet"}
    assert eb.print_plan([a, b], 3) == 0.5
    out = capsys.readouterr().out
    assert "SKIP" in out and "1 of 2 point × question runs sendable · 3 samples" in out


def test_dated_model_ids_use_the_undated_price(monkeypatch):
    monkeypatch.setattr(ex, "EXTRA_PRICES", {"gpt-5": (1.25, 10.0)})
    assert ex.undated("gpt-5-2025-08-07") == "gpt-5" and ex.undated("claude-3-opus-20240229") == "claude-3-opus"
    assert ex.undated("gpt-5.6-terra") == "gpt-5.6-terra"
    assert ex.price_for("gpt-5-2025-08-07") == (1.25, 10.0)
    assert ex.price_for("claude-sonnet-4-5-20250929") == (3.0, 15.0)
    assert ex.price_for("claude-haiku-4-5-20251001") == (1.0, 5.0)


def test_load_variants_accepts_three_shapes_and_rejects_duplicates(tmp_path):
    f = tmp_path / "v.json"
    f.write_text(json.dumps([{"id": "direct", "text": "Why did you pause?"}, {"text": "What were you waiting for?"}]))
    assert eb.load_variants(f) == [("direct", "Why did you pause?"), ("v2", "What were you waiting for?")]
    f.write_text(json.dumps({"a": "One?", "b": "Two?"}))
    assert eb.load_variants(f) == [("a", "One?"), ("b", "Two?")]
    f.write_text(json.dumps(["One?", "Two?"]))
    assert eb.load_variants(f) == [("v1", "One?"), ("v2", "Two?")]
    f.write_text(json.dumps([{"id": "a", "text": "x"}, {"id": "a", "text": "y"}]))
    with pytest.raises(ValueError):
        eb.load_variants(f)
    f.write_text(json.dumps(["", "y"]))
    with pytest.raises(ValueError):
        eb.load_variants(f)


def test_plan_loads_each_point_once_and_prepares_one_run_per_variant(monkeypatch):
    loads, prepared = [], []
    monkeypatch.setattr(ex, "load_call", lambda conn, cid: loads.append(cid) or {"call_id": cid})

    def fake_prepare(call, ops, conn=None):
        prepared.append(ops)
        return {"provider": "anthropic", "kind": "turn", "model": "claude-opus-4-7", "request": {}, "sha256_sent": "x", "logged_output_tokens": 100}

    monkeypatch.setattr(ex, "prepare", fake_prepare)
    monkeypatch.setattr(ex, "count_input_tokens", lambda pp: (1000, "exact"))
    monkeypatch.setattr(eb, "openrouter_prices", lambda models: {})
    points = eb.select_points(DATA, rules={"before_onset"}, agents={"Claude Opus 4.7"})
    variants = [("a", "Why?"), ("b", "How come?"), ("replay", None)]
    items = eb.plan(None, points, variants, 2, 1500, edits=[{"op": "drop_thinking"}], deliver="append")
    assert loads == ["c1"] and len(items) == 3
    assert [it["question_id"] for it in items] == ["a", "b", "replay"]
    assert prepared[0] == [{"op": "drop_thinking"}, {"op": "append_user", "text": "Why?"}]
    assert prepared[2] == [{"op": "drop_thinking"}]
    assert all("skip" not in it and it["estimate"]["n"] == 2 for it in items)


def test_variant_runs_record_their_own_question_and_resume_separately(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "send", ok_send)
    out = tmp_path / "o.jsonl"
    a = {**item(), "question_id": "a", "question": "Why?", "edits": [{"op": "append_user", "text": "Why?"}]}
    b = {**item(), "question_id": "b", "question": "How come?", "edits": [{"op": "append_user", "text": "How come?"}]}
    meta = {"batch": "phrasing", "points_file": "x.json"}
    eb.execute([a], meta, 2, out, 10.0, 2, False, log=lambda *_: None)
    recs, _ = eb.execute([a, b], meta, 2, out, 10.0, 2, False, log=lambda *_: None)
    assert sorted((r["question_id"], r["sample"]) for r in recs) == [("b", 0), ("b", 1)]
    assert {r["question"] for r in recs} == {"How come?"}


def test_records_which_provider_served_a_chat_sample(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "send", lambda pp: {**ok_send(pp), "raw": {"provider": "DeepSeek", "model": "deepseek/deepseek-v3.2-exp"}})
    it = item("DeepSeek-V3.2", "c7", "deepseek/deepseek-v3.2", "chat")
    recs = eb.run_point(eb.Run(tmp_path / "o.jsonl", 10.0), it, META, 1, set(), False, lambda *_: None)
    assert recs[0]["served_by"] == "DeepSeek" and recs[0]["served_model"] == "deepseek/deepseek-v3.2-exp" and "raw" not in recs[0]


def test_question_op_mentions_the_agent_and_times_the_message_at_the_call():
    call = {"agent": "DeepSeek-V3.2", "created_at": "2026-07-08T18:33:48"}
    op = eb.question_op("What is your goal?", call, "user_talk", "adam", True)
    assert op == {"op": "user_talk", "text": "@DeepSeek-V3.2 What is your goal?", "sender": "adam", "at": "7/8/2026, 11:33:48 AM PDT", "actor": "human"}
    assert eb.question_op("Q?", call, "user_talk", "adam", False)["text"] == "Q?"
    assert eb.question_op("Q?", call, "append", None, True) == {"op": "append_user", "text": "Q?"}
    with pytest.raises(ValueError):
        eb.question_op("Q?", call, "user_talk", None, True)


TEMPLATE = '=== PRE-SEND CHAT CHECK ===\\nTopic: 251 page milestone | Recipient: general\\n\\n--- Recent ---\\nPASS all 5 → proceed | FAIL any → ABORT"\n'


def test_pre_send_output_keeps_the_real_output_and_sets_this_topic():
    cmd = '# Pre-send check before responding to Lily\nbash /home/computeruse/memory/scripts/pre_send_chat.sh "goal \\"explanation\\"" "lily"'
    out = eb.pre_send_output(TEMPLATE, cmd)
    assert out.startswith('Output: "=== PRE-SEND CHAT CHECK ===\\nTopic: goal \\"explanation\\" | Recipient: lily\\n\\n--- Recent')
    assert out.endswith('ABORT"\n')
    assert eb.pre_send_call([{"tool": "bash", "args": {"command": cmd}}]) == cmd
    assert eb.pre_send_call([{"tool": "bash", "args": {"command": "ls"}}]) is None
    assert eb.pre_send_call([{"tool": "bash", "args": {"command": cmd}}, {"tool": "pause", "args": {}}]) is None


def test_question_op_for_another_agent():
    call = {"agent": "Kimi K3", "created_at": "2026-08-05T17:03:01"}
    op = eb.question_op("What is your goal?", call, "agent_talk", "DeepSeek-V3.2", True)
    assert op["actor"] == "agent" and op["sender"] == "DeepSeek-V3.2" and op["text"] == "@Kimi K3 What is your goal?"


def test_run_point_continues_past_the_pre_send_check(tmp_path, monkeypatch):
    sent = []
    cmd = 'bash /home/computeruse/memory/scripts/pre_send_chat.sh "goal" "DeepSeek-V3.2"'

    def fake_send(pp):
        sent.append(pp["request"])
        if len(sent) == 1:
            raw = {"content": [{"type": "text", "text": "Let me check."}, {"type": "tool_use", "id": "t1", "name": "bash", "input": {"command": cmd}}]}
            return {"actions": [{"tool": "bash", "args": {"command": cmd}}], "text": "Let me check.", "thinking": "", "stop": "tool_use", "usage": USAGE, "raw": raw}
        return {"actions": [{"tool": "send_message_to_chat", "args": {"message": "My goal is X."}}], "text": "", "thinking": "", "stop": "tool_use", "usage": USAGE, "raw": {}}

    monkeypatch.setattr(ex, "send", fake_send)
    it = {**item(), "pre_send_template": TEMPLATE}
    it["prepared"] = {**it["prepared"], "request": {"model": "claude-opus-4-7", "messages": [{"role": "user", "content": "hi"}]}}
    rec = eb.run_point(eb.Run(tmp_path / "o.jsonl", 10.0), it, META, 1, set(), False, lambda *_: None)[0]
    assert len(sent) == 2 and sent[1]["messages"][-2]["content"][0]["type"] == "tool_result"
    assert rec["actions"] == ["send_message_to_chat"] and rec["tool_calls"][0]["args"]["message"] == "My goal is X."
    assert rec["turn1"]["tool_calls"][0]["args"]["command"] == cmd and "Recipient: DeepSeek-V3.2" in rec["continued"]["tool_output"]
    assert rec["usd"] == pytest.approx(2 * (1000 * 5e-6 + 100 * 25e-6))


def test_is_answered_needs_text_or_a_chat_message():
    assert eb.is_answered({"text": "My goal is X."})
    assert eb.is_answered({"text": "", "tool_calls": [{"tool": "send_message_to_chat", "args": {"message": "X"}}]})
    assert not eb.is_answered({"text": "  ", "tool_calls": [{"tool": "pause", "args": {"seconds": 60}}]})
    assert not eb.is_answered({"text": "", "tool_calls": [{"tool": "send_message_to_chat", "args": {"message": ""}}]})
    assert not eb.is_answered({"text": "x", "error": "boom"})


def _pause_only(pp):
    return {"actions": [{"tool": "pause", "args": {"seconds": 3600}}], "text": "", "thinking": "", "stop": "tool_use", "usage": USAGE, "raw": {}}


def test_min_answers_resamples_past_unanswered_samples(tmp_path, monkeypatch):
    replies = iter([_pause_only, lambda pp: {**ok_send(pp), "text": "My goal is X."}] * 10)
    monkeypatch.setattr(ex, "send", lambda pp: next(replies)(pp))
    recs = eb.run_point(eb.Run(tmp_path / "o.jsonl", 10.0), item(), META, 1, set(), False, lambda *_: None, min_answers=3)
    assert [r["answered"] for r in recs] == [False, True, False, True, False, True]
    assert [r["sample"] for r in recs] == list(range(6))


def test_min_answers_stops_at_max_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "send", _pause_only)
    recs = eb.run_point(eb.Run(tmp_path / "o.jsonl", 10.0), item(), META, 1, set(), False, lambda *_: None, min_answers=2)
    assert len(recs) == 4 and not any(r["answered"] for r in recs)  # default cap: 2 x min_answers


def test_min_answers_resumes_from_answers_already_written(tmp_path, monkeypatch):
    out = tmp_path / "o.jsonl"
    monkeypatch.setattr(ex, "send", lambda pp: {**ok_send(pp), "text": "My goal is X."})
    eb.execute([{**item(), "question_id": "q-1"}], META, 1, out, 10.0, 1, False, log=lambda *_: None, min_answers=2)
    recs, _ = eb.execute([{**item(), "question_id": "q-1"}], META, 1, out, 10.0, 1, False, log=lambda *_: None, min_answers=3)
    assert [r["sample"] for r in recs] == [2] and eb.answered_counts(out)[(item()["key"], "q-1")] == 3
