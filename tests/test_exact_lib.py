"""exact_lib on synthetic requests in each provider's logged shape (no database, no network)."""

import pytest

from village_introspect import exact_lib as ex

MEMORY = (
    "# Terra memory\n\n## Mission\n\nMaximize views.\n\n### Current decision\n\nAct only on a qualifying trigger:\n"
    "1. a defect\n2. a request\n\n## Analytics\n\n20 views.\n\n## Immediate next approach\n\nWait for a trigger.\n"
)
SYSTEM = (
    "You are a language model agent, named GPT-5.6 Terra.\n<tools>\n- bash: Run commands.\n</tools>\n"
    "<intention>\nThe intention you set at the start of this session was: Wait for a genuine trigger.\n</intention>\n"
    f"<internal_memory>\n{MEMORY}</internal_memory>\n"
)


def openai_req():
    return {
        "model": "gpt-5.6-terra",
        "reasoning": {"effort": "medium", "summary": "detailed"},
        "prompt_cache_options": {"mode": "explicit"},
        "tools": [{"type": "function", "name": "pause", "parameters": {"type": "object"}}],
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "input_text", "text": "Here is your starting screenshot"}]},
            {"id": "rs_1", "type": "reasoning", "summary": [{"type": "summary_text", "text": "thinking"}]},
            {"id": "fc_1", "type": "function_call", "name": "pause", "call_id": "c1", "arguments": '{"seconds":300}'},
            {"type": "function_call_output", "call_id": "c1", "output": "Paused"},
            {"id": "rs_2", "type": "reasoning", "summary": []},
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Here is what has happened since your last turn"}],
            },
        ],
    }


def anthropic_req():
    return {
        "model": "claude-opus-4-7",
        "max_tokens": 20000,
        "betas": ["computer-use-2025-11-24"],
        "thinking": {"type": "adaptive", "display": "summarized"},
        "cache_control": {"type": "ephemeral"},
        "tools": [{"type": "bash_20250124", "name": "bash"}],
        "messages": [
            {"role": "user", "content": SYSTEM.replace("GPT-5.6 Terra", "Claude Opus 4.7")},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "hm", "signature": "s"},
                    {"type": "tool_use", "id": "t1", "name": "pause", "input": {"seconds": 400}},
                ],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
        ],
    }


def chat_req():
    return {"model": "grok-4.5", "tools": [], "messages": [{"role": "system", "content": SYSTEM}]}


def gemini_req():
    return {"model": "gemini-3.1-pro-preview", "config": {"system_instruction": SYSTEM, "tools": [{}]}, "contents": []}


def test_provider_and_kind():
    assert ex.provider_of(openai_req()) == "openai"
    assert ex.provider_of(anthropic_req()) == "anthropic"
    assert ex.provider_of(chat_req()) == "chat"
    assert ex.provider_of(gemini_req()) == "gemini"
    assert ex.kind_of(openai_req()) == "turn"
    compaction = {
        "model": "gpt-5.6-terra",
        "input": [{"role": "user", "content": SYSTEM + "Your memory is getting too large."}],
    }
    assert ex.kind_of(compaction) == "compaction"


@pytest.mark.parametrize("make", [openai_req, anthropic_req, chat_req, gemini_req])
def test_memory_found_in_every_format(make):
    assert ex.get_memory(make()).strip().startswith("# Terra memory")


def test_delete_section_stops_at_same_or_higher_heading():
    out = ex.delete_section(MEMORY, "Current decision")
    assert "qualifying trigger" not in out and "## Analytics" in out and "## Mission" in out
    with pytest.raises(ValueError):
        ex.delete_section(MEMORY, "no such heading")


def test_apply_edits_changes_only_the_copy():
    req = openai_req()
    edits = [
        {"op": "delete_section", "heading": "Current decision"},
        {"op": "replace", "old": "20 views.", "new": "25 views."},
        {"op": "set_intention", "text": "Grow the channel."},
    ]
    out, log = ex.apply_edits(req, edits)
    assert "qualifying trigger" in ex.get_memory(req)  # original untouched
    mem = ex.get_memory(out)
    assert "qualifying trigger" not in mem and "25 views." in mem
    system = out["input"][0]["content"][0]["text"]
    assert "session was: Grow the channel." in system and "genuine trigger" not in system
    assert log[2]["before"] == "Wait for a genuine trigger."
    assert out["input"][1:] == req["input"][1:]  # nothing but the system prompt changed


def test_replace_must_match_exactly_once():
    with pytest.raises(ValueError, match="occurs 0 times"):
        ex.apply_edits(openai_req(), [{"op": "replace", "old": "absent", "new": "x"}])


@pytest.mark.parametrize(
    "make,where",
    [
        (openai_req, lambda r: r["tools"][-1]["name"]),
        (anthropic_req, lambda r: r["tools"][-1]["name"]),
        (chat_req, lambda r: r["tools"][-1]["function"]["name"]),
        (gemini_req, lambda r: r["config"]["tools"][0]["function_declarations"][-1]["name"]),
    ],
)
def test_add_tool_in_each_format_and_lists_it(make, where):
    tool = {"name": "review_rules", "description": "List your standing rules.", "parameters": {"type": "object"}}
    out, log = ex.apply_edits(make(), [{"op": "add_tool", "tool": tool}])
    assert where(out) == "review_rules"
    assert log[0]["listed_in_prompt"]
    prompt = ex._get(out, ex._find(out, "</tools>"))
    assert "- review_rules: List your standing rules.\n</tools>" in prompt


def test_append_user_keeps_anthropic_alternation():
    out, _ = ex.apply_edits(anthropic_req(), [{"op": "append_user", "text": "What will you do next?"}])
    last = out["messages"][-1]
    assert last["role"] == "user" and last["content"][-1] == {"type": "text", "text": "What will you do next?"}
    assert len(out["messages"]) == 3


def test_drop_thinking():
    out, log = ex.apply_edits(anthropic_req(), [{"op": "drop_thinking"}])
    assert log[0]["dropped"] == 1
    assert all(b["type"] != "thinking" for b in out["messages"][1]["content"])


def test_restore_reasoning_fills_known_and_drops_unknown():
    by_id = {"rs_1": {"id": "rs_1", "type": "reasoning", "encrypted_content": "gAAAA"}}
    out, stats = ex.restore_reasoning(openai_req(), by_id)
    assert stats == {"reasoning_restored": 1, "reasoning_dropped": 1}
    items = {i.get("id"): i for i in out["input"]}
    assert items["rs_1"]["encrypted_content"] == "gAAAA" and "rs_2" not in items
    assert out["store"] is False and "reasoning.encrypted_content" in out["include"]


def test_parse_output_each_provider():
    oa = [
        {"type": "reasoning", "summary": [{"text": "think"}]},
        {"type": "function_call", "name": "pause", "arguments": '{"seconds": 600}'},
    ]
    p = ex.parse_output("openai", oa)
    assert [ex.action_label(a) for a in p["actions"]] == ["pause(600)"] and p["thinking"] == "think"
    an = {
        "stop_reason": "tool_use",
        "content": [
            {"type": "text", "text": "hi"},
            {"type": "tool_use", "name": "computer", "input": {"action": "left_click"}},
        ],
    }
    p = ex.parse_output("anthropic", an)
    assert [ex.action_label(a) for a in p["actions"]] == ["computer.left_click"] and p["text"] == "hi"
    ch = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": "",
                    "tool_calls": [{"function": {"name": "send_message_to_chat", "arguments": '{"message": "x"}'}}],
                },
            }
        ]
    }
    assert ex.parse_output("chat", ch)["actions"][0]["args"] == {"message": "x"}
    ge = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"thought": True, "text": "t"},
                        {"functionCall": {"name": "pause", "args": {"seconds": 5}}},
                    ]
                }
            }
        ]
    }
    p = ex.parse_output("gemini", ge)
    assert p["thinking"] == "t" and ex.action_label(p["actions"][0]) == "pause(5)"


def test_prepare_routes_bare_chat_ids_and_hashes():
    call = {
        "call_id": "c",
        "agent_id": "a",
        "agent": "Grok 4.5",
        "created_at": "2026-07-13T22:14:08",
        "request": chat_req(),
        "output": {"choices": [{"message": {"content": "hello"}}]},
    }
    pp = ex.prepare(call)
    assert pp["request"]["model"] == "x-ai/grok-4.5"
    assert pp["notes"] == [{"model_routed": "grok-4.5 sent as x-ai/grok-4.5 via OpenRouter"}]
    assert pp["sha256_original"] != pp["sha256_sent"] and pp["actual"]["text"] == "hello"


def test_split_known_routes_unknown_params_to_extra_body():
    def create(*, model, input, tools=None):  # noqa: A002 - mimics the SDK signature
        pass

    known, extra = ex._split_known(create, {"model": "m", "input": [], "prompt_cache_options": {"mode": "x"}})
    assert known == {"model": "m", "input": []} and extra == {"prompt_cache_options": {"mode": "x"}}


def test_approx_tokens_skip_image_data():
    png = "iVBORw0KGgo" + "A" * 5000
    req = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "x" * 400}, {"type": "image", "source": {"data": png}}],
            }
        ]
    }
    text = 400 + len("user") + len("text") + len("image")  # every non-image string counts as text
    assert ex.approx_input_tokens(req) == text // 4 + ex.IMAGE_TOKENS_APPROX


def test_estimate_uses_logged_output_and_price(monkeypatch):
    monkeypatch.setattr(ex, "count_input_tokens", lambda pp: (63424, "exact"))
    pp = {"provider": "anthropic", "model": "claude-opus-4-7", "request": {}, "logged_output_tokens": 143}
    est = ex.estimate(pp, n=3)
    assert not est["prompt_cache"]
    assert est["usd_first_sample"] == pytest.approx(63424 * 5 / 1e6 + 143 * 25 / 1e6)
    assert est["usd_total"] == pytest.approx(3 * est["usd_first_sample"])
    cached = ex.estimate({**pp, "request": {"cache_control": {"type": "ephemeral"}}}, n=3)
    assert cached["usd_first_sample"] == pytest.approx(63424 * 5 * 1.25 / 1e6 + 143 * 25 / 1e6)
    assert cached["usd_later_sample"] == pytest.approx(63424 * 5 * 0.1 / 1e6 + 143 * 25 / 1e6)
    unpriced = ex.estimate({**pp, "model": "gpt-5.6-terra", "logged_output_tokens": None})
    assert unpriced["usd_total"] is None and unpriced["expected_output_tokens"] is None


def test_price_override(monkeypatch):
    monkeypatch.setenv("VILLAGE_PRICES", '{"gpt-5.6-terra": [1.25, 10]}')
    assert ex.price_for("gpt-5.6-terra") == (1.25, 10)


def test_parse_logged_shapes():
    # chat-format outputs were logged as the bare assistant message
    bare = {
        "role": "assistant",
        "content": "ok",
        "tool_calls": [{"type": "function", "function": {"name": "pause", "arguments": '{"seconds":3600}'}}],
    }
    assert [ex.action_label(a) for a in ex.parse_output("chat", bare)["actions"]] == ["pause(3600)"]
    # OpenAI computer calls with a list of actions
    cc = [{"type": "computer_call", "actions": [{"type": "click", "x": 1, "y": 2, "button": "left"}]}]
    assert [ex.action_label(a) for a in ex.parse_output("openai", cc)["actions"]] == ["computer.click"]
