"""exact_lib.py — exact replay of a village agent's turn from the request the harness actually sent.

The public export has no `llm_calls`, so tierb_lib can only approximate what an agent saw. With read
access to the AI Village database (VILLAGE_DB_URI), every turn's full model request is available:
system prompt with the agent's internal memory, tools, the event feed, its own earlier tool calls and
screenshots, and the reasoning settings. This module loads that request, optionally modifies it, and
resends it to the same model through the provider's own API.

FIDELITY. An unmodified replay sends the logged request as recorded, with these documented changes:
  - OpenAI Responses: the harness logged reasoning items without their encrypted content (it is in the
    previous calls' outputs). It is restored from those outputs by item id; an item that can't be
    restored is dropped and counted. `include` / `store` are set so encrypted reasoning round-trips.
  - Parameters the installed SDK doesn't know (e.g. `prompt_cache_options`) are sent via `extra_body`.
  - Sampling is not deterministic, so compare several samples with the logged output, never one.
A modified replay (memory edit, added tool, appended question) records each modification and the
sha256 of the original and the sent request.

Request formats, recognised by their top-level keys:
  anthropic  {model, messages, tools, thinking, betas?, cache_control?}  — Claude agents
  openai     {model, input, tools, reasoning}                            — GPT agents (Responses API)
  chat       {model, messages, tools, provider?}                         — OpenRouter-style agents
  gemini     {model, contents, config}                                   — Gemini agents (send unsupported)
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import inspect
import json
import os
import re
from typing import Optional

MEMORY_OPEN, MEMORY_CLOSE = "<internal_memory>", "</internal_memory>"
TOOLS_CLOSE = "</tools>"
INTENTION_OPEN, INTENTION_CLOSE = "<intention>", "</intention>"
SESSION_END_PREFIX = "Now that you are consolidating your session"
COMPACTION_MARK = "Your memory is getting too large"
PT = datetime.timezone(datetime.timedelta(hours=-7), "PDT")  # the goal-max period is all daylight time


# --------------------------------------------------------------------------------------------------
# request shape
# --------------------------------------------------------------------------------------------------
def provider_of(req: dict) -> Optional[str]:
    if "contents" in req:
        return "gemini"
    if "input" in req:
        return "openai"
    if "messages" in req:
        return "anthropic" if ("thinking" in req or "betas" in req or "max_tokens" in req) else "chat"
    return None


def _strings(o, path=()):
    """Yield (path, string) for every string value in a JSON-like object."""
    if isinstance(o, str):
        yield path, o
    elif isinstance(o, dict):
        for k, v in o.items():
            yield from _strings(v, path + (k,))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _strings(v, path + (i,))


def _get(o, path):
    for p in path:
        o = o[p]
    return o


def _set(o, path, value):
    for p in path[:-1]:
        o = o[p]
    o[path[-1]] = value


def _find(req: dict, needle: str) -> Optional[tuple]:
    """Path of the first string in the request that contains `needle` (the system prompt comes first)."""
    return next((p for p, s in _strings(req) if needle in s), None)


def kind_of(req: dict) -> str:
    """'turn' (a normal step, tools offered), 'session_end' (asked what to add to memory), 'compaction'
    (memory too large, rewrite it), or 'other'."""
    text = json.dumps(req, ensure_ascii=False)
    if COMPACTION_MARK in text:
        return "compaction"
    if SESSION_END_PREFIX in text[-20000:]:
        return "session_end"
    if req.get("tools") or (req.get("config") or {}).get("tools"):
        return "turn"
    return "other"


def get_memory(req: dict) -> Optional[str]:
    path = _find(req, MEMORY_OPEN)
    if path is None:
        return None
    s = _get(req, path)
    a = s.index(MEMORY_OPEN) + len(MEMORY_OPEN)
    b = s.find(MEMORY_CLOSE, a)
    return s[a:] if b < 0 else s[a:b]


def _replace_memory(req: dict, new: str) -> None:
    path = _find(req, MEMORY_OPEN)
    if path is None:
        raise ValueError("this request has no <internal_memory> block")
    s = _get(req, path)
    a = s.index(MEMORY_OPEN) + len(MEMORY_OPEN)
    b = s.find(MEMORY_CLOSE, a)
    _set(req, path, s[:a] + new + ("" if b < 0 else s[b:]))


# --------------------------------------------------------------------------------------------------
# modifications
# --------------------------------------------------------------------------------------------------
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def delete_section(text: str, heading: str) -> str:
    """Remove a markdown section whose heading line contains `heading` (case-sensitive), up to the next
    heading of the same or a higher level. Raises if no heading matches or more than one does."""
    lines = text.split("\n")
    hits = [i for i, ln in enumerate(lines) if _HEADING.match(ln.strip()) and heading in ln]
    if len(hits) != 1:
        raise ValueError(f"heading {heading!r} matched {len(hits)} sections; make it unique")
    i = hits[0]
    level = len(_HEADING.match(lines[i].strip()).group(1))
    j = i + 1
    while j < len(lines):
        m = _HEADING.match(lines[j].strip())
        if m and len(m.group(1)) <= level:
            break
        j += 1
    return "\n".join(lines[:i] + lines[j:])


def _tool_def(provider: str, spec: dict):
    """A neutral tool spec {name, description, parameters (JSON schema)} in the request's own format."""
    name, desc = spec["name"], spec.get("description", "")
    params = spec.get("parameters") or {"type": "object", "properties": {}}
    if provider == "anthropic":
        return {"name": name, "description": desc, "input_schema": params}
    if provider == "openai":
        return {"type": "function", "name": name, "description": desc, "parameters": params}
    if provider == "chat":
        return {"type": "function", "function": {"name": name, "description": desc, "parameters": params}}
    if provider == "gemini":
        return {"name": name, "description": desc, "parameters": params}
    raise ValueError(f"unknown provider {provider!r}")


def _append_user(req: dict, provider: str, text: str) -> None:
    if provider == "anthropic":
        msgs = req["messages"]
        if msgs and msgs[-1]["role"] == "user":
            c = msgs[-1]["content"]
            blocks = [{"type": "text", "text": c}] if isinstance(c, str) else list(c)
            msgs[-1]["content"] = blocks + [{"type": "text", "text": text}]
        else:
            msgs.append({"role": "user", "content": [{"type": "text", "text": text}]})
    elif provider == "openai":
        req["input"].append({"role": "user", "content": [{"type": "input_text", "text": text}]})
    elif provider == "chat":
        req["messages"].append({"role": "user", "content": text})
    elif provider == "gemini":
        req["contents"].append({"role": "user", "parts": [{"text": text}]})


def apply_edits(req: dict, edits: list) -> tuple:
    """(modified copy, log). Ops:
    {"op": "delete_section", "heading": str}      remove a section of the internal memory
    {"op": "replace", "old": str, "new": str}     replace text that occurs exactly once in the memory
    {"op": "set_memory", "text": str}             replace the whole internal memory
    {"op": "set_intention", "text": str}          replace the session intention (its last session goal)
    {"op": "add_tool", "tool": {name, description, parameters}, "doc": str?}
                                                  offer a new tool (and list it in the <tools> section)
    {"op": "append_user", "text": str}            add a user message at the end (an interview question)
    {"op": "drop_thinking"}                       drop earlier Anthropic thinking blocks (needed when an
                                                  edit would invalidate them on models that check)"""
    out = copy.deepcopy(req)
    provider = provider_of(out)
    log = []
    for e in edits:
        op = e.get("op")
        if op in ("delete_section", "replace", "set_memory"):
            mem = get_memory(out)
            if mem is None:
                raise ValueError("this request has no <internal_memory> block")
            if op == "delete_section":
                new = delete_section(mem, e["heading"])
            elif op == "replace":
                n = mem.count(e["old"])
                if n != 1:
                    raise ValueError(f"replace: the text occurs {n} times in memory; it must occur once")
                new = mem.replace(e["old"], e["new"])
            else:
                new = e["text"]
            _replace_memory(out, new)
            log.append({**e, "chars_before": len(mem), "chars_after": len(new)})
        elif op == "set_intention":
            path = _find(out, INTENTION_OPEN)
            if path is None:
                raise ValueError("this request has no <intention> block")
            s = _get(out, path)
            a = s.index(INTENTION_OPEN) + len(INTENTION_OPEN)
            b = s.index(INTENTION_CLOSE, a)
            prefix = "The intention you set at the start of this session was: "
            _set(out, path, s[:a] + "\n" + prefix + e["text"] + "\n" + s[b:])
            log.append({**e, "before": s[a:b].strip().removeprefix(prefix)})
        elif op == "add_tool":
            tool = _tool_def(provider, e["tool"])
            if provider == "gemini":
                cfg = out.setdefault("config", {})
                tools = cfg.setdefault("tools", [{}])
                tools[0].setdefault("function_declarations", []).append(tool)
            else:
                out.setdefault("tools", []).append(tool)
            doc = e.get("doc") or f"- {e['tool']['name']}: {e['tool'].get('description', '')}"
            path = _find(out, TOOLS_CLOSE)
            if path is not None:
                s = _get(out, path)
                _set(out, path, s.replace(TOOLS_CLOSE, doc.rstrip("\n") + "\n" + TOOLS_CLOSE, 1))
            log.append({"op": op, "name": e["tool"]["name"], "listed_in_prompt": path is not None})
        elif op == "append_user":
            _append_user(out, provider, e["text"])
            log.append(e)
        elif op == "drop_thinking":
            n = 0
            if provider == "anthropic":
                for m in out["messages"]:
                    if m["role"] == "assistant" and isinstance(m["content"], list):
                        kept = [b for b in m["content"] if b.get("type") not in ("thinking", "redacted_thinking")]
                        n += len(m["content"]) - len(kept)
                        m["content"] = kept
            log.append({"op": op, "dropped": n})
        else:
            raise ValueError(f"unknown edit op {op!r}")
    return out, log


def request_sha(req: dict) -> str:
    return hashlib.sha256(json.dumps(req, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


# --------------------------------------------------------------------------------------------------
# OpenAI reasoning restoration
# --------------------------------------------------------------------------------------------------
def restore_reasoning(req: dict, by_id: dict) -> tuple:
    """(copy, stats): fill each input reasoning item's encrypted_content from `by_id` (reasoning items from
    earlier outputs, keyed by id); drop items that can't be restored, since a bare id only resolves in the
    org that stored it. Also asks for encrypted reasoning back, so a follow-up turn could round-trip it."""
    out = copy.deepcopy(req)
    restored = dropped = 0
    items = []
    for it in out.get("input", []):
        if isinstance(it, dict) and it.get("type") == "reasoning" and not it.get("encrypted_content"):
            src = by_id.get(it.get("id"))
            if src and src.get("encrypted_content"):
                it = {**it, "encrypted_content": src["encrypted_content"]}
                restored += 1
            else:
                dropped += 1
                continue
        items.append(it)
    out["input"] = items
    out["store"] = False
    include = list(out.get("include") or [])
    if "reasoning.encrypted_content" not in include:
        include.append("reasoning.encrypted_content")
    out["include"] = include
    return out, {"reasoning_restored": restored, "reasoning_dropped": dropped}


# --------------------------------------------------------------------------------------------------
# reading a response (logged or live) as actions
# --------------------------------------------------------------------------------------------------
def _args(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return {"_raw": v}
    return v or {}


def parse_output(provider: str, out) -> dict:
    """{actions: [{tool, args}], text, thinking, stop} from a provider response (as a dict / list)."""
    actions, text, thinking, stop = [], [], [], None
    if provider == "anthropic":
        for b in (out or {}).get("content", []):
            t = b.get("type")
            if t == "tool_use":
                actions.append({"tool": b.get("name"), "args": b.get("input") or {}})
            elif t == "text":
                text.append(b.get("text", ""))
            elif t == "thinking":
                thinking.append(b.get("thinking", ""))
        stop = (out or {}).get("stop_reason")
    elif provider == "openai":
        items = out if isinstance(out, list) else (out or {}).get("output", [])
        for it in items:
            t = it.get("type")
            if t == "function_call":
                actions.append({"tool": it.get("name"), "args": _args(it.get("arguments"))})
            elif t == "computer_call":
                actions.append({"tool": "computer", "args": it.get("action") or {}})
            elif t == "message":
                text += [c.get("text", "") for c in it.get("content", []) if isinstance(c, dict)]
            elif t == "reasoning":
                thinking += [s.get("text", "") for s in it.get("summary", []) if isinstance(s, dict)]
    elif provider == "chat":
        ch = ((out or {}).get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            actions.append({"tool": fn.get("name"), "args": _args(fn.get("arguments"))})
        if msg.get("content"):
            text.append(msg["content"])
        r = msg.get("reasoning") or msg.get("reasoning_content")
        if r:
            thinking.append(r)
        stop = ch.get("finish_reason")
    elif provider == "gemini":
        cand = ((out or {}).get("candidates") or [{}])[0]
        for p in (cand.get("content") or {}).get("parts", []):
            if p.get("functionCall") or p.get("function_call"):
                fc = p.get("functionCall") or p.get("function_call")
                actions.append({"tool": fc.get("name"), "args": fc.get("args") or {}})
            elif p.get("thought") and p.get("text"):
                thinking.append(p["text"])
            elif p.get("text"):
                text.append(p["text"])
        stop = cand.get("finishReason") or cand.get("finish_reason")
    return {"actions": actions, "text": "\n".join(t for t in text if t), "thinking": "\n".join(thinking), "stop": stop}


def action_label(a: dict) -> str:
    """A short label for comparing actions: pause(600), send_message_to_chat, bash, computer.left_click."""
    tool, args = a.get("tool") or "?", a.get("args") or {}
    if tool == "pause":
        return f"pause({args.get('seconds')})"
    if tool == "computer" and args.get("action"):
        return f"computer.{args['action']}"
    return tool


# --------------------------------------------------------------------------------------------------
# database
# --------------------------------------------------------------------------------------------------
def _connect():
    uri = os.environ.get("VILLAGE_DB_URI")
    if not uri:
        raise RuntimeError("VILLAGE_DB_URI is not set; exact replay needs read access to the village database")
    import psycopg

    return psycopg.connect(uri)


def _utc(ts) -> datetime.datetime:
    """A timestamp as naive UTC (the database's convention). Strings without an offset are read as PT."""
    if isinstance(ts, str):
        ts = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=PT)
    return ts.astimezone(datetime.timezone.utc).replace(tzinfo=None)


def agent_ids(conn, name: str) -> list:
    rows = conn.execute("select id from agents where name = %s", (name,)).fetchall()
    if not rows:
        raise ValueError(f"no agent named {name!r}")
    return [str(r[0]) for r in rows]


def list_turns(conn, name: str, start, end) -> list:
    """The agent's model calls between start and end (PT unless an offset is given), oldest first:
    [{call_id, created_at (UTC), kind, actions}] with the actions parsed from the logged output."""
    ids = agent_ids(conn, name)
    rows = conn.execute(
        "select id, created_at, input->>'model', output, "
        "  ((input ? 'tools') or ((input->'config') ? 'tools')) as has_tools "
        "from llm_calls where agent_id = any(%s::uuid[]) and created_at between %s and %s order by created_at",
        (ids, _utc(start), _utc(end)),
    ).fetchall()
    out = []
    for cid, created, model, output, has_tools in rows:
        prov = "openai" if isinstance(output, list) else _provider_from_model(model)
        parsed = parse_output(prov, output)
        out.append(
            {
                "call_id": str(cid),
                "created_at": created.isoformat(),
                "model": model,
                "kind": "turn" if has_tools else "memory",
                "actions": [action_label(a) for a in parsed["actions"]],
                "text": parsed["text"][:200],
            }
        )
    return out


def _provider_from_model(model: str) -> str:
    m = (model or "").lower()
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("gemini"):
        return "gemini"
    if m.startswith(("gpt-", "o")):
        return "openai"
    return "chat"


def load_call(conn, call_id: str) -> dict:
    row = conn.execute(
        "select l.id, l.agent_id, a.name, l.created_at, l.input, l.output "
        "from llm_calls l left join agents a on a.id = l.agent_id where l.id = %s",
        (call_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"no llm_call {call_id}")
    cid, aid, name, created, inp, outp = row
    return {
        "call_id": str(cid),
        "agent_id": str(aid),
        "agent": name,
        "created_at": created.isoformat(),
        "request": inp,
        "output": outp,
    }


def reasoning_index(conn, agent_id: str, before: str, hours: int = 24) -> dict:
    """Reasoning items from the agent's outputs in the `hours` before a call, keyed by item id."""
    t1 = datetime.datetime.fromisoformat(before)
    rows = conn.execute(
        "select output from llm_calls where agent_id = %s and created_at between %s and %s "
        "and jsonb_typeof(output) = 'array'",
        (agent_id, t1 - datetime.timedelta(hours=hours), t1),
    ).fetchall()
    by_id = {}
    for (output,) in rows:
        for it in output or []:
            if isinstance(it, dict) and it.get("type") == "reasoning" and it.get("id"):
                by_id[it["id"]] = it
    return by_id


# --------------------------------------------------------------------------------------------------
# preparing and sending
# --------------------------------------------------------------------------------------------------
def prepare(call: dict, edits: Optional[list] = None, conn=None) -> dict:
    """The request to send for a logged call, with its fidelity notes. No model call."""
    original = call["request"]
    provider = provider_of(original)
    req, notes = copy.deepcopy(original), []
    if provider == "openai":
        by_id = reasoning_index(conn, call["agent_id"], call["created_at"]) if conn is not None else {}
        req, stats = restore_reasoning(req, by_id)
        notes.append(stats)
    elif provider == "chat" and "/" not in (req.get("model") or ""):
        mapped = OPENROUTER_IDS.get(req.get("model"))
        if mapped:
            notes.append({"model_routed": f"{req['model']} sent as {mapped} via OpenRouter"})
            req["model"] = mapped
    edit_log = []
    if edits:
        req, edit_log = apply_edits(req, edits)
    return {
        "call_id": call["call_id"],
        "agent": call.get("agent"),
        "provider": provider,
        "kind": kind_of(original),
        "model": original.get("model"),
        "request": req,
        "edits": edit_log,
        "notes": notes,
        "sha256_original": request_sha(original),
        "sha256_sent": request_sha(req),
        "approx_tokens": approx_input_tokens(req),
        "logged_output_tokens": _output_tokens(provider, call["output"]),
        "actual": parse_output(provider, call["output"]),
    }


# --------------------------------------------------------------------------------------------------
# cost estimate
# --------------------------------------------------------------------------------------------------
# $ per million tokens (input, output): Anthropic first-party list prices as of 2026-09-25, without
# cache discounts. Other providers' prices are not built in (set VILLAGE_PRICES='{"model": [in, out]}').
PRICES = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
}
# a rough allowance per screenshot where tokens can't be counted exactly (a 1024x768 PNG)
IMAGE_TOKENS_APPROX = 1600


def price_for(model: str) -> Optional[tuple]:
    extra = os.environ.get("VILLAGE_PRICES")
    if extra:
        try:
            table = json.loads(extra)
            if model in table:
                return tuple(table[model])
        except ValueError:
            pass
    return PRICES.get(model)


def _is_image_data(s: str) -> bool:
    return s.startswith("data:image/") or (len(s) > 2000 and s[:8] in ("iVBORw0K", "/9j/4AAQ"))


def approx_input_tokens(req: dict) -> int:
    """Text characters / 4 plus a fixed allowance per image; base64 image data is not counted as text."""
    text = images = 0
    for _, s in _strings(req):
        if _is_image_data(s):
            images += 1
        else:
            text += len(s)
    return text // 4 + images * IMAGE_TOKENS_APPROX


def _output_tokens(provider: str, output) -> Optional[int]:
    if provider == "anthropic" and isinstance(output, dict):
        return (output.get("usage") or {}).get("output_tokens")
    return None  # OpenAI outputs were logged as bare item lists, without usage


def count_input_tokens(prepared: dict) -> tuple:
    """(input tokens, 'exact' | 'approx'). Claude requests are counted with the free count_tokens endpoint;
    everything else (or a failed count) is approximated by approx_input_tokens."""
    if prepared["provider"] == "anthropic":
        try:
            import anthropic

            req = prepared["request"]
            kw = {k: req[k] for k in ("model", "messages", "tools", "thinking", "tool_choice", "betas") if k in req}
            return anthropic.Anthropic().beta.messages.count_tokens(**kw).input_tokens, "exact"
        except Exception:  # noqa: BLE001 - no key, network, or a field count_tokens rejects
            pass
    return approx_input_tokens(prepared["request"]), "approx"


def estimate(prepared: dict, n: int = 1) -> dict:
    """Input tokens, expected output tokens (the logged turn's own, when logged) and cost for n samples."""
    tokens, how = count_input_tokens(prepared)
    out = prepared.get("logged_output_tokens")
    price = price_for(prepared["model"])
    # A request with prompt caching on writes the cache on its first sample (1.25x the input price) and
    # reads it on later samples sent within the 5-minute TTL (0.1x); otherwise every sample pays full input.
    cached = prepared["provider"] == "anthropic" and _uses_cache(prepared["request"])
    first = later = total = None
    if price:
        out_usd = (out or 0) * price[1] / 1e6
        first = tokens * price[0] * (1.25 if cached else 1.0) / 1e6 + out_usd
        later = tokens * price[0] * (0.1 if cached else 1.0) / 1e6 + out_usd
        total = first + later * (n - 1)
    return {
        "input_tokens": tokens,
        "input_count": how,
        "expected_output_tokens": out,
        "price_per_mtok": price,
        "prompt_cache": cached,
        "usd_first_sample": first,
        "usd_later_sample": later,
        "usd_total": total,
        "n": n,
    }


def _uses_cache(req: dict) -> bool:
    return "cache_control" in req or any(p and p[-1] == "cache_control" for p, _ in _strings(req))


# Chat-format agents logged with a vendor's own id are reached through OpenRouter under its id there.
OPENROUTER_IDS = {
    "grok-4.5": "x-ai/grok-4.5",
    "kimi-k3": "moonshotai/kimi-k3",
    "kimi-k2.6": "moonshotai/kimi-k2.6",
}


def _split_known(fn, kw: dict) -> tuple:
    """(known kwargs, extra_body) for an SDK method: unknown keys go through extra_body unchanged."""
    params = inspect.signature(fn).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return kw, {}
    known = {k: v for k, v in kw.items() if k in params}
    extra = {k: v for k, v in kw.items() if k not in params}
    return known, extra


def send(prepared: dict) -> dict:
    """One live sample for a prepared request. Returns parse_output(...) plus usage and the raw response."""
    provider, req = prepared["provider"], dict(prepared["request"])
    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic()
        known, extra = _split_known(client.beta.messages.stream, req)
        if extra:
            known["extra_body"] = extra
        with client.beta.messages.stream(**known) as stream:
            msg = stream.get_final_message()
        raw = msg.to_dict()
    elif provider == "openai":
        import openai

        client = openai.OpenAI()
        known, extra = _split_known(client.responses.create, req)
        if extra:
            known["extra_body"] = extra
        resp = client.responses.create(**known).to_dict()
        parsed = parse_output(provider, resp.get("output", []))
        return {**parsed, "usage": resp.get("usage"), "raw": resp}
    elif provider == "chat":
        import openai

        client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ.get("OPENROUTER_API_KEY"))
        known, extra = _split_known(client.chat.completions.create, req)
        if extra:
            known["extra_body"] = extra
        raw = client.chat.completions.create(**known).to_dict()
    else:
        raise NotImplementedError(f"sending {provider} requests is not supported yet")
    parsed = parse_output(provider, raw)
    usage = raw.get("usage") if isinstance(raw, dict) else None
    return {**parsed, "usage": usage, "raw": raw}
