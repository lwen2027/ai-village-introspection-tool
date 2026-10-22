"""exact_batch.py — run exact replays for a set of interview points (village-drift-explorer's
data/interview_points_v1.json), in parallel, under a hard budget, resumably.

    village-exact batch POINTS.json --rules before_onset --question "..." -n 3 --dry-run
    village-exact batch POINTS.json --rules before_onset --question "..." -n 3 --budget-usd 10
    village-exact batch POINTS.json --agents "GPT-5.6 Terra,Claude Opus 4.7" --rules before_onset \
        --questions variants.json -n 5 --dry-run

--questions takes several phrasings of a question (a JSON list of {"id", "text"}, a {id: text} object, or
a list of strings, numbered v1, v2, …); every point is asked every variant, each as its own appended
user message on the same logged request, so the answers differ only in the wording.

Planning (always first, no model calls): every selected point's logged request is loaded and prepared
(the question appended, if any), its input counted (free count_tokens for Claude, approximated otherwise)
and priced (Anthropic list prices built in; other models from OpenRouter's public catalog). A live run
refuses to start if the planned total is over --budget-usd, and stops sending once the spend reported
by the providers reaches it.

Each point's samples run one after another, so Claude's prompt cache is written by the first and read
by the rest; different points run in parallel (--workers). Rate limits and server errors are retried
with backoff; a quota error stops the whole run. Every sample is one line in the output JSONL, and a
rerun with the same output file skips samples that already succeeded.
"""

from __future__ import annotations

import collections
import concurrent.futures
import datetime
import hashlib
import json
import os
import re
import shlex
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from . import exact_lib as ex
from .config import STATE_DIR
from .probe_jobs import _append_line

SENDABLE = {"anthropic", "openai", "chat", "moonshot"}
RETRY_DELAYS = (5, 15, 45, 120)  # seconds before each retry of a rate-limited or failed request
OPENROUTER_API = "https://openrouter.ai/api/v1"


# --------------------------------------------------------------------------------------------------
# selection and prices
# --------------------------------------------------------------------------------------------------
def point_key(agent: str, p: dict) -> str:
    return f"{agent}|{p.get('track', 0)}|{p['rule']}|{p['call_id']}"


def select_points(data: dict, rules=None, agents=None, limit: Optional[int] = None) -> list:
    """[(agent, point)] from an interview-points file, in file order, filtered by rule and agent name."""
    out = []
    for a in data["agents"]:
        if agents and a["agent"] not in agents:
            continue
        for p in a["points"]:
            if rules and p["rule"] not in rules:
                continue
            if p.get("call_id"):
                out.append((a["agent"], p))
    return out[:limit] if limit else out


def question_id(question: Optional[str], given: Optional[str] = None) -> str:
    if given:
        return given
    if not question:
        return "replay"
    return "q-" + hashlib.sha256(question.encode("utf-8")).hexdigest()[:8]


def load_variants(path) -> list:
    """[(question id, text)] from a variants file; see the module docstring for the accepted shapes."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        out = list(raw.items())
    else:
        out = []
        for i, v in enumerate(raw, 1):
            out.append((f"v{i}", v) if isinstance(v, str) else (v.get("id") or f"v{i}", v["text"]))
    ids = [q for q, _ in out]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate question ids in {path}: {ids}")
    if not all(isinstance(t, str) and t.strip() for _, t in out):
        raise ValueError(f"every variant in {path} needs non-empty text")
    return out


def openrouter_prices(models) -> dict:
    """{model: ($ in, $ out, $ cache read) per Mtok} from OpenRouter's public catalog, for the models
    without a built-in price. A logged id is matched as is, or as the last part of a catalog id
    (gpt-5.5 → openai/gpt-5.5), with or without a release date; variants such as :batch are never used."""
    want = {m for m in models if m and not ex.price_for(m)}
    if not want:
        return {}
    import openai  # its HTTP client brings its own certificates, unlike urllib on python.org builds

    client = openai.OpenAI(base_url=OPENROUTER_API, api_key=os.environ.get("OPENROUTER_API_KEY") or "none")
    catalog = [m.to_dict() for m in client.models.list()]
    by_id = {m["id"]: m["pricing"] for m in catalog if ":" not in m["id"] and m.get("pricing")}
    out = {}
    for model in want:
        bare = ex.undated(model)
        pricing = by_id.get(model) or next((v for k, v in by_id.items() if k.split("/", 1)[-1] in (model, bare)), None)
        if not pricing:
            continue
        try:
            p_in, p_out = float(pricing["prompt"]) * 1e6, float(pricing["completion"]) * 1e6
        except (KeyError, TypeError, ValueError):
            continue
        read = pricing.get("input_cache_read")
        out[model] = (p_in, p_out, float(read) * 1e6) if read else (p_in, p_out)
    return out


# --------------------------------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------------------------------
def question_op(text: str, call: dict, deliver: str, sender: Optional[str], mention: bool) -> dict:
    """The edit op that asks `text` at a logged call: as a human's chat message in the turn's feed
    (deliver='user_talk', timed at the call and, with `mention`, @-addressed to the agent, as humans in the
    village write), or as a bare user message (deliver='append')."""
    if deliver == "append":
        return {"op": "append_user", "text": text}
    if not sender:
        raise ValueError(f"delivering a question as {deliver} needs a sender name")
    body = f"@{call['agent']} {text}" if mention and call.get("agent") else text
    actor = "agent" if deliver == "agent_talk" else "human"
    return {"op": "user_talk", "text": body, "sender": sender, "at": ex.feed_time(call["created_at"]), "actor": actor}


# --------------------------------------------------------------------------------------------------
# continuing past an agent's own pre-send check (Claude Sonnet 4.6's memory rule L1)
# --------------------------------------------------------------------------------------------------
PRE_SEND_SCRIPT = "pre_send_chat.sh"
PRE_SEND_MARK = "=== PRE-SEND CHAT CHECK ==="


def pre_send_template(conn, agent_id: str, before: str) -> Optional[str]:
    """The latest real output of the agent's pre-send check logged before a call (as the harness wrote
    it, escaped inside 'Output: "..."'), from the check onwards; None if it never ran in the 2 days before."""
    rows = conn.execute(
        "select input from llm_calls where agent_id = %s and created_at between %s::timestamp - interval '2 days' and %s "
        "and position(%s in input::text) > 0 order by created_at desc limit 1",
        (agent_id, before, before, PRE_SEND_MARK),
    ).fetchall()
    for (inp,) in rows:
        for m in reversed(inp.get("messages") or []):
            for b in m.get("content") if isinstance(m.get("content"), list) else []:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    c = b.get("content")
                    t = c if isinstance(c, str) else "".join(x.get("text", "") for x in c or [] if isinstance(x, dict))
                    if PRE_SEND_MARK in t:
                        return t[t.index(PRE_SEND_MARK) :]
    return None


def pre_send_call(actions: list) -> Optional[str]:
    """The bash command, if the response's only action is running the pre-send check."""
    if len(actions) == 1 and actions[0].get("tool") == "bash":
        cmd = (actions[0].get("args") or {}).get("command") or ""
        if PRE_SEND_SCRIPT in cmd:
            return cmd
    return None


def pre_send_output(template: str, command: str) -> str:
    """The check's output for this command: the real logged output with its Topic/Recipient line set to
    the arguments the agent passed now."""
    line = next(l for l in command.splitlines() if PRE_SEND_SCRIPT in l)
    try:
        args = shlex.split(line.split(PRE_SEND_SCRIPT, 1)[1])
    except ValueError:
        args = []
    esc = lambda v: v.replace("\\", "\\\\").replace('"', '\\"')
    topic, recipient = (args + ["", ""])[:2]
    body = re.sub(r"Topic: .*? \| Recipient: [^\\]*", lambda _: f"Topic: {esc(topic)} | Recipient: {esc(recipient)}", template, count=1)
    return 'Output: "' + body if not body.startswith("Output:") else body


def plan(
    conn,
    points: list,
    variants: list,
    n: int,
    assume_output: int,
    edits: Optional[list] = None,
    deliver: str = "user_talk",
    sender: Optional[str] = None,
    mention: bool = True,
    continue_pre_send: bool = False,
) -> list:
    """One entry per (point, question variant): {key, agent, point, question_id, question, edits,
    prepared?, estimate?, skip?}. `variants` is [(question id, text or None for a plain replay)]; `edits`
    are applied before the question, which is delivered per question_op. No model calls."""
    items = []
    for agent, p in points:
        try:
            call = ex.load_call(conn, p["call_id"])
            err = None
        except Exception as e:  # noqa: BLE001 - reported in the plan, the rest still runs
            call, err = None, f"could not load: {type(e).__name__}: {str(e)[:200]}"
        template = None
        if continue_pre_send and call and ex.provider_of(call["request"]) == "anthropic":
            template = pre_send_template(conn, call["agent_id"], call["created_at"])
        for qid, text in variants:
            item = {"key": point_key(agent, p), "agent": agent, "point": p, "question_id": qid, "question": text}
            items.append(item)
            if err:
                item["skip"] = err
                continue
            try:
                ops = list(edits or []) + ([question_op(text, call, deliver, sender, mention)] if text else [])
            except ValueError as e:
                item["skip"] = str(e)
                continue
            item["edits"] = ops
            if template:
                item["pre_send_template"] = template
            try:
                pp = ex.prepare(call, ops, conn=conn)
            except Exception as e:  # noqa: BLE001
                item["skip"] = f"could not prepare: {type(e).__name__}: {str(e)[:200]}"
                continue
            item["prepared"] = pp
            if pp["provider"] not in SENDABLE:
                item["skip"] = f"sending {pp['provider']} requests is not supported yet"
            elif pp["kind"] != "turn":
                item["skip"] = f"logged call is a {pp['kind']} step, not a turn"
    ex.EXTRA_PRICES.update(openrouter_prices({it["prepared"]["model"] for it in items if "prepared" in it}))
    for it in items:
        if "prepared" in it and "skip" not in it:
            it["estimate"] = ex.estimate(it["prepared"], n, assume_output=assume_output)
            if it["estimate"]["usd_total"] is None:
                it["skip"] = f"no price known for {it['prepared']['model']} (set VILLAGE_PRICES)"
    return items


def print_plan(items: list, n: int, out=None) -> float:
    out = out or sys.stdout
    total = 0.0
    for it in items:
        p = it["point"]
        head = f"{it['agent'][:20]:20} {p['rule'][:14]:14} {p.get('call_at_pt', '')[4:20]:16} {it.get('question_id', '')[:10]:10}"
        if "skip" in it:
            print(f"{head} SKIP  {it['skip']}", file=out)
            continue
        est, pp = it["estimate"], it["prepared"]
        total += est["usd_total"]
        cache = " cached" if est["prompt_cache"] else ""
        print(
            f"{head} {pp['model'][:22]:22} {est['input_tokens']:>8,} in ({est['input_count']}{cache})"
            f"  ${est['usd_total']:.3f} for {n}",
            file=out,
        )
    sendable = sum("skip" not in it for it in items)
    print(f"\n{sendable} of {len(items)} point × question runs sendable · {sendable * n} samples · planned ≈ ${total:.2f}", file=out)
    return total


# --------------------------------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------------------------------
def done_samples(path: Path) -> set:
    """(point key, question id, sample index) already answered without error in an output file."""
    done = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if not r.get("error"):
            done.add((r["key"], r["question_id"], r["sample"]))
    return done


PRE_SEND_SCRIPT = "pre_send_chat.sh"
LEAD_IN_CHARS = 200  # text shorter than this before a pre-send check is a lead-in ("Let me run my check…")


def is_answered(rec: dict) -> bool:
    """Whether a sample answered: text output, or a chat message with content (an action alone, such as a
    pause, is not an answer). A short lead-in followed by the pre-send chat check is not an answer either."""
    if rec.get("error"):
        return False
    calls = rec.get("tool_calls") or []
    for c in calls:
        if c.get("tool") == "send_message_to_chat" and str((c.get("args") or {}).get("message") or "").strip():
            return True
    text = (rec.get("text") or "").strip()
    if any(PRE_SEND_SCRIPT in json.dumps(c.get("args") or {}) for c in calls):
        return len(text) >= LEAD_IN_CHARS
    return bool(text)


def answered_counts(path: Path) -> dict:
    """{(point key, question id): answered samples} already in an output file."""
    counts = collections.Counter()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if is_answered(r):
                counts[(r["key"], r["question_id"])] += 1
    return counts


def _retryable(e: Exception) -> bool:
    msg = str(e).lower()
    if "insufficient_quota" in msg or "credit" in msg and "402" in msg:
        return False
    status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
    if status in (408, 409, 429, 500, 502, 503, 504, 529):
        return True
    return type(e).__name__ in ("APIConnectionError", "APITimeoutError", "RemoteProtocolError", "ReadTimeout")


def _fatal(e: Exception) -> bool:
    """Errors that will fail every later request too (no credit, bad key)."""
    msg = str(e).lower()
    status = getattr(e, "status_code", None)
    return "insufficient_quota" in msg or status in (401, 402, 403)


class Run:
    def __init__(self, out_path: Path, budget: float, spent: float = 0.0):
        self.out_path, self.budget, self.spent = out_path, budget, spent
        self.lock = threading.Lock()
        self.stop: Optional[str] = None

    def charge(self, usd: Optional[float]) -> None:
        with self.lock:
            self.spent += usd or 0.0
            if self.spent >= self.budget and not self.stop:
                self.stop = f"budget reached (${self.spent:.2f} of ${self.budget:.2f})"

    def write(self, rec: dict) -> None:
        with self.lock:
            _append_line(self.out_path, rec)


def _send_retrying(run: "Run", pp: dict) -> dict:
    """One sample, retried on rate limits and server errors; a failure comes back as {"error": ...}."""
    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            return ex.send(pp)
        except Exception as e:  # noqa: BLE001 - retried, or recorded with the sample
            if _fatal(e):
                run.stop = f"{type(e).__name__}: {str(e)[:200]}"
            if attempt < len(RETRY_DELAYS) and _retryable(e) and not run.stop:
                time.sleep(RETRY_DELAYS[attempt])
                continue
            return {"error": f"{type(e).__name__}: {str(e)[:500]}"}
    return {"error": "no attempt made"}


def run_point(
    run: Run,
    it: dict,
    meta: dict,
    n: int,
    done: set,
    keep_raw: bool,
    log,
    min_answers: Optional[int] = None,
    max_attempts: Optional[int] = None,
    answered_before: int = 0,
) -> list:
    """n samples; or, with min_answers, samples until that many answered (is_answered) or max_attempts
    (default 2 x min_answers) samples have been drawn, counting what an earlier run already wrote."""
    pp, p = it["prepared"], it["point"]
    meta = {**meta, **{k: it[k] for k in ("question_id", "question", "edits") if k in it}}
    recs = []
    answered = answered_before
    limit = (max_attempts or 2 * min_answers) if min_answers else n
    for i in range(limit):
        if min_answers and answered >= min_answers:
            break
        if (it["key"], meta["question_id"], i) in done:
            continue
        if run.stop:
            break
        rec = {
            **meta,
            "key": it["key"],
            "agent": it["agent"],
            "track": p.get("track", 0),
            "rule": p["rule"],
            "category": p.get("category"),
            "episode_id": p.get("episode_id"),
            "call_id": p["call_id"],
            "moment_pt": p.get("moment_pt"),
            "logged_action": p.get("call_action"),
            "sample": i,
            "model": pp["model"],
            "provider": pp["provider"],
            "sha256_sent": pp["sha256_sent"],
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        }
        s = _send_retrying(run, pp)
        first_usd = None
        cmd = pre_send_call(s.get("actions", [])) if it.get("pre_send_template") and "error" not in s else None
        if cmd:  # the agent ran its own pre-send check: give it the check's output and take its next turn
            output = pre_send_output(it["pre_send_template"], cmd)
            first_usd = ex.usage_usd(pp["provider"], pp["model"], s.get("usage"))
            rec["turn1"] = {
                "text": s.get("text"),
                "thinking": s.get("thinking"),
                "tool_calls": s.get("actions", []),
                "usage": s.get("usage"),
                "usd": first_usd,
            }
            rec["continued"] = {"after": PRE_SEND_SCRIPT, "tool_output": output}
            try:
                req2 = ex.continue_after_tool(pp["request"], s["raw"], output)
            except (KeyError, ValueError) as e:
                s = {"error": f"could not continue: {e}"}
            else:
                s = _send_retrying(run, {**pp, "request": req2})
        raw = s.get("raw") if isinstance(s.get("raw"), dict) else {}
        if raw.get("provider"):  # OpenRouter: the upstream provider that actually served the sample
            rec["served_by"] = raw["provider"]
        if raw.get("model") and raw["model"] != pp["model"]:
            rec["served_model"] = raw["model"]
        if not keep_raw:
            s.pop("raw", None)
        rec.update(s)
        rec["tool_calls"] = s.get("actions", [])  # with their arguments: a chat message's text lives here
        rec["actions"] = [ex.action_label(a) for a in s.get("actions", [])]
        rec["usd"] = ex.usage_usd(pp["provider"], pp["model"], s.get("usage"))
        if first_usd is not None:
            rec["usd"] = (rec["usd"] or 0.0) + first_usd
        rec["answered"] = is_answered(rec)
        answered += rec["answered"]
        run.charge(rec["usd"])
        run.write(rec)
        recs.append(rec)
        what = rec.get("error") or (", ".join(rec["actions"]) or "(text only)") + ("" if rec["answered"] else " · NO ANSWER")
        cost = f"${rec['usd']:.3f}" if rec["usd"] is not None else "$?"
        log(f"{it['agent'][:20]:20} {meta['question_id'][:10]:10} #{i + 1}  {what[:80]}  {cost} · total ${run.spent:.2f}")
    return recs


def execute(
    items: list,
    meta: dict,
    n: int,
    out_path: Path,
    budget: float,
    workers: int,
    keep_raw: bool,
    log=print,
    min_answers: Optional[int] = None,
    max_attempts: Optional[int] = None,
):
    """Send the plan's samples (see run_point for min_answers). Returns (records written this run, Run)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = done_samples(out_path)
    before = answered_counts(out_path) if min_answers else {}
    run = Run(out_path, budget)
    todo = [it for it in items if "skip" not in it]
    recs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = [
            pool.submit(
                run_point, run, it, meta, n, done, keep_raw, log, min_answers, max_attempts,
                before.get((it["key"], it.get("question_id")), 0),
            )
            for it in todo
        ]
        for f in concurrent.futures.as_completed(futs):
            recs.extend(f.result())
    return recs, run


def summarize(path: Path, batch: Optional[str] = None, out=None) -> None:
    """Per point and question variant: the sampled actions against the logged one, from every record in
    the output file (only batch `batch`, if given)."""
    out = out or sys.stdout
    by_key = collections.OrderedDict()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if batch is None or r.get("batch") == batch:
            by_key.setdefault((r["key"], r["question_id"]), []).append(r)
    total = 0.0
    for rs in by_key.values():
        ok = [r for r in rs if not r.get("error")]
        unanswered = sum(1 for r in ok if not is_answered(r))
        total += sum(r.get("usd") or 0 for r in rs)
        tally = collections.Counter(", ".join(r["actions"]) or "(text only)" for r in ok)
        errs = len(rs) - len(ok)
        r0 = rs[0]
        print(
            f"{r0['agent'][:20]:20} {r0['rule'][:14]:14} {r0['question_id'][:10]:10} logged {str(r0.get('logged_action'))[:22]:22} → "
            + ("; ".join(f"{k} ×{v}" for k, v in tally.most_common()) or "-")
            + (f"  ({errs} errors)" if errs else "")
            + (f"  ({unanswered} without an answer)" if unanswered else ""),
            file=out,
        )
    print(f"\n{len(by_key)} point × question runs · ${total:.2f} spent in total", file=out)
