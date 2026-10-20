"""Command line for exact replays (exact_lib). Needs VILLAGE_DB_URI; live samples need the provider's key.

    village-exact turns  --agent "GPT-5.6 Terra" --from 2026-07-13T15:00 --to 2026-07-13T15:30
    village-exact show   CALL_ID [--memory]
    village-exact replay CALL_ID [--edits edits.json] [--question "..."] [-n 5] [--dry-run]

Times without an offset are Pacific. Every replay (dry runs included) is appended to
$VILLAGE_STATE/exact_probes.jsonl with the request's sha256 before and after modification.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import sys
import uuid

from . import exact_lib as ex
from .config import STATE_DIR
from .probe_jobs import _append_line


def _cmd_turns(args) -> None:
    with ex._connect() as conn:
        for t in ex.list_turns(conn, args.agent, args.start, args.end):
            when = datetime.datetime.fromisoformat(t["created_at"]).replace(tzinfo=datetime.timezone.utc)
            pt = when.astimezone(ex.PT).strftime("%a %m-%d %H:%M:%S PT")
            acts = ", ".join(t["actions"]) or ("(memory step)" if t["kind"] == "memory" else "(no action)")
            print(f"{t['call_id']}  {pt}  {acts}")


def _cmd_show(args) -> None:
    with ex._connect() as conn:
        call = ex.load_call(conn, args.call_id)
        pp = ex.prepare(call, conn=conn)
    mem = ex.get_memory(call["request"]) or ""
    print(f"agent     {call['agent']}  ({pp['provider']}, {pp['model']})")
    print(f"call      {call['call_id']} at {call['created_at']} UTC · kind {pp['kind']}")
    print(f"request   {_fmt_estimate(ex.estimate(pp))}")
    print(f"memory    {len(mem):,} chars · notes {pp['notes']}")
    print(f"actual    {[ex.action_label(a) for a in pp['actual']['actions']]}")
    if pp["actual"]["thinking"]:
        print(f"thinking  {pp['actual']['thinking'][:600]}")
    if args.memory:
        print("\n" + mem)


def _load_edits(args) -> list:
    edits = []
    if args.edits:
        with open(args.edits, encoding="utf-8") as f:
            edits = json.load(f)
        if not isinstance(edits, list):
            raise SystemExit("--edits must be a JSON list of edit ops")
    if args.question:
        edits.append({"op": "append_user", "text": args.question})
    return edits


def _fmt_estimate(est: dict) -> str:
    out = est["expected_output_tokens"]
    parts = [
        f"{est['input_tokens']:,} input tokens ({est['input_count']})",
        f"~{out:,} output (as logged)" if out is not None else "output unknown",
    ]
    if est["usd_total"] is not None:
        if est["prompt_cache"]:
            parts.append(
                f"≈ ${est['usd_first_sample']:.2f} first sample (cache write), ${est['usd_later_sample']:.2f} each "
                f"after within 5 min · ${est['usd_total']:.2f} for {est['n']}"
            )
        else:
            parts.append(f"≈ ${est['usd_first_sample']:.2f}/sample · ${est['usd_total']:.2f} for {est['n']}")
    else:
        parts.append("no price known")
    return " · ".join(parts)


def _fmt_usage(usage, model: str) -> str:
    if not usage:
        return "usage not reported"
    tin = usage.get("input_tokens") or 0
    tout = usage.get("output_tokens") or 0
    cache_r = usage.get("cache_read_input_tokens") or 0
    cache_w = usage.get("cache_creation_input_tokens") or 0
    price = ex.price_for(model)
    cost = ""
    if price:  # cache writes at 1.25x and reads at 0.1x the input price (5-minute cache)
        usd = (tin + 1.25 * cache_w + 0.1 * cache_r) * price[0] / 1e6 + tout * price[1] / 1e6
        cost = f" · ${usd:.3f}"
    return f"{tin + cache_w + cache_r:,} in ({cache_r:,} cached) / {tout:,} out{cost}"


def _cmd_replay(args) -> None:
    edits = _load_edits(args)
    with ex._connect() as conn:
        call = ex.load_call(conn, args.call_id)
        pp = ex.prepare(call, edits, conn=conn)
    est = ex.estimate(pp, args.n)
    print(f"{pp['agent']} · {pp['model']} · {pp['provider']}")
    print("estimate " + _fmt_estimate(est))
    print(f"sha256 original {pp['sha256_original'][:16]} · sent {pp['sha256_sent'][:16]}")
    for e in pp["edits"]:
        print(f"edit     {e}")
    for n in pp["notes"]:
        print(f"note     {n}")
    print(f"actual   {[ex.action_label(a) for a in pp['actual']['actions']]}")
    rec = {
        "replay_id": uuid.uuid4().hex[:12],
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "call_id": pp["call_id"],
        "agent": pp["agent"],
        "model": pp["model"],
        "provider": pp["provider"],
        "edits": pp["edits"],
        "notes": pp["notes"],
        "sha256_original": pp["sha256_original"],
        "sha256_sent": pp["sha256_sent"],
        "actual": pp["actual"],
        "estimate": est,
        "dry_run": args.dry_run,
        "samples": [],
    }
    blocked = None
    if not args.dry_run and not args.force:
        if est["usd_total"] is None:
            blocked = f"no price is known for {pp['model']}; set VILLAGE_PRICES or pass --force"
        elif est["usd_total"] > args.max_usd:
            blocked = f"estimated ${est['usd_total']:.2f} is over --max-usd {args.max_usd:.2f}"
    if blocked:
        rec["blocked"] = blocked
        print(f"not sent: {blocked}", file=sys.stderr)
    elif not args.dry_run:
        for i in range(args.n):
            try:
                s = ex.send(pp)
            except Exception as e:  # recorded with the replay, then reported
                rec["samples"].append({"error": f"{type(e).__name__}: {str(e)[:500]}"})
                print(f"sample {i + 1}: ERROR {type(e).__name__}: {str(e)[:300]}", file=sys.stderr)
                continue
            if not args.keep_raw:
                s.pop("raw", None)
            rec["samples"].append(s)
            labels = [ex.action_label(a) for a in s["actions"]]
            print(f"sample {i + 1}: {labels}  {s['text'][:160]!r}  ({_fmt_usage(s.get('usage'), pp['model'])})")
        tally = collections.Counter(
            ", ".join(ex.action_label(a) for a in s.get("actions", [])) or "(text only)"
            for s in rec["samples"]
            if "error" not in s
        )
        if tally:
            print("tally   " + "; ".join(f"{k} ×{v}" for k, v in tally.most_common()))
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    _append_line(STATE_DIR / "exact_probes.jsonl", rec)
    print(f"logged  {rec['replay_id']} → {STATE_DIR / 'exact_probes.jsonl'}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        prog="village-exact", description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("turns", help="list an agent's logged model calls in a time window")
    t.add_argument("--agent", required=True)
    t.add_argument("--from", dest="start", required=True)
    t.add_argument("--to", dest="end", required=True)
    s = sub.add_parser("show", help="summarise one logged call")
    s.add_argument("call_id")
    s.add_argument("--memory", action="store_true", help="print the internal memory it was sent")
    r = sub.add_parser("replay", help="resend a logged call, optionally modified")
    r.add_argument("call_id")
    r.add_argument("--edits", help="JSON file with a list of edit ops (see exact_lib.apply_edits)")
    r.add_argument("--question", help="append an interview question as a user message")
    r.add_argument("-n", type=int, default=1, help="number of live samples (default 1)")
    r.add_argument("--dry-run", action="store_true", help="prepare and log, but don't call the model")
    r.add_argument("--keep-raw", action="store_true", help="keep each sample's raw response in the log")
    r.add_argument("--max-usd", type=float, default=1.0, help="don't send if the estimate is over this (default 1.00)")
    r.add_argument("--force", action="store_true", help="send even if the cost is over --max-usd or unknown")
    args = ap.parse_args(argv)
    {"turns": _cmd_turns, "show": _cmd_show, "replay": _cmd_replay}[args.cmd](args)


if __name__ == "__main__":
    main()
