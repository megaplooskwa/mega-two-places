"""Worker for the MEGA.dev challenge "two places" (S02E04).

One program, two jobs, two places:
  job1  incident-lookup   search -> read -> redact -> submit   (runs in CI, elsewhere)
  job2  voice-to-memo     transcribe -> memo_write* -> memo_search -> submit   (runs here)

The worker speaks Streamable HTTP MCP to the job's own link, follows the job's own
skill, and hands the job's result in. The model is consulted only where judgement is
needed; every value that reaches `submit` is validated in code first, so a bad model
answer aborts the run instead of handing in a wrong result.

Usage:
  python worker.py --job job1 --links job-links.env --skill skills/job1/SKILL.md \
                   --transcript transcript-job1.json [--clip URL] [--tokened N] \
                   [--dry-run] [--stop-before-submit]
Environment:
  OPENROUTER_API_KEY   optional; without it the worker runs its deterministic path
  OPENROUTER_MODEL     default deepseek/deepseek-v4.1-flash
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import pathlib
import re
import ssl
import sys
import time
import urllib.parse

UA = "mega-worker/1.0"          # the same honest name in both places
FIELDS = ["date", "time", "clock", "line", "cause"]
MODEL_DEFAULT = "deepseek/deepseek-v4.1-flash"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# --------------------------------------------------------------------------- helpers

def mask(value: str) -> str:
    """Mask anything token-shaped before it is printed."""
    if not isinstance(value, str):
        return value
    out = re.sub(r"(eyJ[A-Za-z0-9_\-\.]{10,})", "<token>", value)
    return re.sub(r"(sk-or-v1-[A-Za-z0-9\-]{4})[A-Za-z0-9\-]+", r"\1<masked>", out)


def deep_find(obj, keys):
    """Yield every value under any of `keys` anywhere in a nested structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys:
                yield v
            yield from deep_find(v, keys)
    elif isinstance(obj, list):
        for v in obj:
            yield from deep_find(v, keys)


def deep_text(obj) -> str:
    parts = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("text", "content") and isinstance(v, str):
                parts.append(v)
            else:
                parts.append(deep_text(v))
    elif isinstance(obj, list):
        parts = [deep_text(v) for v in obj]
    return "\n".join(p for p in parts if p)


def load_jsonish(text: str):
    """Parse JSON out of a tool result: bare JSON, an fenced block, or an SSE body."""
    if not isinstance(text, str):
        return None
    candidates = [text.strip()]
    candidates += re.findall(r"```(?:json)?\s*(.+?)```", text, re.S)
    start, depth = None, 0
    for i, ch in enumerate(text):
        if ch in "{[":
            if depth == 0:
                start = i
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start:i + 1])
                start = None
    for c in candidates:
        try:
            return json.loads(c)
        except Exception:
            continue
    return None


# --------------------------------------------------------------------------- MCP client

class Mcp:
    """Minimal Streamable-HTTP MCP client (initialize, tools, resources)."""

    def __init__(self, url: str, transcript: list):
        u = urllib.parse.urlsplit(url)
        self.host = u.netloc
        self.path = u.path + ("?" + u.query if u.query else "")
        self.https = u.scheme == "https"
        self.session = None
        self.rid = 0
        self.transcript = transcript
        self.protocol_version = None
        self.ctx = ssl.create_default_context()

    def _post(self, payload: dict, timeout: float = 90.0):
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": UA,
            "Content-Length": str(len(body)),
        }
        if self.session:
            headers["Mcp-Session-Id"] = self.session
        if self.protocol_version:
            headers["MCP-Protocol-Version"] = self.protocol_version
        conn = http.client.HTTPSConnection(self.host, timeout=timeout, context=self.ctx) if self.https \
            else http.client.HTTPConnection(self.host, timeout=timeout)
        conn.request("POST", self.path, body=body, headers=headers)
        resp = conn.getresponse()
        sid = resp.getheader("Mcp-Session-Id")
        if sid:
            self.session = sid
        deadline = time.time() + timeout
        buf = b""
        want = payload.get("id")
        while time.time() < deadline:
            chunk = resp.read(65536)
            if not chunk:
                break
            buf += chunk
            if want is not None:
                got = self._messages(buf.decode("utf-8", "replace"))
                if any(m.get("id") == want for m in got if isinstance(m, dict)):
                    break
            elif b"\n\n" in buf and not resp.getheader("Content-Length"):
                break
        conn.close()
        raw = buf.decode("utf-8", "replace")
        status = resp.status
        return status, raw

    @staticmethod
    def _messages(raw: str):
        out = []
        stripped = raw.strip()
        if stripped.startswith("{"):
            try:
                out.append(json.loads(stripped))
            except Exception:
                pass
        for line in raw.splitlines():
            if line.startswith("data:"):
                try:
                    out.append(json.loads(line[5:].strip()))
                except Exception:
                    pass
        return out

    def _call(self, method: str, params: dict, tag: str = "", record: bool = True):
        self.rid += 1
        payload = {"jsonrpc": "2.0", "id": self.rid, "method": method}
        if params is not None:
            payload["params"] = params
        status, raw = self._post(payload)
        msgs = self._messages(raw)
        mine = [m for m in msgs if isinstance(m, dict) and m.get("id") == self.rid]
        msg = mine[0] if mine else (msgs[-1] if msgs else None)
        if record:
            self.transcript.append({"tag": tag or method, "request": payload,
                                    "status": status, "response": msg})
        if msg is None:
            raise RuntimeError(f"{method} -> no JSON-RPC reply (HTTP {status}): {raw[:400]!r}")
        if "error" in msg:
            raise RuntimeError(f"{method} -> {msg['error']}")
        return msg.get("result")

    def initialize(self):
        for vers in ("2025-06-18", "2025-03-26", "2024-11-05"):
            self.protocol_version = None
            params = {"protocolVersion": vers,
                      "capabilities": {},
                      "clientInfo": {"name": "mega-two-places-worker", "version": "1.0"}}
            try:
                res = self._call("initialize", params, tag=f"initialize:{vers}", record=False)
            except Exception as exc:                                  # noqa: BLE001
                last = str(exc)
                continue
            self.protocol_version = vers
            self.transcript.append({"tag": "initialize", "protocol_version": vers, "result": res})
            break
        else:
            raise RuntimeError(f"initialize failed for every protocol version: {last}")  # noqa: F821
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return self.protocol_version

    def list_tools(self):
        return self._call("tools/list", {}, tag="tools/list")

    def read_resource(self, uri: str):
        return self._call("resources/read", {"uri": uri}, tag=f"resources/read:{uri}")

    def call_tool(self, name: str, arguments: dict):
        return self._call("tools/call", {"name": name, "arguments": arguments}, tag=f"tool:{name}")

    # ---- step token handling: the newest token always comes from the last call ----

    def token_of(self, result) -> str:
        for key in ("step", "stepToken", "nextStep", "next_step", "token"):
            for v in deep_find(result, {key}):
                if isinstance(v, str) and v:
                    return v
        parsed = load_jsonish(deep_text(result))
        if parsed:
            for key in ("step", "stepToken", "nextStep", "next_step", "token"):
                for v in deep_find(parsed, {key}):
                    if isinstance(v, str) and v:
                        return v
        m = re.search(r"(?:step|token)[\"'\s:]+([A-Za-z0-9_\-]{6,})", deep_text(result), re.I)
        return m.group(1) if m else ""


# --------------------------------------------------------------------------- model

class Model:
    """OpenRouter chat call. Absent key means the worker uses its deterministic path."""

    def __init__(self, key: str | None, model: str):
        self.key = key
        self.model = model

    @property
    def available(self) -> bool:
        return bool(self.key)

    def ask(self, system: str, user: str, max_tokens: int = 1500) -> str:
        if not self.key:
            raise RuntimeError("no OPENROUTER_API_KEY")
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }).encode("utf-8")
        conn = http.client.HTTPSConnection("openrouter.ai", timeout=180.0,
                                           context=ssl.create_default_context())
        conn.request("POST", "/api/v1/chat/completions", body=payload, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.key}",
            "User-Agent": UA,
            "Content-Length": str(len(payload)),
        })
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8", "replace")
        conn.close()
        if resp.status != 200:
            raise RuntimeError(f"openrouter HTTP {resp.status}: {mask(raw[:300])}")
        data = json.loads(raw)
        return data["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------- procedures

def result_summary(result) -> str:
    txt = deep_text(result)
    return txt if txt.strip() else json.dumps(result, ensure_ascii=False)[:2000]


def flat_text(obj) -> str:
    """Every string in a structure, joined: the whole record as text."""
    out = []
    if isinstance(obj, dict):
        out = [flat_text(v) for v in obj.values()]
    elif isinstance(obj, list):
        out = [flat_text(v) for v in obj]
    elif isinstance(obj, (str, int, float)):
        out.append(str(obj))
    return "\n".join(p for p in out if p)


def payload_of(result):
    """The JSON object a tool puts in its text content, when it is one."""
    parsed = load_jsonish(deep_text(result))
    return parsed if isinstance(parsed, dict) else None


def entries_from(result) -> list:
    """Lift the entry list out of a search result, whatever shape it takes."""
    parsed = load_jsonish(deep_text(result)) or result
    for key in ("entries", "results", "matches", "items", "records", "rows"):
        for v in deep_find(parsed, {key}):
            if isinstance(v, list) and v:
                return v
    for v in deep_find(parsed, {"id"}):
        if isinstance(v, list):
            return v
    if isinstance(parsed, list):
        return parsed
    return []


def entry_id(entry) -> str:
    if isinstance(entry, dict):
        for k in ("id", "entryId", "entry_id", "key"):
            if isinstance(entry.get(k), str):
                return entry[k]
    return ""


def fields_from(result) -> dict:
    parsed = load_jsonish(deep_text(result))
    src = parsed if isinstance(parsed, dict) else (result if isinstance(result, dict) else {})
    out = {}
    for f in FIELDS:
        for v in deep_find(src, {f}):
            if isinstance(v, (str, int, float)) and str(v).strip():
                out[f] = str(v).strip()
                break
    if not out:
        for line in deep_text(result).splitlines():
            m = re.match(r"^\s*([A-Za-z_]+)\s*:\s*(.+)$", line)
            if m and m.group(1).lower() in FIELDS:
                out[m.group(1).lower()] = m.group(2).strip()
    return out


def receipt_from(result) -> str:
    for key in ("receipt", "receiptId", "receipt_id", "proof"):
        for v in deep_find(result, {key}) if isinstance(result, (dict, list)) else []:
            if isinstance(v, str) and v:
                return v
    parsed = load_jsonish(deep_text(result))
    if parsed:
        for key in ("receipt", "receiptId", "receipt_id"):
            for v in deep_find(parsed, {key}):
                if isinstance(v, str) and v:
                    return v
    m = re.search(r"receipt[^A-Za-z0-9]{0,4}([A-Za-z0-9_\-\.]{6,})", deep_text(result), re.I)
    return m.group(1) if m else ""


def segments_from(result) -> list:
    parsed = load_jsonish(deep_text(result)) or result
    for key in ("segments", "transcript", "lines", "items", "data"):
        for v in deep_find(parsed, {key}):
            if isinstance(v, list) and v and isinstance(v[0], (dict, list)):
                return v
    if isinstance(parsed, list):
        return parsed
    return []


def mmss(offset) -> str:
    if isinstance(offset, str):
        s = offset.strip()
        if re.fullmatch(r"\d{1,3}:\d{2}", s):
            return s
        try:
            offset = float(s)
        except ValueError:
            return ""
    try:
        secs = float(offset)
    except (TypeError, ValueError):
        return ""
    return f"{int(secs) // 60}:{int(secs) % 60:02d}"


def segment_view(seg) -> dict:
    if isinstance(seg, dict):
        text = ""
        for k in ("text", "content", "sentence", "line"):
            if isinstance(seg.get(k), str):
                text = seg[k]
                break
        off = None
        for k in ("start", "offset", "at", "time", "ts", "begin"):
            if k in seg:
                off = seg[k]
                break
        return {"offset": mmss(off) if not isinstance(off, str) or not re.fullmatch(r"\d{1,3}:\d{2}", off.strip()) else off.strip(),
                "text": text}
    if isinstance(seg, list) and len(seg) >= 2:
        return {"offset": mmss(seg[0]), "text": str(seg[1])}
    return {"offset": "", "text": str(seg)}


def job1(mcp: Mcp, model: Model, skill: str, stop_before_submit: bool, log):
    """search -> read -> redact -> submit."""
    search_queries = ["tram-3", "tram-3 stall", "tram 3"]
    entries, used_query = [], None
    for q in search_queries:
        res = mcp.call_tool("search", {"query": q, "scope": "ops/live"})
        token = mcp.token_of(res)
        entries = entries_from(res)
        log(f"search({q!r}, ops/live) -> {len(entries)} entries; token {'yes' if token else 'NO'}")
        if entries:
            used_query = q
            break
    if not entries:
        raise RuntimeError("search returned no entries in ops/live for any query")

    # the entry has to be the one about tram-3; the model may pick, code decides
    def mentions_tram3(e) -> bool:
        return "tram-3" in flat_text(e).lower()

    candidates = [e for e in entries if mentions_tram3(e)] or entries
    order = [entry_id(e) for e in candidates if entry_id(e)]
    if model.available:
        try:
            listing = [{"id": entry_id(e), "text": flat_text(e)[:300]} for e in candidates]
            answer = model.ask(
                "You pick one entry from a list. Answer with that entry's id only, nothing else.",
                "The entry we need is tram-3's incident record from the stall night.\n"
                + json.dumps(listing, ensure_ascii=False))
            got = (load_jsonish(answer) or {}).get("id") if answer.strip().startswith("{") else None
            if not got:
                m = re.search(r"[A-Za-z][A-Za-z0-9_\-]{2,}", answer)
                got = m.group(0) if m else ""
            if got and got in order:
                order.remove(got)
                order.insert(0, got)
                log(f"model picked entry {got} (validated against the search result)")
            else:
                log(f"model pick {mask(str(got))!r} not in the search result; using the code path")
        except Exception as exc:                                     # noqa: BLE001
            log(f"model unavailable for the pick ({mask(str(exc))}); using the code path")
    if not order:
        raise RuntimeError("no usable entry id in the search result")
    log(f"entry order: {order}")

    # read candidates until one really is a tram-3 record: a wrong entry cannot be handed in
    values, pick = {}, ""
    for candidate in order[:3]:
        res = mcp.call_tool("read", {"id": candidate, "fields": FIELDS, "step": token})
        token = mcp.token_of(res)
        got = fields_from(res)
        log(f"read({candidate}) -> fields {sorted(got)}; token {'yes' if token else 'NO'}")
        missing = [f for f in FIELDS if f not in got]
        if missing:
            log(f"read({candidate}) is missing {missing}; trying the next candidate")
            continue
        if "tram-3" not in got["line"].lower():
            log(f"read({candidate}) is line {got['line']!r}, not tram-3; trying the next candidate")
            continue
        values, pick = got, candidate
        break
    if not values:
        raise RuntimeError("no candidate read back as a complete tram-3 record")

    text = "\n".join(f"{f}: {values[f]}" for f in FIELDS)
    res = mcp.call_tool("redact", {"text": text, "step": token})
    token = mcp.token_of(res)
    envelope = payload_of(res) or {}
    redacted = (envelope.get("text") or "").strip()
    if not redacted:
        log(f"redact returned no text field; full result: {mask(result_summary(res))[:800]}")
        raise RuntimeError("redact returned nothing to hand in")
    log(f"redact -> {len(redacted)} chars; token {'yes' if token else 'NO'}")

    log("--- the text that will be handed in ---")
    log(redacted)
    log("---------------------------------------")
    if stop_before_submit:
        log("stop-before-submit: not submitting")
        return {"stopped": True, "result": redacted}

    res = mcp.call_tool("submit", {"result": redacted, "skillVersion": "1", "step": token})
    receipt = receipt_from(res)
    log(f"submit -> {mask(result_summary(res))[:2000]}")
    if not receipt:
        raise RuntimeError("submit returned no receipt")
    return {"receipt": receipt, "result": redacted}


def job2(mcp: Mcp, model: Model, skill: str, clip: str, stop_before_submit: bool, log):
    """transcribe -> memo_write per finding -> memo_search -> submit."""
    if not clip:
        raise RuntimeError("job2 needs the clip url (--clip)")
    res = mcp.call_tool("transcribe", {"url": clip})
    token = mcp.token_of(res)
    segs = [segment_view(s) for s in segments_from(res)]
    log(f"transcribe -> {len(segs)} segments; token {'yes' if token else 'NO'}")
    if not segs:
        raise RuntimeError(f"transcribe returned no segments: {result_summary(res)[:300]}")
    for s in segs[:5]:
        log(f"  {s['offset']}  {s['text'][:90]}")

    if not model.available:
        raise RuntimeError("job2 needs the model to read the transcript")
    system = ("You read a transcript and list findings. A finding is anything that happened on a line: "
              "a fault, a freeze or a stall, a hold, a reset - every time it happened, even when it repeats. "
              "Talk around it is not a finding. Answer with JSON only: "
              "{\"findings\": [{\"what\": \"<what happened>\", \"offset\": \"<m:ss from the list>\"}]}. "
              "Use only offsets that appear in the list. One finding per entry, no duplicates.")
    user = ("The skill for this job:\n\n" + skill + "\n\nThe transcript, one segment per line as "
            "'<offset>\\t<text>':\n\n" + "\n".join(f"{s['offset']}\t{s['text']}" for s in segs))
    answer = model.ask(system, user, max_tokens=2000)
    parsed = load_jsonish(answer) or {}
    findings = parsed.get("findings") if isinstance(parsed, dict) else None
    if not findings:
        raise RuntimeError(f"the model returned no findings: {mask(answer)[:300]}")

    offsets = {s["offset"] for s in segs if s["offset"]}
    notes, seen = [], set()
    for f in findings:
        what = str(f.get("what", "")).strip().replace("\n", " ")
        off = str(f.get("offset", "")).strip()
        if not what or off not in offsets:
            log(f"dropped a finding with a bad offset: {mask(json.dumps(f))[:160]}")
            continue
        note = f"{what} @{off}"
        if note in seen:
            continue
        seen.add(note)
        notes.append(note)
    log(f"model findings kept: {len(notes)} of {len(findings)}")
    for n in notes:
        res = mcp.call_tool("memo_write", {"text": n, "step": token})
        token = mcp.token_of(res) or token
        log(f"memo_write -> {mask(n)[:110]}")
    if not notes:
        raise RuntimeError("no valid note to write")

    res = mcp.call_tool("memo_search", {"step": token})
    token = mcp.token_of(res) or token
    stored = result_summary(res)
    missing = [n for n in notes if n.split(" @")[0][:40] not in stored]
    log(f"memo_search -> {len(stored)} chars; notes missing on the memo: {len(missing)}")
    for n in missing:
        res = mcp.call_tool("memo_write", {"text": n, "step": token})
        token = mcp.token_of(res) or token
        log(f"memo_write (missing) -> {mask(n)[:110]}")
    if missing:
        res = mcp.call_tool("memo_search", {"step": token})
        token = mcp.token_of(res) or token
        log("memo_search again after the missing notes")

    result = "\n".join(notes)
    log("--- the notes that will be handed in ---")
    log(result)
    log("----------------------------------------")
    if stop_before_submit:
        log("stop-before-submit: not submitting")
        return {"stopped": True, "result": result}

    res = mcp.call_tool("submit", {"result": result, "skillVersion": "1", "step": token})
    receipt = receipt_from(res)
    log(f"submit -> {mask(result_summary(res))[:2000]}")
    if not receipt:
        raise RuntimeError("submit returned no receipt")
    return {"receipt": receipt, "result": result}


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", choices=["job1", "job2"], required=True)
    ap.add_argument("--links", default="job-links.env")
    ap.add_argument("--skill", default="")
    ap.add_argument("--clip", default="")
    ap.add_argument("--transcript", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stop-before-submit", action="store_true")
    args = ap.parse_args()

    lines = []
    def log(msg):
        print(mask(str(msg)), flush=True)
        lines.append(mask(str(msg)))

    links = {}
    p = pathlib.Path(args.links)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                links[k.strip()] = v.strip()
    url = links.get("JOB1_URL" if args.job == "job1" else "JOB2_URL", "") or \
        os.environ.get("JOB1_URL" if args.job == "job1" else "JOB2_URL", "")
    if not url:
        log("no job url found")
        sys.exit(3)
    clip = args.clip or os.environ.get("JOB2_CLIP", "")
    if args.job == "job2" and not clip:
        log("warning: no clip url")

    skill = pathlib.Path(args.skill).read_text(encoding="utf-8") if args.skill and pathlib.Path(args.skill).exists() else ""
    model = Model(os.environ.get("OPENROUTER_API_KEY") or None,
                  os.environ.get("OPENROUTER_MODEL", MODEL_DEFAULT))
    log(f"job={args.job} place={'ci' if os.environ.get('CI') else 'local'} "
        f"model={'on' if model.available else 'off'} skill={len(skill)}B")

    transcript = []
    mcp = Mcp(url, transcript)
    try:
        vers = mcp.initialize()
        log(f"mcp session ok, protocol {vers}")
        served = mcp.read_resource(f"skill://{'incident-lookup' if args.job == 'job1' else 'voice-to-memo'}/SKILL.md")
        served_text = deep_text(served)
        log(f"skill served by the job: {len(served_text)} chars")
        tools = mcp.list_tools()
        names = [t.get("name") for t in (tools or {}).get("tools", [])]
        log(f"tools on this link: {names}")
        if args.dry_run:
            log("dry-run: no tool called")
            out = {"dry_run": True}
        elif args.job == "job1":
            out = job1(mcp, model, served_text or skill, args.stop_before_submit, log)
        else:
            out = job2(mcp, model, served_text or skill, clip, args.stop_before_submit, log)
    except Exception as exc:                                          # noqa: BLE001
        log(f"FAILED: {type(exc).__name__}: {mask(str(exc))}")
        out = {"error": mask(str(exc))}
        code = 1
    else:
        code = 0

    if args.transcript:
        record = json.dumps({"job": args.job, "log": lines, "outcome": out,
                             "transcript": transcript}, indent=1, ensure_ascii=False)
        pathlib.Path(args.transcript).write_text(mask(record), encoding="utf-8")
    if isinstance(out, dict) and out.get("receipt"):
        print("RECEIPT " + out["receipt"], flush=True)
        if args.transcript:
            rp = pathlib.Path(args.transcript).with_suffix(".receipt.txt")
            rp.write_text(out["receipt"] + "\n", encoding="utf-8")
    sys.exit(code)


if __name__ == "__main__":
    main()
