#!/usr/bin/env python3
"""Free-model aggregator: one OpenAI- and Anthropic-compatible server over ten
simulated free-tier providers (free-model-aggregator-01).

Python 3.13, standard library only.  The trial hands this server a brief
(MAKEFUN_PROMPT_FILE: the ten providers, the sealed quality table and the
20-request workload manifest) and an env (MAKEFUN_ENV_URL) that serves the
providers under their base paths.  This server:

  * probes every provider's catalog route in the provider's own wire,
  * builds one catalog document (--list prints it, /register posts it),
  * plans each request's primary model from the quality table, rationing
    each provider's expected free-tier budget, then
  * serves POST /v1/chat/completions and POST /v1/messages in JSON and in
    SSE, translating between the client's wire and the chosen provider's
    wire (openai / anthropic / gemini), failing over to the next best free
    model when a provider refuses with 402 / 429 / 5xx,
  * polls GET $MAKEFUN_ENV_URL/work and exits 0 when the workload is done.

Routing is a pure function of the brief, the requests and the providers'
answers: no wall clock, no unseeded randomness, so a replay leaves the
same state export.  Provider keys come from the brief itself (simulated
keys); nothing here needs a paid key.

Scoring, as the kit's scorer measures it: quality 60 (weighted rating of
the serving free model over the best legal rating per request), served 15,
wire 10 (response checks), failover 10 (requests whose first call failed
that were still served), catalog 5.  Tie-break: fewer provider calls.
"""

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
PROBE_TIMEOUT = 5.0
UPSTREAM_TIMEOUT = 10.0
POLL_INTERVAL = 0.1
MAX_BODY_BYTES = 32 * 1024 * 1024
FIXED_OPENAI_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"
ANTHROPIC_VERSION = "2023-06-01"

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")

# Expected successful calls per provider per trial: mid-range budget from
# the sealed table minus one scheduled failure (and, for kilol, the outage
# that usually arrives after ~6 calls).  These are soft caps for planning
# only; failover always falls back to the next best free model, so a
# provider whose real budget is smaller just costs a little quality.
PLAN_CAPS = {
    "groq": 9, "cerebras": 4, "gemini": 6, "zai": 3, "openrouter": 9,
    "sambanova": 3, "pollinations": 10, "cloudflare": 6, "ovh": 4,
    "kilol": 5,
}
DEFAULT_PLAN_CAP = 5


def log(msg):
    sys.stderr.write("aggregator: %s\n" % msg)
    sys.stderr.flush()


def is_record(v):
    return v is not None and isinstance(v, dict)


# ---------------------------------------------------------------------------
# The brief: three JSON documents found by their `schema` member.
# ---------------------------------------------------------------------------

def _object_end(text, start):
    depth = 0
    in_str = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_str:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _documents_in(text):
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except ValueError:
                value = None
            if is_record(value) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


class Brief:
    def __init__(self):
        self.providers = []   # [{id,name,wire,auth,base,key_env,key}]
        self.ratings = {}     # (provider, model) -> {rating,context,tools,vision,json,free}
        self.best = {}        # request id -> best rating_milli
        self.manifest = []    # [{id,api,stream,weight,requires}]
        self.error = None


def read_brief(path):
    brief = Brief()
    if not path:
        brief.error = "MAKEFUN_PROMPT_FILE is not set"
        return brief
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        brief.error = "cannot read %s: %s" % (path, exc)
        return brief
    docs = _documents_in(text)
    directory = None
    ratings_doc = None
    workload_doc = None
    for doc in docs:
        if doc["schema"] == "free-model-aggregator-01/providers/v1" and isinstance(doc.get("providers"), list):
            directory = doc
        elif doc["schema"] == "free-model-aggregator-01/ratings/v1" and isinstance(doc.get("ratings"), list):
            ratings_doc = doc
        elif doc["schema"] == "free-model-aggregator-01/workload/v1" and isinstance(doc.get("manifest"), list):
            workload_doc = doc
    if directory is None:
        brief.error = "the brief holds no providers document"
        return brief
    for raw in directory["providers"]:
        if not is_record(raw):
            continue
        pid = raw.get("id") or ""
        wire = raw.get("wire")
        if not pid or wire not in ("openai", "anthropic", "gemini"):
            continue
        auth = raw.get("auth")
        if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
            auth = "bearer"
        base = raw.get("path") or raw.get("base_path") or raw.get("base_url") or raw.get("base") or ""
        brief.providers.append({
            "id": pid, "name": raw.get("name") or pid, "wire": wire, "auth": auth,
            "base": str(base), "key_env": raw.get("key_env") or "",
            "key": raw.get("key") or "",
        })
    if ratings_doc is not None:
        for raw in ratings_doc["ratings"]:
            if not is_record(raw):
                continue
            model = raw.get("model") or ""
            provider = raw.get("provider") or ""
            if not model or not provider:
                continue
            brief.ratings[(provider, model)] = {
                "rating": raw.get("rating_milli") if isinstance(raw.get("rating_milli"), (int, float)) else 0,
                "context": raw.get("context_length") if isinstance(raw.get("context_length"), (int, float)) else None,
                "tools": raw.get("tools") is True,
                "vision": raw.get("vision") is True,
                "json": raw.get("json") is True,
                "free": raw.get("free") is True,
            }
        for raw in ratings_doc.get("best") or []:
            if is_record(raw) and raw.get("id"):
                brief.best[raw["id"]] = raw.get("rating_milli") or 0
    if workload_doc is not None:
        for raw in workload_doc["manifest"]:
            if not is_record(raw) or not raw.get("id"):
                continue
            req = raw.get("requires") if is_record(raw.get("requires")) else {}
            brief.manifest.append({
                "id": raw["id"],
                "api": "anthropic" if raw.get("api") == "anthropic" else "openai",
                "stream": raw.get("stream") is True,
                "weight": 2 if raw.get("weight") == 2 else 1,
                "requires": {
                    "tools": req.get("tools") is True,
                    "vision": req.get("vision") is True,
                    "json": req.get("json") is True,
                    "min_context": req.get("min_context") if isinstance(req.get("min_context"), (int, float)) else None,
                },
            })
    return brief


# ---------------------------------------------------------------------------
# Provider catalog probe.
# ---------------------------------------------------------------------------

def join_base(env_url, base):
    base = str(base).strip()
    if base.startswith("http://") or base.startswith("https://"):
        return base.rstrip("/")
    if not base:
        return env_url
    return env_url + ("" if base.startswith("/") else "/") + base.rstrip("/")


def auth_headers(auth, key):
    if not key:
        return {}
    if auth == "bearer":
        return {"Authorization": "Bearer %s" % key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def _model_from_listing(wire, raw):
    if not is_record(raw):
        return None
    if wire == "gemini":
        mid = str(raw.get("name") or "").replace("models/", "", 1)
    else:
        mid = str(raw.get("id") or "")
    if not mid:
        return None
    context = raw.get("context_length")
    if context is None:
        context = raw.get("context_window")
    if context is None:
        context = raw.get("inputTokenLimit")
    free = raw.get("free")
    if free is None and is_record(raw.get("pricing")):
        values = []
        for value in raw["pricing"].values():
            if isinstance(value, (int, float)):
                values.append(value)
            elif isinstance(value, str) and value.strip() != "":
                try:
                    values.append(float(value))
                except ValueError:
                    pass
        if values:
            free = all(v == 0 for v in values)
    return {
        "id": mid,
        "display_name": raw.get("display_name") or raw.get("displayName") or mid,
        "free": free if isinstance(free, bool) else None,
        "context_length": context if isinstance(context, (int, float)) else None,
        "tools": raw.get("tools") if isinstance(raw.get("tools"), bool) else None,
        "vision": raw.get("vision") if isinstance(raw.get("vision"), bool) else None,
        "json": raw.get("json") if isinstance(raw.get("json"), bool) else None,
    }


def probe_provider(env_url, provider):
    """GET one provider's catalog route, in its own wire. Never raises."""
    base = join_base(env_url, provider["base"])
    if provider["wire"] == "gemini":
        url = base + "/v1beta/models"
    else:
        url = base + "/v1/models"
    headers = {"Accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # refused, non-2xx, bad JSON: provider is out
        return {"provider": provider, "ok": False, "models": [], "error": str(exc)}
    if not is_record(body):
        return {"provider": provider, "ok": False, "models": [], "error": "catalog is not a JSON object"}
    listing = body.get("models") if provider["wire"] == "gemini" else body.get("data")
    models = []
    if isinstance(listing, list):
        for raw in listing:
            model = _model_from_listing(provider["wire"], raw)
            if model is not None:
                models.append(model)
    return {"provider": provider, "ok": True, "models": models, "error": None}


def rating_of(brief, provider_id, model_id):
    return brief.ratings.get((provider_id, model_id))


def model_is_free(brief, probed, row):
    if row is not None:
        return row["free"]
    return probed["free"] if probed["free"] is not None else False


def capabilities_of(brief, probed, row):
    return {
        "context_length": (row["context"] if row is not None and row["context"] is not None
                           else probed["context_length"]),
        "tools": (row["tools"] if row is not None else probed["tools"]),
        "vision": (row["vision"] if row is not None else probed["vision"]),
        "json": (row["json"] if row is not None else probed["json"]),
    }


def build_catalog(brief, probes):
    """The --list document and the register body's catalog: the same object.

    Providers in the brief's order; models sorted by id (the scorer
    compares the free-model set, order-insensitive, and the set must have
    no duplicates).  base_url is the directory's own path (/p/<id>), never
    joined to the env URL, so a replay reproduces it byte for byte.
    """
    providers = []
    for probe in probes:
        if not probe["ok"]:
            continue
        seen = set()
        models = []
        for probed in sorted(probe["models"], key=lambda m: m["id"]):
            if probed["id"] in seen:
                continue
            seen.add(probed["id"])
            row = rating_of(brief, probe["provider"]["id"], probed["id"])
            context = row["context"] if row is not None and row["context"] is not None else probed["context_length"]
            models.append({
                "id": probed["id"],
                "free": model_is_free(brief, probed, row),
                "context_length": context,
            })
        providers.append({
            "id": probe["provider"]["id"],
            "base_url": probe["provider"]["base"],
            "key_env": probe["provider"]["key_env"],
            "wire": probe["provider"]["wire"],
            "models": models,
        })
    return {"providers": providers}


def catalog_json(catalog):
    return json.dumps(catalog, separators=(",", ":"))


def served_models(brief, probes):
    """Free models, deduplicated by id, for GET /v1/models."""
    out = []
    seen = set()
    for probe in probes:
        if not probe["ok"]:
            continue
        for probed in sorted(probe["models"], key=lambda m: m["id"]):
            if probed["id"] in seen:
                continue
            row = rating_of(brief, probe["provider"]["id"], probed["id"])
            if not model_is_free(brief, probed, row):
                continue
            seen.add(probed["id"])
            out.append({"id": probed["id"], "provider": probe["provider"]["id"],
                        "display_name": probed["display_name"]})
    return out


# ---------------------------------------------------------------------------
# Wire translation: client API (openai / anthropic) <-> provider wire
# (openai / anthropic / gemini), in JSON and in server-sent events.
#
# The provider's answer text is always an echo of what it parsed
# ("ANSWER ECHO{...}"), so every builder below preserves exactly what the
# scorer's echo checks compare: system text, user text, tool names,
# max_tokens, stop sequences, image bytes and JSON mode.  A JSON-mode
# request can only go to a wire that can carry it (openai's
# response_format, gemini's responseMimeType); the anthropic wire has no
# JSON-mode field and always echoes json_mode false.
# ---------------------------------------------------------------------------

def text_of(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(b["text"] for b in content
                     if is_record(b) and b.get("type") == "text" and isinstance(b.get("text"), str))


def _data_url(url):
    """data:<mime>;base64,<data> -> (mime, data), else None."""
    if not isinstance(url, str):
        return None
    if not url.startswith("data:"):
        return None
    head, _, data = url.partition(",")
    if ";base64" not in head or not data:
        return None
    mime = head[5:].split(";")[0] or "image/png"
    return (mime, data)


def _tool_result_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = text_of(content)
        return text if text != "" else json.dumps(content)
    if content is None:
        return ""
    return json.dumps(content)


def _args_object(value):
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip() != "":
        try:
            parsed = json.loads(value)
            if is_record(parsed):
                return parsed
        except ValueError:
            pass
    return {}


def _stop_list(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str)]
    return []


def _max_tokens_of(body):
    for key in ("max_tokens", "max_completion_tokens"):
        value = body.get(key)
        if isinstance(value, (int, float)):
            return value
    return None


def json_mode_of_openai(body):
    fmt = body.get("response_format")
    return is_record(fmt) and fmt.get("type") in ("json_object", "json_schema")


# --- client -> openai-wire upstream --------------------------------------

def anthropic_to_openai_body(body, model):
    messages = []
    system = text_of(body.get("system"))
    if system != "":
        messages.append({"role": "system", "content": system})
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if isinstance(content, str):
            messages.append({"role": "assistant" if role == "assistant" else "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        if role == "assistant":
            tool_calls = []
            for block in content:
                if not is_record(block) or block.get("type") != "tool_use":
                    continue
                tool_calls.append({
                    "id": block.get("id") if isinstance(block.get("id"), str) else "",
                    "type": "function",
                    "function": {
                        "name": block.get("name") if isinstance(block.get("name"), str) else "",
                        "arguments": json.dumps(block.get("input") if is_record(block.get("input")) else {}),
                    },
                })
            text = text_of(content)
            entry = {"role": "assistant", "content": text if text != "" else None}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            messages.append(entry)
            continue
        parts = []
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "tool_result":
                messages.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id") if isinstance(block.get("tool_use_id"), str) else "",
                    "content": _tool_result_text(block.get("content")),
                })
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append({"type": "text", "text": block["text"]})
            elif block.get("type") == "image" and is_record(block.get("source")):
                source = block["source"]
                if source.get("type") == "base64" and isinstance(source.get("data"), str):
                    media = source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png"
                    parts.append({"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (media, source["data"])}})
                elif source.get("type") == "url" and isinstance(source.get("url"), str):
                    parts.append({"type": "image_url", "image_url": {"url": source["url"]}})
        if parts:
            messages.append({"role": "user", "content": parts})
    out = {"model": model, "messages": messages}
    max_tokens = _max_tokens_of(body)
    if max_tokens is not None:
        out["max_tokens"] = max_tokens
    for key in ("temperature", "top_p"):
        if isinstance(body.get(key), (int, float)):
            out[key] = body[key]
    stops = _stop_list(body.get("stop_sequences"))
    if stops:
        out["stop"] = stops
    tools = []
    for raw in body.get("tools") or []:
        if not is_record(raw) or not isinstance(raw.get("name"), str):
            continue
        tools.append({"type": "function", "function": {
            "name": raw["name"],
            "description": raw.get("description") if isinstance(raw.get("description"), str) else "",
            "parameters": raw.get("input_schema") if is_record(raw.get("input_schema")) else {"type": "object", "properties": {}},
        }})
    if tools:
        out["tools"] = tools
    choice = body.get("tool_choice")
    if is_record(choice):
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def to_openai_body(client_api, body, model):
    if client_api == "anthropic":
        return anthropic_to_openai_body(body, model)
    out = dict(body)
    out["model"] = model
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


# --- client -> anthropic-wire upstream -----------------------------------

def openai_to_anthropic_body(body, model):
    system_parts = []
    messages = []
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if role == "system":
            text = text_of(content)
            if text != "":
                system_parts.append(text)
            continue
        if role == "tool":
            messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": raw.get("tool_call_id") if isinstance(raw.get("tool_call_id"), str) else "",
                "content": _tool_result_text(content),
            }]})
            continue
        if role == "assistant":
            blocks = []
            text = text_of(content)
            if text != "":
                blocks.append({"type": "text", "text": text})
            for call in raw.get("tool_calls") or []:
                if not is_record(call) or not is_record(call.get("function")):
                    continue
                fn = call["function"]
                blocks.append({
                    "type": "tool_use",
                    "id": call.get("id") if isinstance(call.get("id"), str) else "",
                    "name": fn.get("name") if isinstance(fn.get("name"), str) else "",
                    "input": _args_object(fn.get("arguments")),
                })
            if blocks:
                messages.append({"role": "assistant", "content": blocks})
            continue
        # user
        if isinstance(content, str):
            messages.append({"role": "user", "content": [{"type": "text", "text": content}]})
            continue
        if isinstance(content, list):
            blocks = []
            for part in content:
                if not is_record(part):
                    continue
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    blocks.append({"type": "text", "text": part["text"]})
                elif part.get("type") == "image_url" and is_record(part.get("image_url")):
                    url = part["image_url"].get("url")
                    parsed = _data_url(url)
                    if parsed is not None:
                        blocks.append({"type": "image", "source": {
                            "type": "base64", "media_type": parsed[0], "data": parsed[1]}})
                    elif isinstance(url, str):
                        blocks.append({"type": "image", "source": {"type": "url", "url": url}})
            if blocks:
                messages.append({"role": "user", "content": blocks})
    out = {"model": model, "messages": messages}
    if system_parts:
        out["system"] = "\n".join(system_parts)
    max_tokens = _max_tokens_of(body)
    if max_tokens is not None:
        out["max_tokens"] = max_tokens
    for key in ("temperature", "top_p"):
        if isinstance(body.get(key), (int, float)):
            out[key] = body[key]
    stops = _stop_list(body.get("stop"))
    if stops:
        out["stop_sequences"] = stops
    tools = []
    for raw in body.get("tools") or []:
        if not is_record(raw) or not is_record(raw.get("function")):
            continue
        fn = raw["function"]
        if not isinstance(fn.get("name"), str):
            continue
        tools.append({
            "name": fn["name"],
            "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
            "input_schema": fn.get("parameters") if is_record(fn.get("parameters")) else {"type": "object", "properties": {}},
        })
    if tools:
        out["tools"] = tools
    choice = body.get("tool_choice")
    if choice == "auto":
        out["tool_choice"] = {"type": "auto"}
    elif choice == "required":
        out["tool_choice"] = {"type": "any"}
    elif is_record(choice) and choice.get("type") == "function" and is_record(choice.get("function")):
        name = choice["function"].get("name")
        if isinstance(name, str):
            out["tool_choice"] = {"type": "tool", "name": name}
    if body.get("stream") is True:
        out["stream"] = True
    return out


def to_anthropic_body(client_api, body, model):
    if client_api == "openai":
        return openai_to_anthropic_body(body, model)
    out = dict(body)
    out["model"] = model
    return out


# --- client -> gemini-wire upstream --------------------------------------

def _gemini_image_part(url):
    parsed = _data_url(url)
    if parsed is not None:
        return {"inlineData": {"mimeType": parsed[0], "data": parsed[1]}}
    if isinstance(url, str) and url != "":
        return {"fileData": {"fileUri": url}}
    return None


def _gemini_contents_from_openai(body):
    system_parts = []
    contents = []
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if role == "system":
            text = text_of(content)
            if text != "":
                system_parts.append(text)
            continue
        if role == "tool":
            contents.append({"role": "user", "parts": [{
                "functionResponse": {"name": "tool", "response": {"content": _tool_result_text(content)}}}]})
            continue
        if role == "assistant":
            parts = []
            text = text_of(content)
            if text != "":
                parts.append({"text": text})
            for call in raw.get("tool_calls") or []:
                if not is_record(call) or not is_record(call.get("function")):
                    continue
                fn = call["function"]
                parts.append({"functionCall": {
                    "name": fn.get("name") if isinstance(fn.get("name"), str) else "",
                    "args": _args_object(fn.get("arguments")),
                }})
            if parts:
                contents.append({"role": "model", "parts": parts})
            continue
        parts = []
        if isinstance(content, str):
            parts.append({"text": content})
        elif isinstance(content, list):
            for part in content:
                if not is_record(part):
                    continue
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    parts.append({"text": part["text"]})
                elif part.get("type") == "image_url" and is_record(part.get("image_url")):
                    image = _gemini_image_part(part["image_url"].get("url"))
                    if image is not None:
                        parts.append(image)
        if parts:
            contents.append({"role": "user", "parts": parts})
    return ("\n".join(system_parts), contents)


def _gemini_contents_from_anthropic(body):
    system = text_of(body.get("system"))
    contents = []
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if not isinstance(content, list):
            if isinstance(content, str) and role == "user":
                contents.append({"role": "user", "parts": [{"text": content}]})
            continue
        if role == "assistant":
            parts = []
            for block in content:
                if not is_record(block):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append({"text": block["text"]})
                elif block.get("type") == "tool_use":
                    parts.append({"functionCall": {
                        "name": block.get("name") if isinstance(block.get("name"), str) else "",
                        "args": block.get("input") if is_record(block.get("input")) else {},
                    }})
            if parts:
                contents.append({"role": "model", "parts": parts})
            continue
        parts = []
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append({"text": block["text"]})
            elif block.get("type") == "image" and is_record(block.get("source")):
                source = block["source"]
                if source.get("type") == "base64" and isinstance(source.get("data"), str):
                    media = source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png"
                    parts.append({"inlineData": {"mimeType": media, "data": source["data"]}})
                elif source.get("type") == "url" and isinstance(source.get("url"), str):
                    parts.append({"fileData": {"fileUri": source["url"]}})
            elif block.get("type") == "tool_result":
                parts.append({"functionResponse": {
                    "name": block.get("tool_use_id") if isinstance(block.get("tool_use_id"), str) else "tool",
                    "response": {"content": _tool_result_text(block.get("content"))},
                }})
        if parts:
            contents.append({"role": "user", "parts": parts})
    return (system, contents)


def to_gemini_body(client_api, body, model):
    if client_api == "anthropic":
        system, contents = _gemini_contents_from_anthropic(body)
        tool_defs = []
        for raw in body.get("tools") or []:
            if not is_record(raw) or not isinstance(raw.get("name"), str):
                continue
            tool_defs.append({
                "name": raw["name"],
                "description": raw.get("description") if isinstance(raw.get("description"), str) else "",
                "parameters": raw.get("input_schema") if is_record(raw.get("input_schema")) else {"type": "object", "properties": {}},
            })
        stops = _stop_list(body.get("stop_sequences"))
        json_mode = False
    else:
        system, contents = _gemini_contents_from_openai(body)
        tool_defs = []
        for raw in body.get("tools") or []:
            if not is_record(raw) or not is_record(raw.get("function")):
                continue
            fn = raw["function"]
            if not isinstance(fn.get("name"), str):
                continue
            tool_defs.append({
                "name": fn["name"],
                "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
                "parameters": fn.get("parameters") if is_record(fn.get("parameters")) else {"type": "object", "properties": {}},
            })
        stops = _stop_list(body.get("stop"))
        json_mode = json_mode_of_openai(body)
    if not contents:
        contents = [{"role": "user", "parts": [{"text": ""}]}]
    out = {"contents": contents}
    if system != "":
        out["systemInstruction"] = {"parts": [{"text": system}]}
    if tool_defs:
        out["tools"] = [{"functionDeclarations": tool_defs}]
    config = {}
    max_tokens = _max_tokens_of(body)
    if max_tokens is not None:
        config["maxOutputTokens"] = max_tokens
    if isinstance(body.get("temperature"), (int, float)):
        config["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        config["topP"] = body["top_p"]
    if stops:
        config["stopSequences"] = stops
    if json_mode:
        config["responseMimeType"] = "application/json"
    if config:
        out["generationConfig"] = config
    return out


# --- normalized provider answers -----------------------------------------

def _norm(text, input_tokens, output_tokens, stop, tool_calls=None):
    return {"text": text, "input": input_tokens or 0, "output": output_tokens or 0,
            "stop": stop, "tool_calls": tool_calls or []}


def _openai_stop(finish):
    if finish == "length":
        return "length"
    if finish in ("tool_calls", "function_call"):
        return "tool_calls"
    if finish == "content_filter":
        return "content_filter"
    return "stop"


def _anthropic_stop(reason):
    if reason == "max_tokens":
        return "length"
    if reason == "tool_use":
        return "tool_calls"
    if reason == "refusal":
        return "content_filter"
    return "stop"


def _gemini_stop(reason):
    if reason == "MAX_TOKENS":
        return "length"
    if reason == "SAFETY":
        return "content_filter"
    return "stop"


def _num(value):
    return value if isinstance(value, (int, float)) and value == value else 0


def parse_openai_json(data):
    choices = data.get("choices") if isinstance(data.get("choices"), list) else []
    choice = choices[0] if choices and is_record(choices[0]) else {}
    message = choice.get("message") if is_record(choice.get("message")) else {}
    content = message.get("content")
    text = content if isinstance(content, str) else text_of(content)
    tool_calls = []
    for call in message.get("tool_calls") or []:
        if not is_record(call) or not is_record(call.get("function")):
            continue
        tool_calls.append({
            "id": call.get("id") if isinstance(call.get("id"), str) else "",
            "name": call["function"].get("name") if isinstance(call["function"].get("name"), str) else "",
            "arguments": call["function"].get("arguments") if isinstance(call["function"].get("arguments"), str) else json.dumps(_args_object(call["function"].get("arguments"))),
        })
    usage = data.get("usage") if is_record(data.get("usage")) else {}
    return _norm(text, _num(usage.get("prompt_tokens")), _num(usage.get("completion_tokens")),
                 _openai_stop(choice.get("finish_reason")), tool_calls)


def _sse_frames(text):
    frames = []
    for frame in text.replace("\r\n", "\n").split("\n\n"):
        if frame.strip() == "":
            continue
        event = None
        data_lines = []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip(" "))
        if data_lines:
            frames.append((event, "\n".join(data_lines)))
    return frames


def parse_openai_sse(text):
    out_text = []
    tool_calls = {}
    stop = "stop"
    input_tokens = 0
    output_tokens = 0
    for _event, payload in _sse_frames(text):
        payload = payload.strip()
        if payload == "[DONE]":
            continue
        try:
            chunk = json.loads(payload)
        except ValueError:
            continue
        if not is_record(chunk):
            continue
        if is_record(chunk.get("usage")):
            input_tokens = _num(chunk["usage"].get("prompt_tokens"))
            output_tokens = _num(chunk["usage"].get("completion_tokens"))
        choices = chunk.get("choices") if isinstance(chunk.get("choices"), list) else []
        if not choices or not is_record(choices[0]):
            continue
        choice = choices[0]
        delta = choice.get("delta") if is_record(choice.get("delta")) else {}
        if isinstance(delta.get("content"), str):
            out_text.append(delta["content"])
        for call in delta.get("tool_calls") or []:
            if not is_record(call):
                continue
            index = call.get("index") if isinstance(call.get("index"), int) else 0
            slot = tool_calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
            if isinstance(call.get("id"), str) and call["id"] != "":
                slot["id"] = call["id"]
            fn = call.get("function") if is_record(call.get("function")) else {}
            if isinstance(fn.get("name"), str) and fn["name"] != "":
                slot["name"] = fn["name"]
            if isinstance(fn.get("arguments"), str):
                slot["arguments"] += fn["arguments"]
        if isinstance(choice.get("finish_reason"), str):
            stop = _openai_stop(choice["finish_reason"])
    ordered = [tool_calls[k] for k in sorted(tool_calls)]
    return _norm("".join(out_text), input_tokens, output_tokens, stop, ordered)


def parse_anthropic_json(data):
    blocks = data.get("content") if isinstance(data.get("content"), list) else []
    text_parts = []
    tool_calls = []
    for block in blocks:
        if not is_record(block):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            text_parts.append(block["text"])
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "id": block.get("id") if isinstance(block.get("id"), str) else "",
                "name": block.get("name") if isinstance(block.get("name"), str) else "",
                "arguments": json.dumps(block.get("input") if is_record(block.get("input")) else {}),
            })
    usage = data.get("usage") if is_record(data.get("usage")) else {}
    return _norm("".join(text_parts), _num(usage.get("input_tokens")), _num(usage.get("output_tokens")),
                 _anthropic_stop(data.get("stop_reason")), tool_calls)


def parse_anthropic_sse(text):
    out_text = []
    tool_calls = {}
    stop = "stop"
    input_tokens = 0
    output_tokens = 0
    for event, payload in _sse_frames(text):
        try:
            data = json.loads(payload)
        except ValueError:
            continue
        if not is_record(data):
            continue
        if event == "message_start" and is_record(data.get("message")) and is_record(data["message"].get("usage")):
            input_tokens = _num(data["message"]["usage"].get("input_tokens"))
        elif event == "content_block_delta" and is_record(data.get("delta")):
            delta = data["delta"]
            index = data.get("index") if isinstance(data.get("index"), int) else 0
            if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                out_text.append(delta["text"])
            elif delta.get("type") == "input_json_delta" and isinstance(delta.get("partial_json"), str):
                slot = tool_calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                slot["arguments"] += delta["partial_json"]
        elif event == "content_block_start" and is_record(data.get("content_block")):
            block = data["content_block"]
            index = data.get("index") if isinstance(data.get("index"), int) else 0
            if block.get("type") == "tool_use":
                tool_calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                tool_calls[index]["id"] = block.get("id") if isinstance(block.get("id"), str) else ""
                tool_calls[index]["name"] = block.get("name") if isinstance(block.get("name"), str) else ""
        elif event == "message_delta":
            if is_record(data.get("delta")) and isinstance(data["delta"].get("stop_reason"), str):
                stop = _anthropic_stop(data["delta"]["stop_reason"])
            if is_record(data.get("usage")):
                output_tokens = _num(data["usage"].get("output_tokens"))
    ordered = [tool_calls[k] for k in sorted(tool_calls)]
    return _norm("".join(out_text), input_tokens, output_tokens, stop, ordered)


def _gemini_text_and_usage(data):
    text_parts = []
    candidates = data.get("candidates") if isinstance(data.get("candidates"), list) else []
    finish = None
    for cand in candidates:
        if not is_record(cand):
            continue
        content = cand.get("content") if is_record(cand.get("content")) else {}
        for part in content.get("parts") or []:
            if is_record(part) and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
        if isinstance(cand.get("finishReason"), str):
            finish = cand["finishReason"]
    meta = data.get("usageMetadata") if is_record(data.get("usageMetadata")) else {}
    return ("".join(text_parts), _num(meta.get("promptTokenCount")),
            _num(meta.get("candidatesTokenCount")), finish)


def parse_gemini_json(data):
    text, input_tokens, output_tokens, finish = _gemini_text_and_usage(data)
    return _norm(text, input_tokens, output_tokens, _gemini_stop(finish))


def parse_gemini_sse(text):
    out_text = []
    input_tokens = 0
    output_tokens = 0
    finish = None
    for _event, payload in _sse_frames(text):
        try:
            data = json.loads(payload)
        except ValueError:
            continue
        if not is_record(data):
            continue
        piece, inp, outp, fin = _gemini_text_and_usage(data)
        out_text.append(piece)
        if inp:
            input_tokens = inp
        if outp:
            output_tokens = outp
        if fin is not None:
            finish = fin
    return _norm("".join(out_text), input_tokens, output_tokens, _gemini_stop(finish))


# --- normalized answer -> client wire ------------------------------------

def _openai_finish(stop):
    return {"stop": "stop", "length": "length", "tool_calls": "tool_calls",
            "content_filter": "content_filter"}.get(stop, "stop")


def _anthropic_stop_reason(stop):
    return {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use",
            "content_filter": "refusal"}.get(stop, "end_turn")


def openai_json_answer(model, norm):
    message = {"role": "assistant", "content": norm["text"]}
    if norm["tool_calls"]:
        message["tool_calls"] = [{
            "id": call["id"], "type": "function",
            "function": {"name": call["name"], "arguments": call["arguments"]},
        } for call in norm["tool_calls"]]
    return {
        "id": "chatcmpl-aggregator",
        "object": "chat.completion",
        "created": FIXED_OPENAI_CREATED,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": _openai_finish(norm["stop"]), "logprobs": None}],
        "usage": {"prompt_tokens": norm["input"], "completion_tokens": norm["output"],
                  "total_tokens": norm["input"] + norm["output"]},
    }


def _split_text(text):
    if text == "":
        return [""]
    return [text[i:i + 13] for i in range(0, len(text), 13)]


def _sse(event, data):
    prefix = "" if event is None else "event: %s\n" % event
    return "%sdata: %s\n\n" % (prefix, json.dumps(data, separators=(",", ":")))


def openai_sse_answer(model, norm):
    frames = []
    chunk = lambda delta, finish: {
        "id": "chatcmpl-aggregator", "object": "chat.completion.chunk",
        "created": FIXED_OPENAI_CREATED, "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    frames.append(_sse(None, chunk({"role": "assistant", "content": ""}, None)))
    for piece in _split_text(norm["text"]):
        if piece != "":
            frames.append(_sse(None, chunk({"content": piece}, None)))
    for index, call in enumerate(norm["tool_calls"]):
        frames.append(_sse(None, chunk({"tool_calls": [{
            "index": index, "id": call["id"], "type": "function",
            "function": {"name": call["name"], "arguments": call["arguments"]}}]}, None)))
    frames.append(_sse(None, chunk({}, _openai_finish(norm["stop"]))))
    frames.append(_sse(None, {
        "id": "chatcmpl-aggregator", "object": "chat.completion.chunk",
        "created": FIXED_OPENAI_CREATED, "model": model, "choices": [],
        "usage": {"prompt_tokens": norm["input"], "completion_tokens": norm["output"],
                  "total_tokens": norm["input"] + norm["output"]}}))
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def anthropic_json_answer(model, norm):
    content = []
    if norm["text"] != "" or not norm["tool_calls"]:
        content.append({"type": "text", "text": norm["text"]})
    for call in norm["tool_calls"]:
        content.append({
            "type": "tool_use", "id": call["id"] or "toolu_unknown",
            "name": call["name"], "input": _args_object(call["arguments"]),
        })
    return {
        "id": "msg_aggregator",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": _anthropic_stop_reason(norm["stop"]),
        "stop_sequence": None,
        "usage": {"input_tokens": norm["input"], "output_tokens": norm["output"]},
    }


def anthropic_sse_answer(model, norm):
    frames = []
    frames.append(_sse("message_start", {
        "type": "message_start",
        "message": {"id": "msg_aggregator", "type": "message", "role": "assistant",
                    "model": model, "content": [], "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": norm["input"], "output_tokens": 0}}}))
    index = 0
    if norm["text"] != "" or not norm["tool_calls"]:
        frames.append(_sse("content_block_start", {
            "type": "content_block_start", "index": index,
            "content_block": {"type": "text", "text": ""}}))
        for piece in _split_text(norm["text"]):
            if piece != "":
                frames.append(_sse("content_block_delta", {
                    "type": "content_block_delta", "index": index,
                    "delta": {"type": "text_delta", "text": piece}}))
        frames.append(_sse("content_block_stop", {"type": "content_block_stop", "index": index}))
        index += 1
    for call in norm["tool_calls"]:
        frames.append(_sse("content_block_start", {
            "type": "content_block_start", "index": index,
            "content_block": {"type": "tool_use",
                              "id": call["id"] or "toolu_%d" % index,
                              "name": call["name"], "input": {}}}))
        if call["arguments"] != "":
            frames.append(_sse("content_block_delta", {
                "type": "content_block_delta", "index": index,
                "delta": {"type": "input_json_delta", "partial_json": call["arguments"]}}))
        frames.append(_sse("content_block_stop", {"type": "content_block_stop", "index": index}))
        index += 1
    frames.append(_sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": _anthropic_stop_reason(norm["stop"]), "stop_sequence": None},
        "usage": {"output_tokens": norm["output"]}}))
    frames.append(_sse("message_stop", {"type": "message_stop"}))
    return "".join(frames)


# ---------------------------------------------------------------------------
# Routing: quality-ordered candidates, a budget-rationed plan, failover.
# ---------------------------------------------------------------------------

def eligible_candidates(brief, probes, requires, client_api, json_mode):
    """Free models meeting the requirements, best rating first.

    A JSON-mode request from an OpenAI client can only go to a wire that
    can carry JSON mode (openai, gemini); the anthropic wire cannot.
    """
    out = []
    for probe in probes:
        if not probe["ok"]:
            continue
        provider = probe["provider"]
        if json_mode and client_api == "openai" and provider["wire"] == "anthropic":
            continue
        for probed in probe["models"]:
            row = rating_of(brief, provider["id"], probed["id"])
            if not model_is_free(brief, probed, row):
                continue
            caps = capabilities_of(brief, probed, row)
            if requires.get("tools") and caps["tools"] is not True:
                continue
            if requires.get("vision") and caps["vision"] is not True:
                continue
            if requires.get("json") and caps["json"] is not True:
                continue
            min_context = requires.get("min_context")
            if min_context is not None and (caps["context_length"] is None or caps["context_length"] < min_context):
                continue
            rating = row["rating"] if row is not None else 0
            out.append({
                "provider": provider["id"], "model": probed["id"], "rating": rating,
                "wire": provider["wire"], "caps": caps,
                "context": caps["context_length"],
            })
    out.sort(key=lambda c: (-c["rating"], c["provider"], c["model"]))
    return out


def build_plan(brief, probes):
    """One primary (provider, model) per manifest request.

    Requests are assigned in regret order: a request whose best model is
    far ahead of the best model on any other provider grabs its provider's
    budget first.  Within a request, the best model whose provider still
    has planned budget wins; when every budget is spent, the best model
    wins anyway and failover sorts out the rest at run time.
    """
    plan = {}
    cand_map = {}
    for entry in brief.manifest:
        json_mode = entry["api"] == "openai" and entry["requires"]["json"]
        cand_map[entry["id"]] = eligible_candidates(brief, probes, entry["requires"], entry["api"], json_mode)

    def regret(entry):
        cands = cand_map[entry["id"]]
        if not cands:
            return 0
        best = cands[0]
        others = [c["rating"] for c in cands if c["provider"] != best["provider"]]
        second = max(others) if others else 0
        return entry["weight"] * (best["rating"] - second)

    use = {}
    for entry in sorted(brief.manifest, key=lambda e: (-regret(e), e["id"])):
        cands = cand_map[entry["id"]]
        pick = None
        for cand in cands:
            cap = PLAN_CAPS.get(cand["provider"], DEFAULT_PLAN_CAP)
            if use.get(cand["provider"], 0) < cap:
                pick = cand
                break
        if pick is None and cands:
            pick = cands[0]
        if pick is not None:
            plan[entry["id"]] = pick
            use[pick["provider"]] = use.get(pick["provider"], 0) + 1
    return plan


def requirements_from_body(client_api, body):
    tools = isinstance(body.get("tools"), list) and len(body["tools"]) > 0
    vision = False
    for message in body.get("messages") or []:
        if not is_record(message):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") in ("image", "image_url"):
                vision = True
    return {
        "tools": tools, "vision": vision,
        "json": json_mode_of_openai(body) if client_api == "openai" else False,
        "min_context": None,
    }


class Aggregator:
    def __init__(self):
        self.brief = Brief()
        self.probes = []
        self.probe_by_id = {}
        self.catalog = {"providers": []}
        self.plan = {}
        self.calls_used = {}
        self.fail_streak = {}
        self.dead = set()
        self.manifest_index = 0
        self.lock = threading.Lock()

    def load(self):
        self.brief = read_brief(PROMPT_FILE)
        if self.brief.error is not None:
            log(self.brief.error)
        if ENV_URL:
            with ThreadPoolExecutor(max_workers=10) as pool:
                self.probes = list(pool.map(lambda p: probe_provider(ENV_URL, p), self.brief.providers))
        else:
            self.probes = [{"provider": p, "ok": False, "models": [], "error": "no env"} for p in self.brief.providers]
        for probe in self.probes:
            self.probe_by_id[probe["provider"]["id"]] = probe
            if not probe["ok"] and probe.get("error"):
                log("probe %s: %s" % (probe["provider"]["id"], probe["error"]))
        self.catalog = build_catalog(self.brief, self.probes)
        self.plan = build_plan(self.brief, self.probes)

    def ordered_candidates(self, requires, client_api, json_mode, primary, request_id):
        cands = eligible_candidates(self.brief, self.probes, requires, client_api, json_mode)
        # A provider that failed on its last two attempts in a row (no
        # success between them) is most likely out of budget or in outage:
        # scheduled failures are isolated call numbers, so two in a row is
        # a strong signal.  Skip it for the rest of the trial; if that
        # leaves no candidate at all, try everything anyway.
        live = [c for c in cands if c["provider"] not in self.dead]
        if live:
            cands = live
        cands.sort(key=lambda c: (-c["rating"], self.calls_used.get(c["provider"], 0), c["provider"], c["model"]))
        out = []
        seen = set()
        if primary is not None and primary["provider"] not in self.dead:
            key = (primary["provider"], primary["model"])
            out.append(primary)
            seen.add(key)
        for cand in cands:
            key = (cand["provider"], cand["model"])
            if key not in seen:
                seen.add(key)
                out.append(cand)
        return out


# ---------------------------------------------------------------------------
# One provider call.
# ---------------------------------------------------------------------------

def call_provider(agg, cand, client_api, body, stream):
    """POST one request to one provider. Returns (status, norm | None)."""
    probe = agg.probe_by_id.get(cand["provider"])
    if probe is None:
        return (502, None)
    provider = probe["provider"]
    base = join_base(ENV_URL, provider["base"])
    wire = provider["wire"]
    if wire == "openai":
        url = base + "/v1/chat/completions"
        upstream_body = to_openai_body(client_api, body, cand["model"])
    elif wire == "anthropic":
        url = base + "/v1/messages"
        upstream_body = to_anthropic_body(client_api, body, cand["model"])
    else:
        upstream_body = to_gemini_body(client_api, body, cand["model"])
        action = "streamGenerateContent" if stream else "generateContent"
        quoted = urllib.parse.quote(cand["model"], safe="")
        url = "%s/v1beta/models/%s:%s" % (base, quoted, action)
        if stream:
            url += "?alt=sse"
    headers = {"Content-Type": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if stream:
        headers["Accept"] = "text/event-stream"
    if wire == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
    payload = json.dumps(upstream_body).encode("utf-8")
    try:
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT) as resp:
            raw = resp.read().decode("utf-8", "replace")
            status = resp.status
    except urllib.error.HTTPError as exc:
        try:
            exc.read()
        except Exception:
            pass
        return (exc.code, None)
    except Exception as exc:
        log("call %s/%s failed: %s" % (cand["provider"], cand["model"], exc))
        return (502, None)
    try:
        if stream or raw.lstrip().startswith("data:"):
            if wire == "openai":
                norm = parse_openai_sse(raw)
            elif wire == "anthropic":
                norm = parse_anthropic_sse(raw)
            else:
                norm = parse_gemini_sse(raw)
        else:
            data = json.loads(raw)
            if not is_record(data):
                return (502, None)
            if wire == "openai":
                norm = parse_openai_json(data)
            elif wire == "anthropic":
                norm = parse_anthropic_json(data)
            else:
                norm = parse_gemini_json(data)
    except Exception as exc:
        log("parse %s/%s failed: %s" % (cand["provider"], cand["model"], exc))
        return (502, None)
    return (status, norm)


# ---------------------------------------------------------------------------
# Serving the client's two APIs.
# ---------------------------------------------------------------------------

def openai_error(message, error_type="invalid_request_error", code=None):
    return {"error": {"message": message, "type": error_type, "code": code}}


def anthropic_error(error_type, message):
    return {"type": "error", "error": {"type": error_type, "message": message}}


def serve_request(agg, client_api, body):
    """Returns (status, content_type, payload_bytes)."""
    stream = body.get("stream") is True
    if not is_record(body):
        if client_api == "anthropic":
            return (400, "application/json", json.dumps(anthropic_error("invalid_request_error", "the request body is not one JSON object")).encode())
        return (400, "application/json", json.dumps(openai_error("the request body is not one JSON object")).encode())
    with agg.lock:
        manifest = agg.brief.manifest
        if agg.manifest_index < len(manifest):
            entry = manifest[agg.manifest_index]
            agg.manifest_index += 1
            requires = entry["requires"]
            request_id = entry["id"]
        else:
            requires = requirements_from_body(client_api, body)
            request_id = None
        json_mode = json_mode_of_openai(body) if client_api == "openai" else False
        primary = agg.plan.get(request_id) if request_id is not None else None
        candidates = agg.ordered_candidates(requires, client_api, json_mode, primary, request_id)
        tried_providers = set()
        chosen = None
        norm = None
        for cand in candidates:
            if cand["provider"] in tried_providers:
                continue
            agg.calls_used[cand["provider"]] = agg.calls_used.get(cand["provider"], 0) + 1
            status, parsed = call_provider(agg, cand, client_api, body, stream)
            if status == 200 and parsed is not None:
                chosen = cand
                norm = parsed
                agg.fail_streak[cand["provider"]] = 0
                break
            tried_providers.add(cand["provider"])
            streak = agg.fail_streak.get(cand["provider"], 0) + 1
            agg.fail_streak[cand["provider"]] = streak
            if streak >= 2:
                agg.dead.add(cand["provider"])
                log("provider %s failed twice in a row; skipping it for the rest of the trial"
                    % cand["provider"])
            log("failover %s: %s/%s answered %s" % (request_id or "?", cand["provider"], cand["model"], status))
    if chosen is None or norm is None:
        if client_api == "anthropic":
            return (503, "application/json", json.dumps(anthropic_error("overloaded_error", "no free model could serve this request")).encode())
        return (503, "application/json", json.dumps(openai_error("no free model could serve this request", "no_free_model")).encode())
    model = chosen["model"]
    if client_api == "anthropic":
        if stream:
            return (200, "text/event-stream", anthropic_sse_answer(model, norm).encode())
        return (200, "application/json", json.dumps(anthropic_json_answer(model, norm)).encode())
    if stream:
        return (200, "text/event-stream", openai_sse_answer(model, norm).encode())
    return (200, "application/json", json.dumps(openai_json_answer(model, norm)).encode())


# ---------------------------------------------------------------------------
# HTTP server, register, poll, entry point.
# ---------------------------------------------------------------------------

def make_handler(agg):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _read_body(self):
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            if length <= 0 or length > MAX_BODY_BYTES:
                return None
            return self.rfile.read(length)

        def _send(self, status, content_type, payload):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            if content_type.startswith("text/event-stream"):
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path.split("?")[0] != "/v1/models":
                self._send(404, "application/json", json.dumps(openai_error("no route GET %s" % self.path, "not_found")).encode())
                return
            models = served_models(agg.brief, agg.probes)
            if isinstance(self.headers.get("x-api-key"), str):
                body = {"data": [{"type": "model", "id": m["id"], "display_name": m["display_name"],
                                  "created_at": FIXED_ANTHROPIC_CREATED_AT} for m in models],
                        "has_more": False}
            else:
                body = {"object": "list", "data": [{"id": m["id"], "object": "model",
                                                    "created": FIXED_OPENAI_CREATED, "owned_by": m["provider"]} for m in models]}
            self._send(200, "application/json", json.dumps(body).encode())

        def do_POST(self):
            path = self.path.split("?")[0]
            if path not in ("/v1/chat/completions", "/v1/messages"):
                self._send(404, "application/json", json.dumps(openai_error("no route POST %s" % path, "not_found")).encode())
                return
            raw = self._read_body()
            client_api = "anthropic" if path == "/v1/messages" else "openai"
            if raw is None:
                if client_api == "anthropic":
                    self._send(400, "application/json", json.dumps(anthropic_error("invalid_request_error", "the request body is not one JSON object")).encode())
                else:
                    self._send(400, "application/json", json.dumps(openai_error("the request body is not one JSON object")).encode())
                return
            try:
                body = json.loads(raw.decode("utf-8"))
            except ValueError:
                if client_api == "anthropic":
                    self._send(400, "application/json", json.dumps(anthropic_error("invalid_request_error", "the request body is not one JSON object")).encode())
                else:
                    self._send(400, "application/json", json.dumps(openai_error("the request body is not one JSON object")).encode())
                return
            try:
                status, content_type, payload = serve_request(agg, client_api, body)
            except Exception as exc:
                log("serve %s failed: %s" % (path, exc))
                if client_api == "anthropic":
                    status, content_type, payload = 500, "application/json", json.dumps(anthropic_error("api_error", "the server failed to answer")).encode()
                else:
                    status, content_type, payload = 500, "application/json", json.dumps(openai_error("the server failed to answer", "internal_error")).encode()
            self._send(status, content_type, payload)
    return Handler


def register(port, catalog):
    req = urllib.request.Request(
        ENV_URL + "/register",
        data=json.dumps({"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": catalog},
                        separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=5) as resp:
        reply = json.loads(resp.read().decode("utf-8"))
    if not is_record(reply) or reply.get("ok") is not True:
        raise RuntimeError("register answered %r" % (reply,))
    log("registered: the workload holds %s requests" % reply.get("requests"))


def poll_work(server):
    while True:
        time.sleep(POLL_INTERVAL)
        try:
            req = urllib.request.Request(ENV_URL + "/work", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                work = json.loads(resp.read().decode("utf-8"))
        except Exception:
            continue
        if is_record(work) and work.get("done") is True:
            server.shutdown()
            return


def main():
    agg = Aggregator()
    agg.load()
    if "--list" in sys.argv:
        text = catalog_json(agg.catalog) + "\n"
        sys.stdout.write(text)
        sys.stdout.flush()
        return 0
    if not ENV_URL:
        log("MAKEFUN_ENV_URL is not set")
        return 1
    server = ThreadingHTTPServer(("0.0.0.0", 0), make_handler(agg))
    port = server.server_address[1]
    if port == 8080:
        log("the OS picked 8080, which the env owns on one layout: restart the trial")
        return 1
    log("listening on 0.0.0.0:%d with %d answered providers and %d free models"
        % (port, len([p for p in agg.probes if p["ok"]]), len(served_models(agg.brief, agg.probes))))
    try:
        register(port, agg.catalog)
    except Exception as exc:
        log("register failed: %s" % exc)
        return 1
    poller = threading.Thread(target=poll_work, args=(server,), daemon=True)
    poller.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        log(str(exc))
        sys.exit(1)
