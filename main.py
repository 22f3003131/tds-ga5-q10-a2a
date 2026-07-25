"""
GA5 Q10 - A2A Invoice Agent

Architecture:
- FastAPI app, deployed as a persistent process (Render/Railway), NOT Vercel
  serverless — this task needs real state to survive across separate requests
  (initial message -> grader evaluates -> results continuation -> cancel).
- SQLite file on disk for: tasks, idempotency records, and a package-decision
  cache (so retries/Check-then-Save never re-invoke the model).
- Auth model: ANY presented Bearer token is accepted as an identity — the
  token itself IS the principal id (hashed for storage). This matches "treat
  every Bearer token as a separate user": there's no single shared secret to
  validate against, just "missing token -> 401".
"""

import os
import json
import time
import uuid
import hashlib
import sqlite3
import httpx
import traceback
from fastapi import FastAPI, APIRouter, Request, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
from typing import Optional

app = FastAPI()
a2a = APIRouter(prefix="/a2a")

DEBUG_SECRET = os.environ.get("DEBUG_SECRET", "letmein")
ERROR_LOG_PATH = os.environ.get("ERROR_LOG_PATH", "./error_log.json")


def log_error(context: str, exc: Exception, extra: dict = None):
    try:
        try:
            with open(ERROR_LOG_PATH) as f:
                log = json.load(f)
        except Exception:
            log = []
        log.append({
            "time": time.time(),
            "context": context,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "extra": extra or {},
        })
        log = log[-30:]
        with open(ERROR_LOG_PATH, "w") as f:
            json.dump(log, f)
    except Exception:
        pass  # never let logging itself break the response

DB_PATH = os.environ.get("DB_PATH", "./a2a.db")
BASE_URL = os.environ.get("BASE_URL", "https://tds-ga5-q10-a2a.onrender.com/a2a/")
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN", "")

VALID_ACTIONS = {"settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"}
PROPOSAL_MEDIA = "application/vnd.ga5.invoice-action-proposals+json"
RECEIPT_MEDIA = "application/vnd.ga5.invoice-action-receipts+json"
BATCH_MEDIA = "application/vnd.ga5.invoice-claim-batch+json"
RESULTS_MEDIA = "application/vnd.ga5.invoice-action-results+json"


# ---------------- storage ----------------

def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""CREATE TABLE IF NOT EXISTS tasks (
        id TEXT PRIMARY KEY,
        context_id TEXT,
        principal TEXT,
        state TEXT,
        batch_id TEXT,
        history TEXT,
        artifacts TEXT,
        created_at REAL,
        updated_at REAL
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS idempotency (
        principal TEXT,
        message_id TEXT,
        message_hash TEXT,
        response TEXT,
        status_code INTEGER,
        PRIMARY KEY (principal, message_id)
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS package_cache (
        content_hash TEXT PRIMARY KEY,
        decision TEXT
    )""")
    conn.commit()
    conn.close()


init_db()


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def hash_json(obj) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


# ---------------- auth / version ----------------

def get_principal(authorization: Optional[str] = Header(None)) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing bearer token")
    return hashlib.sha256(token.encode()).hexdigest()


def check_version(a2a_version: Optional[str] = Header(None, alias="A2A-Version")):
    if a2a_version != "1.0":
        raise HTTPException(status_code=400, detail="Unsupported or missing A2A-Version")


# ---------------- agent card (public) ----------------

@app.get("/.well-known/agent-card.json")
def agent_card():
    return {
        "name": "Invoice Action Agent",
        "description": "Reviews invoice packages and proposes one business action per invoice, citing evidence.",
        "version": "1.0.0",
        "capabilities": {},
        "skills": [{
            "id": "invoice_action_agent",
            "name": "Invoice Action Agent",
            "description": "Reads invoice claim batches and proposes settle/approve/hold/reject/escalate actions with cited evidence.",
            "tags": ["invoice", "finance", "a2a"],
        }],
        "supportedInterfaces": [{
            "url": BASE_URL,
            "protocolBinding": "HTTP+JSON",
            "protocolVersion": "1.0",
        }],
        "defaultInputModes": [BATCH_MEDIA],
        "defaultOutputModes": [PROPOSAL_MEDIA, RECEIPT_MEDIA],
    }


# ---------------- AI decision step ----------------

def build_prompt(packages, policy_revision) -> str:
    actions_desc = (
        "settle_invoice: valid, reconciled, and within autonomous authority.\n"
        "request_approval: commercially valid, but outside delegated authority.\n"
        "hold_invoice: payment pauses until a stated verification completes.\n"
        "reject_duplicate: the same commercial invoice was already paid.\n"
        "open_exception: material records conflict and need an exception workflow."
    )
    return (
        "You are an invoice-review agent. For EACH package below, choose exactly one action "
        f"from this list:\n{actions_desc}\n\n"
        f"Policy revision: {policy_revision}\n\n"
        "Each package mixes CONTROLLING facts (the current, decisive statements that determine "
        "the correct action) with distractors: archived/old examples, negated statements, cover-sheet "
        "summaries, and irrelevant action words. You MUST base the decision only on the controlling "
        "statements, never on archived examples, negated claims, or the cover sheet.\n\n"
        "Return strict JSON: {\"decisions\": [{\"packageId\": str, \"action\": str, "
        "\"facts\": {\"vendorName\": str, \"invoiceNumber\": str, \"amountMinor\": int, \"currency\": str}, "
        "\"evidenceRefs\": [str, str, ...], \"rationale\": str}]}\n"
        "evidenceRefs: return the exact bracketed reference IDs (e.g. [Section 3.2]) copied verbatim "
        "from the controlling sentence(s) — at least two, and only ones that actually determine the "
        "action (never the cover-sheet reference or an archived-example reference).\n"
        "rationale: 60-1500 characters. Name the chosen action explicitly and explain, referencing the "
        "evidenceRefs, how each piece of cited evidence supports that specific action.\n\n"
        f"Packages:\n{json.dumps(packages, indent=None)}"
    )


def get_ai_decisions_batch(packages, policy_revision):
    prompt = build_prompt(packages, policy_revision)
    resp = httpx.post(
        "https://aipipe.org/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {AIPIPE_TOKEN}"},
        json={
            "model": "gpt-4.1-nano",
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        },
        timeout=40,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    decisions = {d["packageId"]: d for d in parsed["decisions"]}
    return decisions


def get_decisions_with_cache(conn, packages, policy_revision):
    uncached = []
    cached_map = {}
    for pkg in packages:
        content_hash = hash_json(pkg)
        row = conn.execute("SELECT decision FROM package_cache WHERE content_hash=?", (content_hash,)).fetchone()
        if row:
            cached_map[pkg["packageId"]] = (content_hash, json.loads(row["decision"]))
        else:
            uncached.append(pkg)

    if uncached:
        fresh = get_ai_decisions_batch(uncached, policy_revision)
        for pkg in uncached:
            d = fresh.get(pkg["packageId"])
            if not d or d.get("action") not in VALID_ACTIONS:
                d = {
                    "action": "open_exception",
                    "facts": {"vendorName": "", "invoiceNumber": "", "amountMinor": 0, "currency": ""},
                    "evidenceRefs": ["[unresolved]", "[unresolved]"],
                    "rationale": "Model output missing or invalid; routed to exception workflow for manual review.",
                }
            content_hash = hash_json(pkg)
            conn.execute("INSERT OR REPLACE INTO package_cache VALUES (?,?)", (content_hash, json.dumps(d)))
            cached_map[pkg["packageId"]] = (content_hash, d)

    return cached_map


# ---------------- task response builder ----------------

def build_task_response(row):
    history = json.loads(row["history"])
    artifacts_raw = json.loads(row["artifacts"])
    parts = []
    for a in artifacts_raw:
        mt = PROPOSAL_MEDIA if "proposals" in a else RECEIPT_MEDIA
        parts.append({"parts": [{"mediaType": mt, "data": a}]})
    return {"task": {
        "id": row["id"],
        "contextId": row["context_id"],
        "state": row["state"],
        "history": history,
        "artifacts": parts,
    }}


# ---------------- message handlers ----------------

def handle_initial_batch(conn, principal, message, data):
    batch_id = data.get("batchId")
    policy_revision = data.get("policyRevision")
    packages = data.get("packages", [])
    if not batch_id or not packages:
        raise HTTPException(status_code=400, detail="Malformed batch")

    task_id = str(uuid.uuid4())
    context_id = str(uuid.uuid4())

    decisions = get_decisions_with_cache(conn, packages, policy_revision)

    proposals = []
    for pkg in packages:
        content_hash, d = decisions[pkg["packageId"]]
        action_id = f"act-{content_hash[:20]}"
        proposals.append({
            "packageId": pkg["packageId"],
            "actionId": action_id,
            "action": d["action"],
            "facts": d["facts"],
            "evidenceRefs": d["evidenceRefs"],
            "rationale": d["rationale"],
        })

    history = [message]
    artifact = {"batchId": batch_id, "proposals": proposals}

    conn.execute(
        "INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?,?)",
        (task_id, context_id, principal, "TASK_STATE_INPUT_REQUIRED", batch_id,
         json.dumps(history), json.dumps([artifact]), time.time(), time.time()),
    )
    conn.commit()

    row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    return build_task_response(row)


def handle_result_continuation(conn, principal, message, data, task_id, context_id):
    if not task_id or not context_id:
        raise HTTPException(status_code=400, detail="Missing taskId/contextId")

    row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    if not row or row["principal"] != principal or row["context_id"] != context_id:
        raise HTTPException(status_code=404, detail="Task not found")

    if row["state"] in ("TASK_STATE_COMPLETED", "TASK_STATE_CANCELED"):
        # terminal replay — return the task as-is rather than erroring
        return build_task_response(row)

    artifacts_raw = json.loads(row["artifacts"])
    stored_artifact = artifacts_raw[0]
    batch_id = data.get("batchId")
    results = data.get("results", [])

    if batch_id != stored_artifact["batchId"]:
        raise HTTPException(status_code=400, detail="Batch mismatch")

    proposal_by_key = {(p["packageId"], p["actionId"]): p for p in stored_artifact["proposals"]}

    executions = []
    for r in results:
        key = (r.get("packageId"), r.get("actionId"))
        proposal = proposal_by_key.get(key)
        if not proposal or proposal["action"] != r.get("action") or r.get("outcome") != "ACCEPTED":
            continue  # reject / mismatch: never executed
        executions.append({
            "packageId": proposal["packageId"],
            "actionId": proposal["actionId"],
            "action": proposal["action"],
            "receiptNonce": r.get("receiptNonce"),
            "facts": proposal["facts"],
            "evidenceRefs": proposal["evidenceRefs"],
        })

    history = json.loads(row["history"])
    history.append(message)
    artifacts_raw.append({"batchId": batch_id, "executions": executions})

    # atomic state transition: only succeeds if still non-terminal, so a
    # concurrent cancel and this result continuation can't both "win"
    cur = conn.execute(
        "UPDATE tasks SET state='TASK_STATE_COMPLETED', history=?, artifacts=?, updated_at=? "
        "WHERE id=? AND state='TASK_STATE_INPUT_REQUIRED'",
        (json.dumps(history), json.dumps(artifacts_raw), time.time(), task_id),
    )
    conn.commit()
    if cur.rowcount == 0:
        row2 = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        raise HTTPException(status_code=409, detail=f"Task already {row2['state']}")

    row2 = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    return build_task_response(row2)


@a2a.post("/message:send")
async def message_send(request: Request, principal: str = Depends(get_principal), _=Depends(check_version)):
    content_type = request.headers.get("content-type", "")
    if not content_type.split(";")[0].strip().lower() == "application/a2a+json":
        raise HTTPException(status_code=400, detail="Content-Type must be application/a2a+json")

    body = await request.json()
    message = body.get("message", {})
    message_id = message.get("messageId")
    role = message.get("role")
    parts = message.get("parts", [])
    task_id = message.get("taskId")
    context_id = message.get("contextId")

    if not message_id or role != "ROLE_USER" or not parts:
        raise HTTPException(status_code=400, detail="Malformed message")

    part = parts[0]
    media_type = part.get("mediaType")
    data = part.get("data", {})
    msg_hash = hash_json(message)

    conn = get_db()
    existing = conn.execute(
        "SELECT * FROM idempotency WHERE principal=? AND message_id=?", (principal, message_id)
    ).fetchone()
    if existing:
        if existing["message_hash"] != msg_hash:
            conn.close()
            raise HTTPException(status_code=409, detail="IDEMPOTENCY_CONFLICT")
        resp = JSONResponse(
            content=json.loads(existing["response"]),
            status_code=existing["status_code"],
            media_type="application/a2a+json",
        )
        conn.close()
        return resp

    try:
        if media_type == BATCH_MEDIA:
            result = handle_initial_batch(conn, principal, message, data)
        elif media_type == RESULTS_MEDIA:
            result = handle_result_continuation(conn, principal, message, data, task_id, context_id)
        else:
            raise HTTPException(status_code=400, detail="Unsupported mediaType")
    except HTTPException as e:
        body_out = {"error": str(e.detail)}
        conn.execute(
            "INSERT OR REPLACE INTO idempotency VALUES (?,?,?,?,?)",
            (principal, message_id, msg_hash, json.dumps(body_out), e.status_code),
        )
        conn.commit()
        conn.close()
        raise
    except Exception as e:
        log_error("message_send", e, {"media_type": media_type, "message_id": message_id})
        conn.close()
        raise HTTPException(status_code=500, detail="Internal error processing message")

    conn.execute(
        "INSERT OR REPLACE INTO idempotency VALUES (?,?,?,?,?)",
        (principal, message_id, msg_hash, json.dumps(result), 200),
    )
    conn.commit()
    conn.close()
    return JSONResponse(content=result, media_type="application/a2a+json")


# ---------------- task read / list / cancel ----------------

@a2a.get("/tasks/{task_id}")
def get_task(task_id: str, principal: str = Depends(get_principal), _=Depends(check_version)):
    conn = get_db()
    row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    conn.close()
    if not row or row["principal"] != principal:
        raise HTTPException(status_code=404, detail="Not found")
    return JSONResponse(content=build_task_response(row), media_type="application/a2a+json")


@a2a.get("/tasks")
def list_tasks(principal: str = Depends(get_principal), _=Depends(check_version)):
    conn = get_db()
    rows = conn.execute("SELECT * FROM tasks WHERE principal=?", (principal,)).fetchall()
    conn.close()
    return JSONResponse(
        content={"tasks": [build_task_response(r)["task"] for r in rows]},
        media_type="application/a2a+json",
    )


@a2a.post("/tasks/{task_id}:cancel")
def cancel_task(task_id: str, principal: str = Depends(get_principal), _=Depends(check_version)):
    conn = get_db()
    row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    if not row or row["principal"] != principal:
        conn.close()
        raise HTTPException(status_code=404, detail="Not found")

    if row["state"] == "TASK_STATE_CANCELED":
        conn.close()
        return JSONResponse(content=build_task_response(row), media_type="application/a2a+json")

    cur = conn.execute(
        "UPDATE tasks SET state='TASK_STATE_CANCELED', updated_at=? WHERE id=? AND state='TASK_STATE_INPUT_REQUIRED'",
        (time.time(), task_id),
    )
    conn.commit()
    if cur.rowcount == 0:
        row2 = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        conn.close()
        raise HTTPException(status_code=409, detail=f"Task already {row2['state']}")

    row2 = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    conn.close()
    return JSONResponse(content=build_task_response(row2), media_type="application/a2a+json")


@app.get("/debug/errors")
def debug_errors(secret: Optional[str] = None):
    if secret != DEBUG_SECRET:
        raise HTTPException(status_code=404)
    try:
        with open(ERROR_LOG_PATH) as f:
            return json.load(f)
    except Exception:
        return []


app.include_router(a2a)