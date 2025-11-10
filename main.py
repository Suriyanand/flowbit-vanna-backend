import os, re, json
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv
import requests

# -------------------------------------------------------
# 🧩 Load .env from the current directory
# -------------------------------------------------------
current_dir = os.path.dirname(__file__)
env_path = os.path.join(current_dir, ".env")
print("🔍 forcing .env from", env_path)
load_dotenv(dotenv_path=env_path, override=True)

DATABASE_URL = os.getenv("DATABASE_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
PORT = int(os.getenv("VANNA_PORT", "8000"))

print("🔍 DATABASE_URL from .env =", DATABASE_URL)

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL missing")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY missing")

# -------------------------------------------------------
# ⚙️ FastAPI setup
# -------------------------------------------------------
app = FastAPI(title="Vanna AI — Natural Language to SQL API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # use your frontend URL in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------
# 🧠 Snapshot DB schema for LLM context
# -------------------------------------------------------
def snapshot_schema() -> str:
    ddl = []
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("""
                SELECT table_name, column_name, data_type
                FROM information_schema.columns
                WHERE table_schema='public'
                ORDER BY table_name, ordinal_position;
            """)
            rows = cur.fetchall()
            tables: Dict[str, List[str]] = {}
            for r in rows:
                tables.setdefault(r["table_name"], []).append(
                    f'{r["column_name"]} {r["data_type"]}'
                )
            for t, cols in tables.items():
                ddl.append(f'{t}({", ".join(cols)})')
    return "\n".join(ddl)

SCHEMA = snapshot_schema()

# -------------------------------------------------------
# 🤖 Groq API (LLM SQL generator)
# -------------------------------------------------------
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
SYS_PROMPT = f"""
You are an expert Postgres SQL generator.
Only output a SINGLE SQL query. Constraints:
- Target ONLY the public schema shown below.
- Use ANSI SQL compatible with PostgreSQL 16.
- NEVER modify data. Only SELECT.
- If dates involved, default to UTC and ISO yyyy-mm-dd.
- Use aggregates when asked for totals/summary.
- LIMIT 500 by default unless the user asked for a small set.
- Approved statuses: DRAFT, PENDING, APPROVED, PAID, OVERDUE, CANCELLED.

Schema:
{SCHEMA}

Return only SQL. No explanation.
"""

def llm_generate_sql(question: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": question},
        ],
        "temperature": 0.1,
    }
    r = requests.post(GROQ_URL, headers=headers, json=body, timeout=60)
    if r.status_code >= 400:
        raise HTTPException(status_code=500, detail=f"Groq error: {r.text}")
    sql = r.json()["choices"][0]["message"]["content"].strip()
    # remove ```sql fences if any
    sql = re.sub(r"^```[a-zA-Z]*", "", sql).strip()
    sql = re.sub(r"```$", "", sql).strip()
    return sql

# -------------------------------------------------------
# 🛡️ Safety & Auto-Fix Helpers
# -------------------------------------------------------
DANGEROUS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|GRANT|REVOKE)\b", re.I
)

def sanitize_sql(sql: str) -> str:
    """Auto-fix common casing issues & ensure SELECT safety."""
    if DANGEROUS.search(sql):
        raise HTTPException(status_code=400, detail="Unsafe SQL blocked. Only SELECT allowed.")
    if not re.search(r"\bSELECT\b", sql, re.I):
        raise HTTPException(status_code=400, detail="Only SELECT queries are allowed.")

    # ✅ Fix casing issues for camelCase columns
    fixes = {
    "vendorid": '"vendorId"',
    "invoiceid": '"invoiceId"',
    "customerid": '"customerId"',
    "paymentid": '"paymentId"',
    "totalamount": '"totalAmount"',
    "totalvalue": '"totalValue"',
    "invoicedate": '"invoiceDate"',
}

    for bad, good in fixes.items():
        sql = re.sub(bad, good, sql, flags=re.I)

    # Add LIMIT if missing
    if not re.search(r"\bLIMIT\b", sql, re.I):
        sql = sql.strip().rstrip(";") + " LIMIT 500"

    return sql

# -------------------------------------------------------
# 🧩 NLQ endpoint
# -------------------------------------------------------
class NLQ(BaseModel):
    question: str

@app.post("/nlq")
def nlq(q: NLQ):
    sql = llm_generate_sql(q.question)
    print("🧠 Generated SQL:", sql)

    # sanitize and fix before execution
    sql = sanitize_sql(sql)

    rows: List[Dict[str, Any]] = []
    cols: List[str] = []
    try:
        with psycopg.connect(DATABASE_URL, options="-c statement_timeout=8000") as conn:
            conn.execute("SET TRANSACTION READ ONLY;")
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql)
                if cur.description:
                    cols = [c.name for c in cur.description]
                    rows = [dict(r) for r in cur.fetchall()]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"SQL error: {str(e)}")

    return {"question": q.question, "sql": sql, "columns": cols, "rows": rows}

# -------------------------------------------------------
# 🚀 Run
# -------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, reload=True)
