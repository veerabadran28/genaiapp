# embed_pipeline.py

import os
import csv
import json
import uuid
import logging
import datetime
import boto3
import psycopg2
from psycopg2.extras import Json
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# — Config —
CSV_PATH = os.path.join("data", "lseg_dictionary.csv")
DATA_DIR = "data"
DB_CONFIG = {
    "host": "localhost",
    "dbname": "esg_db",
    "user": "postgres",
    "password": "postgres",
    "port": 5432
}
VECTOR_TABLE = "schema_embeddings"
CLAUDE_MODEL = "us.anthropic.claude-3-5-sonnet-20240620-v1:0"
TITAN_MODEL = "amazon.titan-embed-text-v2"

# — AWS clients —
bedrock = boto3.client("bedrock-runtime")

def load_csv():
    rows = []
    with open(CSV_PATH, encoding='latin-1', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "pillar": r["Pillar"].strip(),
                "category": r["Category"].strip(),
                "hierarchy": r["Hierarchy"].strip(),
                "table": r["Table Name"].strip(),
                "field_name": r["LSEG Data Platform Bulk Field Name"].strip(),
                "title": r["Title"].strip(),
                "description": r["Description"].strip()
            })
    logger.info(f"Loaded {len(rows)} rows from CSV")
    return rows

def group_semantic(rows):
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "temperature": 0.0,
        "messages": [
            {
                "role": "user",
                "content": (
                    "You are an expert data architect. "
                    "Group these fields into semantic clusters with topic names and descriptions. "
                    "Respond in JSON as: "
                    "[{\"topic\":..., \"table\":..., \"fields\":[...], \"description\":...}, ...]\n\n"
                    + json.dumps(rows)
                )
            }
        ]
    }
    resp = bedrock.invoke_model(
        modelId=CLAUDE_MODEL,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json"
    )
    result = json.loads(resp["body"].read())
    logger.info(f"Received {len(result)} groups from Claude")
    return result

def save_metadata(groups):
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(DATA_DIR, f"metadata_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(groups, f, indent=2)
    logger.info(f"Saved metadata to {path}")
    return path

def setup_pgvector(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {VECTOR_TABLE} (
                id UUID PRIMARY KEY,
                topic TEXT,
                table_name TEXT,
                description TEXT,
                fields JSONB,
                chunk TEXT,
                embedding vector(768),
                created_at TIMESTAMPTZ DEFAULT now()
            );
        """)
        cur.execute(f"TRUNCATE {VECTOR_TABLE};")
    conn.commit()
    logger.info("pgvector table created/truncated")

def embed_texts(texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(texts).tolist()

def ingest_chunks(metadata_path):
    with open(metadata_path, encoding='utf-8') as f:
        groups = json.load(f)

    chunks = []
    for g in groups:
        chunk = f"Topic: {g['topic']}\nTable: {g['table']}\nDescription: {g.get('description','')}\nFields:\n"
        for fld in g["fields"]:
            chunk += f"- {fld}\n"
        chunks.append({**g, "chunk": chunk})

    texts = [c["chunk"] for c in chunks]
    embeddings = embed_texts(texts)

    conn = psycopg2.connect(**DB_CONFIG)
    setup_pgvector(conn)
    with conn.cursor() as cur:
        for c, emb in zip(chunks, embeddings):
            cur.execute(f"""
                INSERT INTO {VECTOR_TABLE}(id, topic, table_name, description, fields, chunk, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                str(uuid.uuid4()), c["topic"], c["table"],
                c.get("description",""), Json(c["fields"]), c["chunk"], emb
            ))
    conn.commit()
    conn.close()
    logger.info("Inserted embeddings into pgvector")

def main():
    rows = load_csv()
    groups = group_semantic(rows)
    metadata_path = save_metadata(groups)
    ingest_chunks(metadata_path)
    logger.info("Pipeline complete ✅")

if __name__ == "__main__":
    main()
