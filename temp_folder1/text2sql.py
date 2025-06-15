import os, csv, json, uuid, logging, boto3, psycopg2
from psycopg2.extras import Json
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths + DB
CSV_PATH = os.path.join("data", "lseg_dictionary.csv")
META_JSON = os.path.join("data", "metadata.json")
VECTOR_TABLE = "schema_embeddings"
DB_CONFIG = {
    "host":"localhost", "dbname":"esg_db",
    "user":"postgres", "password":"…", "port":5432
}

# AWS Bedrock clients
bedrock = boto3.client("bedrock-runtime")
TITAN_MODEL = "amazon.titan-embed-text-v2"
CLAUDE_MODEL = "anthropic.claude-3-5-sonnet-20241022-v2:0"

def load_csv():
    rows=[]
    with open(CSV_PATH, encoding="utf-8", newline="") as f:
        reader=csv.DictReader(f)
        for r in reader:
            rows.append({k: r[k].strip() for k in r})
    logger.info(f"{len(rows)} metadata rows loaded")
    with open(META_JSON, "w") as fout: json.dump(rows, fout, indent=2)
    return rows

def group_semantic(rows):
    # Construct conversation payload for Claude
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "temperature": 0.2,
        "messages": [
            {
                "role": "user",
                "content": (
                    "You are an expert schema analyst. "
                    "Given the following metadata list (JSON array), group "
                    "Field Names into semantic clusters and return JSON "
                    "[{\"topic\":\"...\",\"fields\":[...],\"description\":\"...\"}, ...].\n\n"
                    f"Metadata=\n{json.dumps(rows, indent=2)}"
                )
            }
        ]
    }

    # Correct Bedrock invocation
    response = bedrock.invoke_model(
        modelId=CLAUDE_MODEL,
        body=json.dumps(request_body),
        contentType="application/json",
        accept="application/json"
    )

    result_body = response["body"].read()
    grouped = json.loads(result_body)
    logger.info(f"Received {len(grouped)} semantic groups from Claude")
    return grouped

def chunk_embeddings(groups):
    chunks = []
    for g in groups:
        chunk = f"Topic: {g['topic']}\nFields:\n" + "\n".join(
            [f"{f}: {next(r['Description (d)'] for r in rows if r['LSEG Data Platform Bulk Field Name']==f)}" for f in g["fields"]])
        chunks.append({**g, "chunk": chunk})
    return chunks

def setup_pgvector(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(f"""
CREATE TABLE IF NOT EXISTS {VECTOR_TABLE} (
  id UUID PRIMARY KEY,
  topic TEXT,
  description TEXT,
  chunk TEXT,
  fields JSONB,
  embedding vector(768),
  created_at TIMESTAMPTZ DEFAULT now()
);""")
    conn.commit()

def embed_titan(texts):
    resp = bedrock.invoke_model(modelId=TITAN_MODEL,
        body=json.dumps({"inputText": texts}),
        contentType="application/json")
    return json.loads(resp["body"].read())["embeddings"]

def main():
    global rows
    rows = load_csv()

    groups = group_semantic(rows)

    chunks = chunk_embeddings(groups)
    conn = psycopg2.connect(**DB_CONFIG)
    setup_pgvector(conn)

    texts = [c["chunk"] for c in chunks]
    embeddings = embed_titan(texts)

    with conn.cursor() as cur:
        for c, emb in zip(chunks, embeddings):
            cur.execute(f"""
INSERT INTO {VECTOR_TABLE}(id, topic, description, chunk, fields, embedding)
VALUES (%s,%s,%s,%s,%s,%s)
""",(str(uuid.uuid4()), c["topic"], c.get("description",""),
     c["chunk"], Json(c["fields"]), emb))
    conn.commit()
    conn.close()
    logger.info("Completed embedding ingestion into pgvector.")

if __name__ == "__main__":
    main()
