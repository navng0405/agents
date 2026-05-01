# Digital Second Brain Agent

Small FastAPI and LangGraph starter project for a personal "second brain" workflow. The stack includes:

- A FastAPI backend on port `8000`
- A local Langfuse deployment on port `3000`
- Postgres, ClickHouse, Redis, and MinIO for supporting services

## What It Does

The backend exposes a simple workflow endpoint at `POST /brain/run`. It accepts a query, runs a minimal LangGraph pipeline, and returns:

- A detected intent
- A small set of retrieved notes
- A generated response
- The workflow steps that ran

## Run The Stack

From the project root:

```bash
docker-compose -f docker-compose.yml up -d --build

## Add Langfuse Keys

Put your Langfuse keys into `backend/.env` before rebuilding the backend:

```env
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key
LANGFUSE_SECRET_KEY=sk-lf-your-secret-key
```

If you are using the local Langfuse container from this stack, keep:

```env
LANGFUSE_HOST=http://localhost:3000
```

After updating the file, restart the backend:

```bash
docker-compose -f docker-compose.yml up -d --build backend
```

Then call `POST /brain/run` and open Langfuse at `http://localhost:3000` to see traces.
```

Useful endpoints:

- Langfuse UI: `http://localhost:3000`
- FastAPI app: `http://localhost:8000`
- FastAPI docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

## Example Request

```bash
curl -s http://localhost:8000/brain/run \
  -H 'Content-Type: application/json' \
  -d '{"query":"Help me plan my note-taking workflow"}'
```

Expected response shape:

```json
{
  "answer": "Intent detected: planning. Based on the notes I found, ...",
  "intent": "planning",
  "retrieved_notes": [
    "Capture ideas in small, reusable notes so they are easier to search and connect later.",
    "Projects stay useful when notes are grouped by outcomes, decisions, and references."
  ],
  "workflow_steps": [
    "analyze_query",
    "retrieve_notes",
    "generate_answer"
  ]
}
```

## Stop The Stack

```bash
docker-compose -f docker-compose.yml down
```
