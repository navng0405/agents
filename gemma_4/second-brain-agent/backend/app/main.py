from fastapi import FastAPI
from langfuse.decorators import langfuse_context, observe

from app.api.schemas import BrainRequest, BrainResponse
from app.core.config import settings
from app.graph.workflow import run_second_brain_workflow
from app.services.langfuse_client import get_langfuse


app = FastAPI(title=settings.PROJECT_NAME)


@app.get("/")
async def read_root() -> dict[str, str]:
    return {
        "message": f"{settings.PROJECT_NAME} is running",
        "docs": "/docs",
        "health": "/health",
        "brain_endpoint": "/brain/run",
    }


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.on_event("startup")
async def startup_event() -> None:
    get_langfuse()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    langfuse = get_langfuse()
    langfuse.flush()
    langfuse.shutdown()


@observe(name="brain_run_request")
@app.post("/brain/run", response_model=BrainResponse)
async def run_brain(request: BrainRequest) -> BrainResponse:
    langfuse_context.update_current_trace(
        name="brain_run",
        metadata={"request": request.model_dump()},
    )
    langfuse_context.update_current_observation(input=request.model_dump())
    result = run_second_brain_workflow(request.query)
    response = BrainResponse(
        answer=result["answer"],
        intent=result["intent"],
        retrieved_notes=result.get("retrieved_notes", []),
        workflow_steps=result.get("workflow_steps", []),
    )
    langfuse_context.update_current_observation(output=response.model_dump())
    return response
