"""FastAPI entrypoint for the CrossMill MegaForge OpenEnv environment.

Exposes the standard OpenEnv HTTP API:
    GET  /health     - liveness probe
    GET  /metadata   - environment metadata (name / description / version)
    GET  /info       - alias for /metadata (judge-friendly endpoint name)
    GET  /state      - current internal environment state
    GET  /schema     - JSON Schemas for action / observation / state
    POST /reset      - reset env, return first observation
    POST /step       - apply an action, return next obs + reward + done
    WS   /ws         - persistent WebSocket session

Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8002
or:
    python -m app.main
"""

from __future__ import annotations

import os
import sys

# When uvicorn is launched with `cwd` set to the repo root, ensure the repo
# itself is on sys.path so `import app.environment` resolves correctly even
# without an installed package or PYTHONPATH tweaks.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from openenv.core.env_server import create_app  # noqa: E402

from app.environment import MegaForgeEnv  # noqa: E402
from app.models import Action, Observation  # noqa: E402

app = create_app(
    MegaForgeEnv,
    Action,
    Observation,
    env_name="crossmill-megaforge",
)


# ---------------------------------------------------------------------------
# Judge-friendly aliases on top of the OpenEnv-generated FastAPI app.
# ---------------------------------------------------------------------------
@app.get("/info", tags=["Environment Info"], summary="Get environment info")
def info() -> dict:
    """Lightweight environment-info endpoint (alias of /metadata).

    Returns the same payload as /metadata so judge-facing demos can hit a
    short, intuitive URL.
    """
    env = MegaForgeEnv(task_id="easy", seed=0)
    try:
        meta = env.get_metadata()
        if hasattr(meta, "model_dump"):
            return meta.model_dump()
        return dict(meta)
    finally:
        env.close()


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8002"))
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
