"""Small native Sakuga utilities that used to live in sd-webui-misc-scripts."""

import threading

from fastapi import Request

from modules import script_callbacks

_enqueue_queue = []
_enqueue_lock = threading.Lock()


def _sd_models():
    from modules import sd_models

    return sd_models


def on_app_started(_demo, app):
    @app.post("/sakuga/enqueue")
    async def enqueue_prompt(request: Request):
        try:
            data = await request.json()
        except Exception:
            return {"status": "error", "message": "Invalid JSON"}

        with _enqueue_lock:
            _enqueue_queue.append(data)

        return {"status": "queued", "queue_size": len(_enqueue_queue)}

    @app.post("/sakuga/enqueue/next")
    async def dequeue_prompt():
        with _enqueue_lock:
            if not _enqueue_queue:
                return {"job": None, "queue_size": 0}
            job = _enqueue_queue.pop(0)
            return {"job": job, "queue_size": len(_enqueue_queue)}

    @app.get("/sakuga/enqueue/status")
    async def enqueue_status():
        with _enqueue_lock:
            return {"queue_size": len(_enqueue_queue)}

    @app.post("/sakuga/models/unload")
    async def unload_models():
        try:
            _sd_models().unload_model_weights()
            return {"status": "success"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @app.post("/sakuga/models/list")
    async def list_models():
        try:
            _sd_models().list_loaded_weights()
            return {"status": "success"}
        except Exception as e:
            return {"status": "error", "message": str(e)}


def register():
    script_callbacks.on_app_started(on_app_started)
