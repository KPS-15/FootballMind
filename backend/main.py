import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.routes import match_router, video_router, simulation_router


app = FastAPI(
    title="FootballMind API",
    description="Multimodal Deep Learning Framework for Predictive, Tactical and Explainable Football Intelligence",
    version="0.1.0"
)

# Mount uploads directory for static video access
uploads_dir = Path("uploads")
uploads_dir.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Enable CORS for Next.js Dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(match_router, prefix="/api")
app.include_router(video_router, prefix="/api")
app.include_router(simulation_router, prefix="/api")



@app.get("/")
def root():
    return {
        "status": "online",
        "system": "FootballMind AI Framework",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
