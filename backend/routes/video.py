import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Body
from fastapi.responses import FileResponse
from src.vision.video_processor import FootballVideoProcessor
from src.tactical.video_analyzer import VideoTacticalAnalyzer
from src.core.types import VideoAnalysisRequest, VideoAnalysisResponse, VideoMetadata

router = APIRouter(tags=["Video Processing"])
processor = FootballVideoProcessor()
tactical_analyzer = VideoTacticalAnalyzer()
UPLOAD_DIR = Path("uploads")
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
MAX_FILE_SIZE_MB = 500


@router.post("/video/upload")
async def upload_video(file: UploadFile = File(...)):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed formats: MP4, MOV, AVI, MKV."
        )

    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file_size_mb = round(file_path.stat().st_size / (1024 * 1024), 2)
    if file_size_mb > MAX_FILE_SIZE_MB:
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=f"File size ({file_size_mb} MB) exceeds maximum allowed limit of {MAX_FILE_SIZE_MB} MB."
        )

    metadata = processor.get_video_metadata(str(file_path))

    return {
        "status": "success",
        "message": "Video uploaded successfully. Ready for automatic tactical analysis.",
        "video_metadata": metadata.model_dump()
    }


@router.post("/video/analyze", response_model=VideoAnalysisResponse)
def analyze_video(request: VideoAnalysisRequest):
    path = Path(request.file_path)
    if not path.exists():
        # Fallback search inside UPLOAD_DIR
        alt_path = UPLOAD_DIR / path.name
        if alt_path.exists():
            path = alt_path
        else:
            raise HTTPException(status_code=404, detail=f"Uploaded video file not found at '{request.file_path}'. Please upload first.")

    metadata = processor.get_video_metadata(str(path))
    output_annotated_path = UPLOAD_DIR / f"annotated_{path.name}"

    try:
        tactical_frames = processor.process_and_annotate_video(
            video_path=str(path),
            output_path=str(output_annotated_path),
            max_frames=request.max_frames
        )
    except Exception as e:
        print(f"[Backend /video/analyze] Error during CV video processing: {e}")
        tactical_frames = []

    annotated_filename = f"annotated_{path.name}"
    annotated_url = f"/uploads/{annotated_filename}" if output_annotated_path.exists() else None
    download_url = f"/api/video/download/{annotated_filename}" if output_annotated_path.exists() else None

    analysis_result = tactical_analyzer.analyze_video_sequence(
        frames=tactical_frames,
        metadata=metadata,
        annotated_video_url=annotated_url,
        download_url=download_url
    )

    return analysis_result


@router.get("/video/download/{filename}")
def download_video(filename: str):
    """Exposes real downloadable MP4 file generated from uploaded video."""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        # Try finding without annotated_ prefix if needed
        alt = UPLOAD_DIR / f"annotated_{filename}"
        if alt.exists():
            file_path = alt
        else:
            raise HTTPException(status_code=404, detail=f"Processed video file '{filename}' not found.")

    return FileResponse(
        path=str(file_path),
        media_type="video/mp4",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@router.post("/video/process")
def process_video(file_path: str, max_frames: int = 150):
    """Legacy process endpoint maintained for backward compatibility."""
    if not Path(file_path).exists():
        alt = UPLOAD_DIR / Path(file_path).name
        if alt.exists():
            file_path = str(alt)
        else:
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    frames = processor.process_video(file_path, max_frames=max_frames)
    return {
        "status": "success",
        "processed_frames": len(frames),
        "tracking": [f.model_dump() for f in frames[:10]]
    }

