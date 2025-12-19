from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
import shutil
import os
import uuid
from logic import BirdAnalyzer

app = FastAPI(title="Poultry Analysis API", description="Detect, Count, and Estimate Weight of Birds")

# Ensure artifacts directory exists
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Initialize Logic Class
analyzer = BirdAnalyzer()

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "BirdCounter-v1"}

@app.post("/analyze_video")
async def analyze_video(
    file: UploadFile = File(...),
    fps_sample: int = Form(30),
    conf_thresh: float = Form(0.3)
):
    """
    Upload a CCTV video to receive bird counts and weight proxies.
    """
    # 1. Save uploaded file safely
    file_id = str(uuid.uuid4())
    input_filename = f"{file_id}_{file.filename}"
    input_path = os.path.join(ARTIFACTS_DIR, input_filename)
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. Define Output Path
    output_filename = f"processed_{input_filename}"
    output_path = os.path.join(ARTIFACTS_DIR, output_filename)

    try:
        # 3. Process Video
        # Note: In a production app, this should be a background task (Celery/RQ).
        # We run it synchronously here for the prototype timebox.
        result_data = analyzer.process_video(
            video_path=input_path, 
            output_path=output_path, 
            fps_sample=fps_sample,
            conf_thresh=conf_thresh
        )

        # 4. cleanup input file to save space (optional)
        if os.path.exists(input_path):
            os.remove(input_path)

        # 5. Return JSON
        return {
            "analysis_id": file_id,
            "status": "completed",
            "video_download_url": f"/download/{output_filename}",
            "data": result_data
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/download/{filename}")
def download_file(filename: str):
    file_path = os.path.join(ARTIFACTS_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="video/mp4", filename=filename)
    return JSONResponse(status_code=404, content={"error": "File not found"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)