from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import tempfile
from pathlib import Path

from services.resume_screener import ResumeScreener
from services.document_processor import DocumentProcessor

app = FastAPI(title="AI Resume Screener API")

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
resume_screener = ResumeScreener()
document_processor = DocumentProcessor()

class ScreeningRequest(BaseModel):
    job_description: str
    resume_text: str

class ScreeningResponse(BaseModel):
    match_score: float
    skill_matches: List[Dict[str, Any]]
    summary: str

@app.post("/api/screen-resume", response_model=ScreeningResponse)
async def screen_resume(
    job_description: str,
    resume_file: UploadFile = File(...)
):
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(resume_file.filename).suffix) as temp_file:
            content = await resume_file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Extract text from the uploaded file
            resume_text = document_processor.extract_text(temp_file_path)
            
            # Get screening results
            result = resume_screener.screen_resume(job_description, resume_text)
            
            return {
                "match_score": result["match_score"],
                "skill_matches": result["skill_matches"],
                "summary": result["summary"]
            }
            
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "AI Resume Screening API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
