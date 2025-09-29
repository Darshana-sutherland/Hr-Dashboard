# AI-Powered Resume Screening API

This is a FastAPI-based backend service that provides AI-powered resume screening capabilities. It uses Sentence Transformers (MiniLM) to calculate semantic similarity between job descriptions and resumes, and provides detailed skill matching.

## Features

- Extract text from PDF and DOCX resumes
- Calculate semantic similarity between job descriptions and resumes
- Extract and match skills from both documents
- Generate a detailed matching report with scores and analysis
- RESTful API endpoints for easy integration

## Prerequisites

- Python 3.8+
- pip (Python package manager)

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repository-url>
   cd backend
   ```

2. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install spaCy's English language model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Running the API

1. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

2. The API will be available at `http://127.0.0.1:8000`

3. Access the interactive API documentation at `http://127.0.0.1:8000/docs`

## API Endpoints

### Screen a Resume

- **URL**: `/api/screen-resume`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `job_description` (string): The job description text
  - `resume_file` (file): The resume file (PDF or DOCX)

**Example Request**:
```bash
curl -X POST "http://localhost:8000/api/screen-resume" \
  -H "accept: application/json" \
  -F "job_description=Looking for a Python developer with Django experience" \
  -F "resume_file=@/path/to/resume.pdf"
```

**Example Response**:
```json
{
  "match_score": 78.5,
  "skill_matches": [
    {
      "skill": "python",
      "match": true,
      "relevance": 100
    },
    {
      "skill": "django",
      "match": true,
      "relevance": 100
    },
    {
      "skill": "machine learning",
      "match": false,
      "relevance": 0
    }
  ],
  "summary": "This candidate has a good match with the job requirements. The resume matches 2 out of 3 key skills mentioned in the job description.",
  "skill_analysis": {
    "total_skills": 3,
    "matched_skills": 2,
    "match_percentage": 66.67
  }
}
```

## Integration with Frontend

To integrate this with your frontend:

1. Make a POST request to `/api/screen-resume` with form data containing the job description and resume file.
2. Handle the response to display the match score, skill matches, and summary to the user.

## Performance

The model (`all-MiniLM-L6-v2`) is optimized for speed while maintaining good accuracy. On a standard CPU, it can process a typical resume in 1-2 seconds.

## Customization

- **Skills List**: Modify the `common_skills` list in `resume_screener.py` to include domain-specific skills.
- **Matching Thresholds**: Adjust the thresholds in the `generate_summary` method to change how matches are categorized.
- **Model**: You can use a different SentenceTransformer model by changing the `model_name` when initializing the `ResumeScreener` class.

## License

This project is licensed under the MIT License.
