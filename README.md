Recruitment Portal (Flask + SQLite)

Run locally on Windows PowerShell

1) Create/activate venv and install deps

```
cd C:\Users\Administrator\Desktop\temp-main
python -m venv venv
./venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
pip install Flask Flask_SQLAlchemy Werkzeug
```

2) Start the app

```
python app.py
```

The server starts at `http://127.0.0.1:5000`.

Seed credentials (auto-created on first run)
- HR: username `hr`, password `hr123`
- Candidate: username `cand`, password `cand123`

Basic flow
- Landing page → choose HR or Candidate
- HR login → HR Dashboard and Resume Database (download resumes)
- Candidate login → Candidate Dashboard → Apply Now → fill name, email, phone, upload resume → application stored and visible to HR

Files uploaded to `uploads/` and metadata stored in `app.db` (SQLite).

