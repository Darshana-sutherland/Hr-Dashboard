from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash, jsonify, Response
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
import os
from io import BytesIO
from bson import ObjectId
from config import Config
from mongo_models import HR, Candidate, Job, Application
from gridfs_utils import storage

app = Flask(__name__)
app.config.from_object(Config)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# -------------------- Helper Functions --------------------
def require_hr():
    if 'hr_id' not in session:
        return redirect(url_for('login_hr'))
    return None

def require_candidate():
    if 'candidate_id' not in session:
        return redirect(url_for('login_candidate'))
    return None

# -------------------- Routes --------------------
@app.route('/')
def index():
    return render_template('index.html')

# -------------------- Auth Routes --------------------
@app.route('/login/hr', methods=['GET', 'POST'])
def login_hr():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        hr_user = HR.find_by_username(username)
        if hr_user and check_password_hash(hr_user['password_hash'], password):
            session['hr_id'] = str(hr_user['_id'])
            return redirect(url_for('hr_dashboard'))
        flash('Invalid username or password', 'danger')
    return render_template('login_hr.html')

@app.route('/login/candidate', methods=['GET', 'POST'])
def login_candidate():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        candidate = Candidate.find_by_username(username)
        if candidate and check_password_hash(candidate['password_hash'], password):
            session['candidate_id'] = str(candidate['_id'])
            return redirect(url_for('candidate_dashboard'))
        flash('Invalid username or password', 'danger')
    return render_template('login_candidate.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# -------------------- HR Portal Routes --------------------
@app.route('/hr/dashboard')
def hr_dashboard():
    guard = require_hr()
    if guard:
        return guard
    
    # Get counts for dashboard
    total_candidates = db.candidates.count_documents({})
    open_positions = db.jobs.count_documents({})
    in_interview = db.applications.count_documents({"status": "Interview"})
    hired = db.applications.count_documents({"status": "Hired"})
    
    # Get status counts for the chart
    status_counts = {}
    pipeline = [
        {"$group": {"_id": "$status", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    
    for doc in db.applications.aggregate(pipeline):
        status_counts[doc['_id']] = doc['count']
    
    # Get recent applications
    recent = list(db.applications.find().sort("created_at", -1).limit(5))
    
    return render_template('hr_dashboard.html',
                         total_candidates=total_candidates,
                         open_positions=open_positions,
                         in_interview=in_interview,
                         hired=hired,
                         status_counts=status_counts,
                         recent=recent)

# Add other HR routes (hr_resumes, hr_candidates, hr_jobs, etc.) following the same pattern
# ...

# -------------------- Candidate Portal Routes --------------------
@app.route('/candidate/dashboard')
def candidate_dashboard():
    guard = require_candidate()
    if guard:
        return guard
        
    user = Candidate.find_by_id(session['candidate_id'])
    jobs = Job.find_all()
    my_apps = Application.find_with_details({"candidate_id": session['candidate_id']})
    
    return render_template('candidate_dashboard.html',
                         user=user,
                         jobs=jobs,
                         my_apps=my_apps)

@app.route('/jobs')
def public_jobs():
    jobs = Job.find_all()
    return render_template('jobs_public.html', jobs=jobs)

@app.route('/apply/<job_id>', methods=['GET', 'POST'])
def apply(job_id):
    if 'candidate_id' not in session:
        return redirect(url_for('login_candidate'))
    
    job = Job.find_by_id(job_id)
    if not job:
        flash('Job not found', 'danger')
        return redirect(url_for('public_jobs'))
    
    if request.method == 'POST':
        # Check if file was uploaded
        if 'resume' not in request.files:
            flash('No file uploaded', 'danger')
            return redirect(request.url)
        
        file = request.files['resume']
        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            # Save file to GridFS
            file_id = storage.save_file(
                file=file,
                filename=secure_filename(file.filename),
                content_type=file.content_type,
                metadata={
                    'candidate_id': session['candidate_id'],
                    'job_id': job_id,
                    'uploaded_at': datetime.utcnow()
                }
            )
            
            # Create application
            application_data = {
                'candidate_id': session['candidate_id'],
                'job_id': job_id,
                'status': 'New',
                'resume_file_id': file_id,
                'created_at': datetime.utcnow()
            }
            
            Application.create(application_data)
            flash('Application submitted successfully', 'success')
            return redirect(url_for('candidate_dashboard'))
    
    return render_template('apply.html', job=job)

# Add other routes as needed...

if __name__ == '__main__':
    app.run(debug=True)
