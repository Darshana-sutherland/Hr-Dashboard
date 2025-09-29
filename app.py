from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
from datetime import datetime


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev-secret-change'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

db = SQLAlchemy(app)


class HR(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)

    def set_password(self, raw_password: str) -> None:
        self.password_hash = generate_password_hash(raw_password)

    def check_password(self, raw_password: str) -> bool:
        return check_password_hash(self.password_hash, raw_password)


class Candidate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    name = db.Column(db.String(120))
    email = db.Column(db.String(120))
    phone = db.Column(db.String(30))

    def set_password(self, raw_password: str) -> None:
        self.password_hash = generate_password_hash(raw_password)

    def check_password(self, raw_password: str) -> bool:
        return check_password_hash(self.password_hash, raw_password)


class Job(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    company = db.Column(db.String(200), nullable=False)
    tags = db.Column(db.String(200), default='')


class Application(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    candidate_id = db.Column(db.Integer, db.ForeignKey('candidate.id'), nullable=False)
    job_id = db.Column(db.Integer, db.ForeignKey('job.id'), nullable=False)
    status = db.Column(db.String(50), default='New')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    resume_filename = db.Column(db.String(255))

    candidate = db.relationship('Candidate', backref=db.backref('applications', lazy=True))
    job = db.relationship('Job', backref=db.backref('applications', lazy=True))


@app.cli.command('init-db')
def init_db_command():
    db.drop_all()
    db.create_all()
    # seed HR
    hr = HR(username='hr')
    hr.set_password('hr123')
    db.session.add(hr)

    # seed candidates
    cand = Candidate(username='cand')
    cand.set_password('cand123')
    cand.name = 'Sample Candidate'
    cand.email = 'cand@example.com'
    cand.phone = '1234567890'
    db.session.add(cand)

    # seed jobs
    job = Job(title='Senior Frontend Developer', company='Company XYZ', tags='React,TypeScript')
    db.session.add(job)

    db.session.commit()
    print('Database initialized with sample data.')


@app.route('/')
def index():
    return render_template('index.html')


# -------------------- Auth --------------------
@app.route('/login/hr', methods=['GET', 'POST'])
def login_hr():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        hr = HR.query.filter_by(username=username).first()
        if hr and hr.check_password(password):
            session.clear()
            session['hr_id'] = hr.id
            return redirect(url_for('hr_dashboard'))
        flash('Invalid HR credentials', 'danger')
    return render_template('login_hr.html')


@app.route('/login/candidate', methods=['GET', 'POST'])
def login_candidate():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = Candidate.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session.clear()
            session['candidate_id'] = user.id
            return redirect(url_for('candidate_dashboard'))
        flash('Invalid Candidate credentials', 'danger')
    return render_template('login_candidate.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


# -------------------- HR Portal --------------------
def require_hr():
    if 'hr_id' not in session:
        return redirect(url_for('login_hr'))
    return None


@app.route('/hr')
def hr_dashboard():
    guard = require_hr()
    if guard:
        return guard
    total_candidates = Candidate.query.count()
    open_positions = Job.query.count()
    hired = Application.query.filter_by(status='Hired').count()
    in_interview = Application.query.filter_by(status='Interview').count()
    recent = Application.query.order_by(Application.created_at.desc()).limit(5).all()
    status_counts = {
        'New': Application.query.filter_by(status='New').count(),
        'Interview': in_interview,
        'Hired': hired,
        'Rejected': Application.query.filter_by(status='Rejected').count(),
        'Reviewed': Application.query.filter_by(status='Reviewed').count(),
    }
    return render_template(
        'hr_dashboard.html',
        total_candidates=total_candidates,
        open_positions=open_positions,
        hired=hired,
        in_interview=in_interview,
        recent=recent,
        status_counts=status_counts,
    )


@app.route('/hr/resumes')
def hr_resumes():
    guard = require_hr()
    if guard:
        return guard
    apps = Application.query.order_by(Application.created_at.desc()).all()
    return render_template('hr_resumes.html', applications=apps)


STATUS_OPTIONS = ['New', 'Interview', 'Reviewed', 'Hired', 'Rejected']


@app.route('/hr/candidates')
def hr_candidates():
    guard = require_hr()
    if guard:
        return guard
    apps = Application.query.order_by(Application.created_at.desc()).all()
    return render_template('hr_candidates.html', applications=apps, status_options=STATUS_OPTIONS)


@app.route('/hr/jobs')
def hr_jobs():
    guard = require_hr()
    if guard:
        return guard
    jobs = Job.query.order_by(Job.id.desc()).all()
    return render_template('hr_jobs.html', jobs=jobs)


@app.route('/hr/screening', methods=['GET', 'POST'])
def hr_screening():
    guard = require_hr()
    if guard:
        return guard
        
    if request.method == 'POST':
        # Handle AJAX submission: job_description + resume PDF
        job_description = request.form.get('job_description', '')
        resume_file = request.files.get('resume')

        if not job_description or not resume_file or not resume_file.filename:
            # For AJAX use JSON error; for non-AJAX, flash
            if request.accept_mimetypes.best == 'application/json':
                return jsonify({"error": "Please provide both job description and resume PDF."}), 400
            flash('Please provide both job description and resume', 'danger')
            return redirect(url_for('hr_screening'))

        # Only allow PDF for now
        if not resume_file.filename.lower().endswith('.pdf'):
            if request.accept_mimetypes.best == 'application/json':
                return jsonify({"error": "Only PDF resumes are supported at the moment."}), 400
            flash('Only PDF resumes are supported at the moment.', 'danger')
            return redirect(url_for('hr_screening'))

        filename = secure_filename(resume_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        resume_file.save(filepath)

        # Run ATS processing
        try:
            from pathlib import Path
            from ats_service import process_ats
            result = process_ats(job_description, Path(filepath))
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Get recent applications for the table
    apps = Application.query.order_by(Application.created_at.desc()).limit(10).all()
    return render_template('hr_screening.html', applications=apps)


@app.post('/hr/application/<int:app_id>/status')
def hr_update_status(app_id: int):
    guard = require_hr()
    if guard:
        return guard
    new_status = request.form.get('status')
    app_row = Application.query.get_or_404(app_id)
    if new_status in STATUS_OPTIONS:
        app_row.status = new_status
        db.session.commit()
        flash('Status updated', 'success')
    else:
        flash('Invalid status', 'danger')
    # redirect back to candidates page by default
    return redirect(request.referrer or url_for('hr_candidates'))


@app.route('/uploads/<path:filename>')
def download_resume(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


# -------------------- Candidate Portal --------------------
def require_candidate():
    if 'candidate_id' not in session:
        return redirect(url_for('login_candidate'))
    return None


@app.route('/candidate')
def candidate_dashboard():
    guard = require_candidate()
    if guard:
        return guard
    user = Candidate.query.get(session['candidate_id'])
    jobs = Job.query.all()
    my_apps = Application.query.filter_by(candidate_id=user.id).order_by(Application.created_at.desc()).all()
    return render_template('candidate_dashboard.html', user=user, jobs=jobs, my_apps=my_apps)


@app.route('/jobs')
def public_jobs():
    jobs = Job.query.all()
    return render_template('jobs_public.html', jobs=jobs)


@app.route('/apply/<int:job_id>', methods=['GET', 'POST'])
def apply(job_id: int):
    job = Job.query.get_or_404(job_id)
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')

        # Find or create candidate by email
        user = Candidate.query.filter_by(email=email).first()
        if not user:
            username = email or f"user{datetime.utcnow().timestamp()}"
            user = Candidate(username=username)
            user.set_password(os.urandom(8).hex())
            user.name = name
            user.email = email
            user.phone = phone
            db.session.add(user)
            db.session.flush()  # get id
        else:
            user.name = name or user.name
            user.phone = phone or user.phone

        resume_file = request.files.get('resume')
        filename = None
        if resume_file and resume_file.filename:
            safe_name = secure_filename(f"{user.username}_{job.id}_{resume_file.filename}")
            resume_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
            resume_file.save(resume_path)
            filename = safe_name

        app_row = Application(candidate_id=user.id, job_id=job.id, status='New', resume_filename=filename)
        db.session.add(app_row)
        db.session.commit()
        # flash('Application submitted successfully.', 'success')
        return redirect(url_for('index'))

    return render_template('apply.html', job=job)


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        # Auto-seed minimal data on first run for convenience
        if HR.query.count() == 0:
            hr = HR(username='hr')
            hr.set_password('hr123')
            db.session.add(hr)
        if Job.query.count() == 0:
            job = Job(title='Senior Frontend Developer', company='Company XYZ', tags='React,TypeScript')
            db.session.add(job)
        db.session.commit()
    app.run(debug=True)


