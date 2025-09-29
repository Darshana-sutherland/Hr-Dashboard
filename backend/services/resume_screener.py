from typing import Dict, List, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
import spacy

class ResumeScreener:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the resume screener with a pre-trained sentence transformer model.
        
        Args:
            model_name: Name of the SentenceTransformer model to use.
                       'all-MiniLM-L6-v2' is a good balance between speed and accuracy.
        """
        self.model = SentenceTransformer(model_name)
        self.nlp = spacy.load("en_core_web_sm")
        
    def extract_skills(self, text: str) -> List[str]:
        """
        Extract skills from text using NLP.
        This is a basic implementation that can be enhanced with a custom NER model.
        """
        # Common skills to look for (can be expanded)
        common_skills = [
            'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin',
            'django', 'flask', 'react', 'angular', 'vue', 'node.js', 'express', 'spring',
            'machine learning', 'deep learning', 'ai', 'data analysis', 'data science',
            'sql', 'postgresql', 'mysql', 'mongodb', 'redis', 'aws', 'docker', 'kubernetes',
            'git', 'rest api', 'graphql', 'agile', 'scrum', 'devops', 'ci/cd'
        ]
        
        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Find skills in the text
        found_skills = []
        for skill in common_skills:
            if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
                found_skills.append(skill)
                
        return found_skills
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        # Encode the texts to get their embeddings
        embeddings = self.model.encode([text1, text2])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(
            embeddings[0].reshape(1, -1),
            embeddings[1].reshape(1, -1)
        )[0][0]
        
        # Convert to percentage (0-100)
        return float(similarity * 100)
    
    def analyze_skill_match(self, job_description: str, resume_text: str) -> List[Dict[str, Any]]:
        """Analyze skill matches between job description and resume."""
        # Extract skills from both texts
        job_skills = set(self.extract_skills(job_description))
        resume_skills = set(self.extract_skills(resume_text))
        
        # Find matching skills
        matching_skills = job_skills.intersection(resume_skills)
        missing_skills = job_skills - resume_skills
        
        # Calculate skill match percentage
        skill_match_percentage = (len(matching_skills) / len(job_skills)) * 100 if job_skills else 0
        
        # Prepare detailed skill matches
        skill_matches = []
        
        # For matching skills
        for skill in matching_skills:
            skill_matches.append({
                'skill': skill,
                'match': True,
                'relevance': 100  # Since it's a match
            })
            
        # For missing skills
        for skill in missing_skills:
            skill_matches.append({
                'skill': skill,
                'match': False,
                'relevance': 0
            })
            
        return {
            'skill_matches': skill_matches,
            'match_percentage': skill_match_percentage,
            'total_skills': len(job_skills),
            'matched_skills': len(matching_skills)
        }
    
    def generate_summary(self, job_description: str, resume_text: str, match_score: float) -> str:
        """Generate a human-readable summary of the match."""
        skill_analysis = self.analyze_skill_match(job_description, resume_text)
        
        if match_score >= 80:
            strength = "an excellent"
        elif match_score >= 60:
            strength = "a good"
        elif match_score >= 40:
            strength = "a moderate"
        else:
            strength = "a poor"
            
        summary = (
            f"This candidate has {strength} match with the job requirements. "
            f"The resume matches {skill_analysis['matched_skills']} out of {skill_analysis['total_skills']} "
            f"key skills mentioned in the job description."
        )
        
        return summary
    
    def screen_resume(self, job_description: str, resume_text: str) -> Dict[str, Any]:
        """
        Screen a resume against a job description.
        
        Args:
            job_description: The job description text
            resume_text: The resume text to screen
            
        Returns:
            Dict containing match score, skill matches, and summary
        """
        # Calculate overall similarity
        match_score = self.calculate_similarity(job_description, resume_text)
        
        # Analyze skill matches
        skill_analysis = self.analyze_skill_match(job_description, resume_text)
        
        # Generate summary
        summary = self.generate_summary(job_description, resume_text, match_score)
        
        return {
            'match_score': round(match_score, 2),
            'skill_matches': skill_analysis['skill_matches'],
            'summary': summary,
            'skill_analysis': {
                'total_skills': skill_analysis['total_skills'],
                'matched_skills': skill_analysis['matched_skills'],
                'match_percentage': round(skill_analysis['match_percentage'], 2)
            }
        }

# Example usage
if __name__ == "__main__":
    screener = ResumeScreener()
    
    # Example job description and resume text
    job_desc = """
    We are looking for a Python developer with experience in web development.
    The ideal candidate should have experience with Django, REST APIs, and databases.
    Knowledge of machine learning is a plus.
    """
    
    resume = """
    Experienced Python developer with 3 years of experience in web development.
    Worked with Django, Flask, and FastAPI frameworks. Strong knowledge of REST APIs
    and database design. Familiar with machine learning concepts and libraries.
    """
    
    result = screener.screen_resume(job_desc, resume)
    print(f"Match Score: {result['match_score']}%")
    print(f"Summary: {result['summary']}")
    print("\nSkill Matches:")
    for skill in result['skill_matches']:
        status = "✓" if skill['match'] else "✗"
        print(f"{status} {skill['skill']}")
