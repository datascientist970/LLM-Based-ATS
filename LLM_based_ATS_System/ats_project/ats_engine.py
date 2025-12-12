import google.generativeai as genai
from django.conf import settings

genai.configure(api_key=settings.GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash")

def analyze_resume(resume_text, jd_text):
    prompt = f"""
You are an advanced ATS scoring engine with deep reasoning.

Analyze the following RESUME and JOB DESCRIPTION and return:
1. ATS Match Score (0–100)
2. Skills Matched
3. Skills Missing
4. Required Certifications Missing
5. Required Degree or Qualification Missing
6. Experience Gap (if any)
7. Strengths Summary
8. Improvement Recommendations
9. Final Hiring Probability (%)  
10. JSON OUTPUT ONLY.

RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}

Return your response in this exact JSON format:

{{
  "match_score": "",
  "hiring_probability": "",
  "skills_matched": [],
  "skills_missing": [],
  "certifications_missing": [],
  "qualification_missing": "",
  "experience_gap": "",
  "strengths": "",
  "recommendations": ""
}}
"""

    response = model.generate_content(prompt)
    return response.text
