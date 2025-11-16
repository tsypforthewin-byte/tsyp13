from langchain_community.vectorstores import Chroma

from langchain_community.embeddings import HuggingFaceEmbeddings
import re
from google.cloud import firestore
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from uuid import uuid4

CHROMA_PATH = "chroma_db"


db = firestore.Client(
    project="tsyp-477814",
    database="tsypstore",  # replace with your database ID (e.g., "(default)" or another)
)
# ### using all-MiniLM-L6-v2


model_name = "intfloat/e5-large-v2"
embedding_function = HuggingFaceEmbeddings(model_name=model_name)

vector_store = Chroma(
    persist_directory="chroma_jobs_db",
    collection_name="intfloat_e5_large_v2",
    embedding_function=embedding_function,
)


def preprocess_text(text):
    text = text.lower()
    text = re.sub("[^a-z]", " ", text)  # remove punctuation
    text = re.sub(r"\d+", "", text)  # remove numbers
    text = " ".join(text.split())  # remove extra spaces
    return text


def build_job_text(job):
    return f"""
  Job Title: {job['title']}
  Company: {job['company']}
  Description: {job['company_description']}
  Requirements: {job['requirements']}
  Skills:{', '.join(job['skills'])}
  Location:{job['location']['city']},{job['location']['country']}
  Contract:{job['contract_type']}
  Experience:{job['experience_level']}
  Education:{job['education_level']}
  """


def cv_to_text(cv: dict) -> str:
    """Convert a CV JSON object into a single text string for embeddings."""

    text_parts = []

    # Personal information
    personal = cv.get("personal_information", {})
    name = personal.get("full_name", "")
    email = personal.get("email", "")
    phone = personal.get("phone", "")
    text_parts.append(f"{name}\nEmail: {email}\nPhone: {phone}")

    # Website and social links
    links = cv.get("website_and_social_links", {})
    linkedin = links.get("linkedin")
    github = links.get("github")
    portfolio = links.get("portfolio")
    link_texts = []
    if linkedin:
        link_texts.append(f"LinkedIn: {linkedin}")
    if github:
        link_texts.append(f"GitHub: {github}")
    if portfolio:
        link_texts.append(f"Portfolio: {portfolio}")
    if link_texts:
        text_parts.append("\n".join(link_texts))

    # Education
    education_list = cv.get("education", [])
    edu_texts = []
    for edu in education_list:
        degree = edu.get("degree", "")
        field = edu.get("field_of_study", "")
        school = edu.get("school", "")
        location = edu.get("location", "")
        years = f"{edu.get('start_year','')} - {edu.get('end_year','')}".strip(" -")
        edu_texts.append(f"{degree} in {field} from {school}, {location} ({years})")
    if edu_texts:
        text_parts.append("Education:\n" + "\n".join(edu_texts))

    # Work experience
    work_list = cv.get("work_experience", [])
    work_texts = []
    for work in work_list:
        title = work.get("job_title", "")
        company = work.get("company", "")
        location = work.get("location", "")
        dates = f"{work.get('start_date','')} - {work.get('end_date','')}".strip(" -")
        responsibilities = work.get("responsibilities", [])
        responsibilities_text = (
            "\n- " + "\n- ".join(responsibilities) if responsibilities else ""
        )
        work_texts.append(
            f"{title} at {company}, {location} ({dates}){responsibilities_text}"
        )
    if work_texts:
        text_parts.append("Work Experience:\n" + "\n".join(work_texts))

    # Certifications
    cert_list = cv.get("certification", [])
    cert_texts = [f"{c.get('name')} ({c.get('issue_date','')})" for c in cert_list]
    if cert_texts:
        text_parts.append("Certifications:\n" + "\n".join(cert_texts))

    # Projects
    projects = cv.get("projects", [])
    proj_texts = [f"{p.get('name')}: {p.get('description')}" for p in projects]
    if proj_texts:
        text_parts.append("Projects:\n" + "\n".join(proj_texts))

    # Skills and interests
    skills = cv.get("skills_and_interests", {})
    technical = ", ".join(skills.get("technical_skills", []))
    soft = ", ".join(skills.get("soft_skills", []))
    hobbies = ", ".join(skills.get("hobbies_and_interests", []))
    skill_texts = []
    if technical:
        skill_texts.append(f"Technical Skills: {technical}")
    if soft:
        skill_texts.append(f"Soft Skills: {soft}")
    if hobbies:
        skill_texts.append(f"Hobbies & Interests: {hobbies}")
    if skill_texts:
        text_parts.append("\n".join(skill_texts))

    # Volunteering
    volunteering = cv.get("volunteering", [])
    vol_texts = []
    for v in volunteering:
        org = v.get("organization", "")
        role = v.get("role", "")
        vol_texts.append(f"{role} at {org}")
    if vol_texts:
        text_parts.append("Volunteering:\n" + "\n".join(vol_texts))

    # Combine everything
    full_text = "\n\n".join(text_parts)
    return full_text


# --- your imports ---
# from your_module import build_job_text, cv_to_text, preprocess_text, embedding_function, vector_store

app = FastAPI(title="Job Matching API", version="1.0")


# ----------- MODELS -----------


class Location(BaseModel):
    city: str
    country: str
    remote: bool


class Job(BaseModel):
    id: str
    external_reference: str
    title: str
    company: str
    company_description: str
    industry: str
    location: Location
    contract_type: str
    experience_level: str
    education_level: str
    skills: List[str]
    salary_min: float
    salary_max: float
    salary_currency: str
    salary_period: str
    language: str
    posted_at: str
    expires_at: str
    description: str
    requirements: str
    recruitment_process: str
    apply_url: str
    status: str


class CV(BaseModel):
    cv_json: Dict[str, Any]
    user_id: str


# ----------- ENDPOINTS -----------


@app.post("/store_jobs")
def store_job_embeddings(jobs: List[Job]):
    try:
        for job in jobs:
            job_dict = job.dict()
            job_text = build_job_text(job_dict)
            vector_store.add_texts(
                texts=[job_text],
                metadatas=[
                    {
                        "job_id": job.id,
                        "title": job.title,
                        "company": job.company,
                        "industry": job.industry,
                        "location": f"{job.location.city}, {job.location.country}",
                        "remote": job.location.remote,
                        "skills": ", ".join(job.skills),
                        "contract_type": job.contract_type,
                    }
                ],
                ids=[str(uuid4())],
            )
        vector_store.persist()
        return {
            "status": "success",
            "count": len(jobs),
            "message": "Jobs stored in Chroma",
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error storing job embeddings: {e}"
        )


print("Connected to:", db.project)


@app.post("/match_jobs")
def match_jobs(cv: CV):
    try:
        # Convert and embed CV
        cv_text = cv_to_text(cv.cv_json)
        cv_text = preprocess_text(cv_text)
        cv_emb = embedding_function.embed_query(cv_text)

        # Perform similarity search
        results = vector_store.similarity_search_by_vector(cv_emb, k=10)

        matched = [
            {
                "job_id": r.metadata.get("job_id"),
                "title": r.metadata.get("title"),
                "company": r.metadata.get("company"),
                "industry": r.metadata.get("industry"),
                "location": r.metadata.get("location"),
                "remote": r.metadata.get("remote"),
                "skills": r.metadata.get("skills"),
                "score": getattr(r, "score", None),
            }
            for r in results
        ]

        # ----------------------------
        # Save to Firestore
        # ----------------------------
        user_ref = db.collection("users").document(cv.user_id)
        matches_ref = user_ref.collection("job_matches")

        # create a new document in subcollection (keeps history)
        matches_ref.add(
            {
                "cv_text": cv_text,
                "matches": matched,
                "count": len(matched),
                "timestamp": firestore.SERVER_TIMESTAMP,
            }
        )

        # ----------------------------
        # Return response
        # ----------------------------
        return {"matches": matched, "count": len(matched)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error matching jobs: {e}")


@app.delete("/clean_db")
def clean_db():
    try:
        try:
            all_ids = vector_store.get().get("ids", [])
        except Exception as e:
            all_ids = []
            print("⚠️ Could not fetch IDs:", e)

        if all_ids:
            vector_store.delete(ids=all_ids)
            return {"status": "success", "deleted_count": len(all_ids)}
        return {"status": "success", "message": "Collection already empty"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cleaning database: {e}")


# Optional health check
@app.get("/health")
def health_check():
    return {"status": "ok"}
