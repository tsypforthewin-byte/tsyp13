import whisper
from fastapi import FastAPI, File, UploadFile, Form
import tempfile
import os
import json
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai
import io
import base64
from gtts import gTTS


# Whisper
whisper_model = whisper.load_model("base")
client_genai = genai.Client(api_key="AIzaSyCceD0FNj2T9hh9GvqdCCrtioZOLW4GPTc")
chat = client_genai.chats.create(model="gemini-2.0-flash")

print(" Gemini chargé.")
# Embeddings + Chroma
client = chromadb.Client()
try:
    collection = client.get_collection("cv_chunks")
except:
    collection = client.create_collection("cv_chunks")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def text_to_base64_audio(text: str, lang: str = "en") -> str:
    """
    Convert text to speech and return base64-encoded MP3.
    """
    tts = gTTS(text=text, lang=lang, slow=False)
    buffer = io.BytesIO()
    tts.write_to_fp(buffer)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def extract_cv_chunks(cv):
    chunks = []
    for proj in cv.get("projects", []):
        chunks.append(f"Projet : {proj['name']}. {proj['description']}")
    for cert in cv.get("certifications", []):
        chunks.append(f"Certification : {cert['name']} délivrée par {cert['issuer']}")
    for skill in cv.get("skills_and_interests", {}).get("technical_skills", []):
        chunks.append(f"Compétence : {skill}")
    edu = cv.get("education", {})
    if edu:
        chunks.append(f"Éducation : {edu.get('degree')} à {edu.get('school')}")
    return chunks


def analyze_cv_file(cv_data):
    cv_chunks = extract_cv_chunks(cv_data)
    embeddings = embedding_model.encode(cv_chunks)

    collection.add(
        documents=cv_chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(cv_chunks))],
    )

    print(" ChromaDB initialisé avec", len(cv_chunks), "chunks.")


def transcrire_audio(path):
    result = whisper_model.transcribe(path, language="en")
    os.remove(path)
    return result["text"].strip()


def retrieve_context(user_answer, top_k=3):
    if not user_answer.strip():
        return ""

    embedding = embedding_model.encode([user_answer])
    results = collection.query(query_embeddings=embedding, n_results=top_k)
    return "\n".join(results["documents"][0])


def analyser_reponse(texte):
    texte_low = texte.lower().strip()

    if len(texte_low) < 3:
        return "vide"

    vague_words = ["i don't know", "i'm not sure", "no idea"]
    confuse = ["huh", "what", "pardon"]

    if any(w in texte_low for w in vague_words):
        return "vague"

    if any(w in texte_low for w in confuse):
        return "confuse"

    return "bonne"


from pydantic import BaseModel
from typing import List, Dict, Optional

app = FastAPI()


class CVModel(BaseModel):
    projects: Optional[List[Dict]] = []
    certifications: Optional[List[Dict]] = []
    skills_and_interests: Optional[Dict] = {}
    education: Optional[Dict] = {}


class HistoryItem(BaseModel):
    question: str
    answer: str


class GreetingResponse(BaseModel):
    text: str
    audio_base64: Optional[str] = None


class StartResponse(BaseModel):
    question: str
    history: list[HistoryItem]
    audio_base64: Optional[str] = None


class AnswerResponse(BaseModel):
    end: bool
    transcript: Optional[str] = None
    status: Optional[str] = None
    next_question: Optional[str] = None
    history: Optional[list[HistoryItem]] = None
    audio_base64: Optional[str] = None


class StopResponse(BaseModel):
    message: str


@app.get("/interview/greeting", response_model=GreetingResponse)
def interview_greeting():
    greeting_text = "Hello! Welcome to this interview. Are you ready to start?"
    audio_base64 = text_to_base64_audio(greeting_text, lang="en")

    return GreetingResponse(text=greeting_text, audio_base64=audio_base64)


@app.post("/interview/start", response_model=StartResponse)
def interview_start(cv: CVModel):
    import json

    analyze_cv_file(cv.model_dump())
    cv_json_str = json.dumps(cv.model_dump(), ensure_ascii=False)

    intro_prompt = f"""
You are a professional recruiter.
Here is the candidate's CV: {cv_json_str}

Start by asking the first question about a key project.
"""

    q = chat.send_message(intro_prompt).text.strip()
    audio_base64 = text_to_base64_audio(q, lang="en")
    return StartResponse(question=q, history=[], audio_base64=audio_base64)


from fastapi import UploadFile, Form
import tempfile


@app.post("/interview/answer", response_model=AnswerResponse)
async def interview_answer(
    file: UploadFile = File(...),
    history: Optional[str] = Form(None),
    last_question: Optional[str] = Form(None),
):

    try:
        history_list = json.loads(history) if history else []
        history_items = [HistoryItem(**h) for h in history_list]
    except json.JSONDecodeError:
        history_items = []

    # Save uploaded audio temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Transcribe with Whisper
    result = whisper_model.transcribe(tmp_path, language="en")
    os.remove(tmp_path)

    answer = result["text"].strip()

    # Stop condition
    if answer.lower() in ["stop", "quit", "fin"]:
        return AnswerResponse(end=True)

    # Analyse
    status = analyser_reponse(answer)
    contextual_data = retrieve_context(answer)

    # Update history
    history_items.append(HistoryItem(question=last_question, answer=answer))
    history_dicts = [h.dict() for h in history_items]
    history_json = json.dumps(history_dicts, ensure_ascii=False)
    # Generate next question
    next_prompt = f"""
You are an AI recruiter.
Here is the interview history: {history_json}
Analyze the candidate's response:
- status: {status}
- relevant CV context: {contextual_data}

RULES:
1. If the answer is "empty", reply: "I didn’t hear that, could you repeat please?"
2. If "vague", simplify the question.
3. If "confused", clarify gently.
4. If "good", ask a new question coherent with CV, previous answer, full history.
5. Never ask two questions at once.
6. No repetitions, no thanks. Only the next question.

Next question:
"""

    q = chat.send_message(next_prompt).text.strip()

    # Generate TTS for next question
    audio_base64 = text_to_base64_audio(q, lang="en")

    return AnswerResponse(
        end=False,
        transcript=answer,
        status=status,
        next_question=q,
        history=history_items,
        audio_base64=audio_base64,
    )


@app.post("/interview/stop", response_model=StopResponse)
def stop():
    return StopResponse(message="Entretien terminé.")
