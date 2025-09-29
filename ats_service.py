import os
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional

# PDF reading
import PyPDF2

# For embedding-based keyword similarity
from sentence_transformers import SentenceTransformer, util

# Optional model generation: HF transformers or OpenAI fallback
USE_OPENAI = os.getenv("USE_OPENAI", "") == "1"
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
HF_CHAT_MODEL = os.getenv("HF_CHAT_MODEL", "meta-llama/Llama-2-7b-chat-hf")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")

# Try to import transformers only if not forcing OpenAI
HF_AVAILABLE = False
try:
    if not USE_OPENAI:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False


def _call_openai_chat(prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
    import openai
    openai.api_key = OPENAI_KEY
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")
    # Keep a simple model selection for now
    model = "gpt-4o-mini"
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp["choices"][0]["message"]["content"]


class _HFChatWrapper:
    def __init__(self, model_name: str = HF_CHAT_MODEL):
        self.model_name = model_name
        # NOTE: Loading large Llama models requires sufficient RAM/GPU
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
            offload_folder="offload",
        )
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device_map="auto")

    def chat(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.1) -> str:
        out = self.pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False, temperature=temperature)
        return out[0]["generated_text"]


def _read_pdf_text(path: Path) -> str:
    text_parts = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for p in reader.pages:
            page_text = p.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def _build_prompt(job_text: str, resume_text: str) -> str:
    return f"""
You are an ATS expert and hiring advisor.

1) Briefly summarize the JOB DESCRIPTION (3-6 bullet points): required skills, responsibilities, experience level.
2) Briefly summarize the RESUME (3-6 bullet points): top skills, years of experience, notable projects.
3) Compare the two and output:
   - ATS Score (0-100): compute a numeric match based on required skills present, experience, and role fit.
   - Fit Verdict: one of "Strong Fit", "Partial Fit", "Not a Good Fit". Use thresholds: >=80 Strong; 60-79 Partial; <60 Not a Good Fit.
   - Matched Skills: list of skills that appear in both (normalize names).
   - Missing Skills: required skills from JD that are not present in resume.
   - Feedback: concise, actionable (2-4 sentences).

JOB DESCRIPTION:
{job_text}

RESUME:
{resume_text}

Output strictly as JSON with keys:
{{"Job Summary": "...", "Resume Summary": "...", "ATS Score": number, "Fit Verdict": "...",
 "Matched Skills": [...], "Missing Skills": [...], "Feedback": "..." }}
"""


def _safe_parse_json_like(text: str) -> Dict[str, Any]:
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        substring = text[start:end]
        return json.loads(substring)
    except Exception:
        return {"raw": text}


def _ensure_verdict(score: int) -> str:
    if score >= 80:
        return "Strong Fit"
    if score >= 60:
        return "Partial Fit"
    return "Not a Good Fit"


def process_ats(job_text: str, resume_pdf_path: Path) -> Dict[str, Any]:
    # 1) Read resume PDF
    if not resume_pdf_path.exists():
        raise FileNotFoundError(f"Resume PDF not found: {resume_pdf_path}")
    resume_text = _read_pdf_text(resume_pdf_path)
    if not resume_text.strip():
        raise ValueError("Could not extract text from PDF. Try a different PDF or enable OCR externally.")

    # 2) Validate job text
    job_text = job_text or ""
    if not job_text.strip():
        raise ValueError("No job description provided.")

    # 3) Embedding-based heuristic
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    emb_job = embed_model.encode(job_text, convert_to_tensor=True)
    emb_resume = embed_model.encode(resume_text, convert_to_tensor=True)
    cosine = float(util.cos_sim(emb_job, emb_resume).item())
    base_score = max(0, min(100, int((cosine * 100) * 1.05)))

    # 4) LLM prompt
    prompt = _build_prompt(job_text, resume_text)

    model_output: Optional[str] = None
    if HF_AVAILABLE:
        try:
            hf = _HFChatWrapper(HF_CHAT_MODEL)
            model_output = hf.chat(prompt, max_new_tokens=600, temperature=0.1)
        except Exception:
            model_output = None

    if (model_output is None) and (OPENAI_KEY or USE_OPENAI):
        model_output = _call_openai_chat(prompt, max_tokens=700, temperature=0.1)

    if model_output is None:
        # Fall back to embedding-only result
        final_score = base_score
        verdict = _ensure_verdict(final_score)
        return {
            "Job Summary": "",
            "Resume Summary": "",
            "ATS Score": final_score,
            "Fit Verdict": verdict,
            "Matched Skills": [],
            "Missing Skills": [],
            "Feedback": "Language model unavailable. Score derived from semantic similarity only.",
            "raw_model_output": ""
        }

    parsed = _safe_parse_json_like(model_output)

    parsed_score: Optional[int] = None
    if isinstance(parsed, dict) and "ATS Score" in parsed:
        try:
            parsed_score = int(parsed["ATS Score"])  # type: ignore
        except Exception:
            parsed_score = None

    final_score = parsed_score if parsed_score is not None else base_score
    verdict = parsed.get("Fit Verdict") if isinstance(parsed, dict) else None
    if not verdict:
        verdict = _ensure_verdict(final_score)

    result = {
        "Job Summary": parsed.get("Job Summary") if isinstance(parsed, dict) else "",
        "Resume Summary": parsed.get("Resume Summary") if isinstance(parsed, dict) else "",
        "ATS Score": final_score,
        "Fit Verdict": verdict,
        "Matched Skills": parsed.get("Matched Skills", []) if isinstance(parsed, dict) else [],
        "Missing Skills": parsed.get("Missing Skills", []) if isinstance(parsed, dict) else [],
        "Feedback": parsed.get("Feedback", "") if isinstance(parsed, dict) else model_output,
        "raw_model_output": model_output,
        "embedding_cosine": cosine,
    }
    return result
