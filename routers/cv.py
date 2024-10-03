from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
import pdfplumber
from io import BytesIO
from config import model
import json

router = APIRouter(prefix="", tags=["cv"])

@router.post("/cv")
async def analyze_cv(file: UploadFile = File(...), job_title: str = Form(...), description: str = Form(...)):
    try:
        content = await file.read()

        with pdfplumber.open(BytesIO(content)) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        prompt = """Berikut adalah text dari sebuah Resume. {text}. 
                Berikan analisis kesesuaian Resume tersebut berdasarkan lowongan pekerjaan ini. nama lowongan: {job_title} dan deskripsi: {description}. 
                Berikan jawaban yang mengandung 
                    summary: Tinjauan singkat tentang seberapa baik resume.
                    jobKeywords: Daftar kata kunci yang ideal untuk nama lowongan tersebut.
                    resumeKeywords: Daftar kata kunci yang ditemukan dalam resume kandidat.
                    RelevanceScore: Skor yang menunjukkan seberapa baik pengalaman kandidat selaras dengan persyaratan pekerjaan.
                    quantifiedScore: Skor yang menunjukkan tingkat kuantifikasi dalam resume (misalnya, menggunakan metrik untuk mengukur pencapaian).
                    improvement: list saran tentang bagaimana kandidat dapat meningkatkan resume mereka dan alasannya.
                Berikan jawaban dalam format json sebagai berikut {{"summary": "long text", "jobKeywords": ["satu", "dua",...], "resumeKeywords": ["satu", "dua",...], RelevanceScore: number, quantifiedScore: number, improvement: ["satu", "dua",...]}} tanpa tambahan karakter apapun termasuk ```json```"""
        formatted = prompt.format(text=text, job_title=job_title, description=description)
        response = model.generate_content([formatted])
        return json.loads(response.text)
    except:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)