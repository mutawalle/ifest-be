from fastapi import APIRouter, HTTPException, status, Request, BackgroundTasks, UploadFile, File, Form, Response
from config import questionCollection, frameCollection, audioCollection, model
import json
from collections import Counter
from pathlib import Path
from const import UPLOAD_DIRECTORY
import uuid
import datetime
from utils.analyze import analyze, analyzeCode
import os
from google.cloud import storage

router = APIRouter(prefix="", tags=["question"])

@router.get("/question/{id}")
async def get_question_by_id(id: str):
    try:
        question = questionCollection.find_one({"id": id})
        del question["_id"]
        return question
    except:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
@router.get("/question-result/{id}")
async def get_question_result(id: str):
    try:
        question = questionCollection.find_one({"id": id})
        frame = frameCollection.find_one({"id": id})
        audio = audioCollection.find_one({"id": id})
        if not question:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question not found")
        
        if question["category"] == "technical":
            prompt = """Berikut adalah pertanyaan teknikcal dan jawaban baik yang diucapkan dan kode programnya dari sebuah interview. pertanyaan: {question}. kode: {code} jawaban: {answer}.
                    Berikan analisis berdasarkan jawaban tersebut.
                    Berikan jawaban yang mengandung 
                        summary: Tinjauan singkat tentang seberapa baik jawaban.
                        improvement: list saran tentang bagaimana kandidat dapat meningkatkan jawaban mereka dan alasannya.
                        relevance: seberapa sesuai jawaban dalam angka 0-1.
                        clarity: seberapa jelas kalimat yang digunakan dalam angka 0-1.
                        originality: originilitas dalam angka 0-1.
                    Berikan jawaban dalam format json sebagai berikut {{"summary": "long text", improvement: ["satu", "dua",...], "relevance": 0.1, "clarity": 0.6, "originality": 0.4}} tanpa tambahan karakter apapun termasuk ```json```"""
            formatted = prompt.format(question=question["question"], answer=question["answer"])
            response = model.generate_content([formatted])
            responseJson = json.loads(response.text)
            return {
                "question": question["question"],
                "answer": question["answer"],
                "summary": responseJson["summary"],
                "improvement": responseJson["improvement"],
                "relevance": responseJson["relevance"],
                "clarity": responseJson["clarity"],
                "originality": responseJson["originality"]
            }
        else:
            if not frame:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Frame not found")
            if not audio:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Audio not found")
            prompt = """Berikut adalah pertanyaan dan jawaban dari sebuah interview. pertanyaan: {question}. jawaban: {answer}.
                    Berikan analisis berdasarkan jawaban tersebut.
                    Berikan jawaban yang mengandung 
                        summary: Tinjauan singkat tentang seberapa baik jawaban.
                        improvement: list saran tentang bagaimana kandidat dapat meningkatkan jawaban mereka dan alasannya.
                        relevance: seberapa sesuai jawaban dalam angka 0-1.
                        clarity: seberapa jelas kalimat yang digunakan dalam angka 0-1.
                        originality: originilitas dalam angka 0-1.
                    Berikan jawaban dalam format json sebagai berikut {{"summary": "long text", improvement: ["satu", "dua",...], "relevance": 0.1, "clarity": 0.6, "originality": 0.4}} tanpa tambahan karakter apapun termasuk ```json```"""
            formatted = prompt.format(question=question["question"], answer=question["answer"])
            response = model.generate_content([formatted])
            responseJson = json.loads(response.text)
            averageHand = sum(frame["hands"]) / len(frame["hands"])*4*10
            prompt2 = f"Seorang melakukan wawancara dengan rata-rata perubahan gesture tangan {averageHand}cm per detik. beri nilai 0-1 dalam angka tanpa ada tambahan karakter apapun termasuk enter, titik, dsb"
            response2 = model.generate_content([prompt2])

            emotionTexts = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
            emotion_frequency = Counter(emotionTexts)
            emotion_frequency_dict = dict(emotion_frequency)

            return {
                "question": question["question"],
                "answer": question["answer"],
                "summary": responseJson["summary"],
                "improvement": responseJson["improvement"],
                "relevance": responseJson["relevance"],
                "clarity": responseJson["clarity"],
                "originality": responseJson["originality"],
                "engagement": float(response2.text),
                "emotion": emotion_frequency_dict, 
                "body": frame["hands"],
                "voice": audio["snr"],
                "result": question["result"]
            }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@router.get("/questions-by-vacancy/{id}")
async def get_questions_by_vacancy_id(id: str):
    try:
        questions = questionCollection.find({"vacancy_id": id})
        listQuestions = list(questions)
        for question in listQuestions:
            del question["_id"]
        return {"questions": listQuestions}
    except:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
@router.get("/questions")
async def get_questions(request: Request):
    try:
        token = request.headers.get("Authorization")
        if token:
            token = token.split("Bearer ")[1]
            email = json.loads(request.headers.get("Userinfo"))["email"]
            questions = questionCollection.find({"email": email, "status": { "$in": ["SUCCESS", "UPLOADED"]}})
            listQuestions = list(questions)
            for question in listQuestions:
                del question["_id"]
            return {"questions": listQuestions}
        else:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Token required")
    except:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
@router.post("/question")
async def add_question(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...), question: str = Form(...), category: str = Form(...)):
    try:
        token = request.headers.get("Authorization")
        if token:
            token = token.split("Bearer ")[1]
            email = json.loads(request.headers.get("Userinfo"))["email"]
            newUuid = uuid.uuid4().hex

            response = model.generate_content([f"berikan contoh jawaban untuk pertanyaan wawancara ini. {question}"])

            file_extension = Path(file.filename).suffix
            file_location = UPLOAD_DIRECTORY / f"{newUuid}{file_extension}"

            with open(file_location, "wb") as f:
                f.write(await file.read())

            questionCollection.insert_one({
                "id": newUuid,
                "email": email,
                "vacancy_id": "-",
                "question": question,
                "category": category,
                "example_answer": response.text,
                "status": "UPLOADED",
                "messages": ["no video", "uploaded"],
                "created_at": datetime.datetime.now()
            })


            return {"message": "uploaded and analyzing started"}
        else:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Token required")
    except:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
@router.post("/answer-question")
async def answer_question(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...), id: str = Form(...), code: str = Form(...)):
    try:
        token = request.headers.get("Authorization")
        if token:
            token = token.split("Bearer ")[1]
            email = json.loads(request.headers.get("Userinfo"))["email"]
            
            file_extension = Path(file.filename).suffix
            file_location = UPLOAD_DIRECTORY / f"{id}{file_extension}"
            
            with open(file_location, "wb") as f:
                f.write(await file.read())

            question = questionCollection.find_one({"id": id})
            messages = question["messages"]
            messages.append("uploaded")
            question["messages"] = messages        
            questionCollection.find_one_and_update({"id": id},{ "$set": { "messages": messages, "status": "UPLOADED"}})



            return {"message": "uploaded and analyzing started"}
        else:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Token required")
    except:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    

@router.get("/stream/{video_name}")
async def stream_video(video_name: str, request: Request):
    bucket = storage.Client().bucket(os.getenv('BUCKET_NAME'))
    blob = bucket.blob(video_name + ".mp4")
    blob.reload()
    
    byte_range = request.headers.get("range")
    
    if byte_range:
        start, end = byte_range.replace("bytes=", "").split("-")
        start = int(start)
        end = int(end) if end else None
        print(start, end)

        file_size = blob.size
        if end is None:
            end = file_size - 1

        if start >= file_size or end >= file_size:
            raise HTTPException(status_code=416, detail="Requested Range Not Satisfiable")

        file_data = blob.download_as_bytes(start=start, end=end + 1) 

        # Set headers for partial content response
        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(len(file_data)),
            "Content-Type": "video/mp4",
        }

        return Response(content=file_data, status_code=206, headers=headers)
    
    file_data = blob.download_as_bytes()
    file_size = len(file_data)
    
    headers = {
        "Content-Length": str(file_size),
        "Content-Type": "video/mp4",
    }

    return Response(content=file_data, headers=headers)