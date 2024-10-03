from fastapi import APIRouter, Request, HTTPException, status
import json
from config import userCollection, frameCollection, audioCollection, vacancyCollection, questionCollection
import datetime
import os
from const import UPLOAD_DIRECTORY

router = APIRouter(prefix="", tags=["basic"])

@router.get("/")
def hello():
    return "Hello World"

@router.get("/login")
async def login(request: Request):
    try:
        email = json.loads(request.headers.get("Userinfo"))["email"]
        user = userCollection.find_one({"email": email})
        if not user:
            new_user = {"email": email, "created_at": datetime.datetime.now()}
            userCollection.insert_one(new_user)
            return {"message": "New user created"}
        del user["_id"]
        return user
    except Exception as e:
        print(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@router.get("/reset")
async def reset(request: Request):
    try:
       frameCollection.delete_many({})
       audioCollection.delete_many({})
       questionCollection.delete_many({})
       vacancyCollection.delete_many({})
       return {"message": "OK"}
    except:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@router.delete("/delete-all-files/")
async def delete_all_files():

    video_extensions = [".mp4", ".avi"] 
    audio_extensions = [".wav", ".mp3"] 
    deleted_files = []

    for file in os.listdir(UPLOAD_DIRECTORY):
        file_path = UPLOAD_DIRECTORY / file
        if file_path.is_file() and (any(file.endswith(ext) for ext in video_extensions) or any(file.endswith(ext) for ext in audio_extensions)):
            try:
                os.remove(file_path)
                deleted_files.routerend(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
    
    if len(deleted_files) == 0:
        return {"message": "No video or audio files were deleted. No matching files found."}
    
    return {"message": f"Deleted files: {deleted_files}"}