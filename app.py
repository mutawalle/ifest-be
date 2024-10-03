from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv('GOOGLE_CREDENTIAL')

from routers import question_router, cv_router, basic_router, vacancy_router


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def verify_token(request: Request, call_next):
    allowed_paths = ["/login", "/", "/cv"]
    
    if request.url.path.startswith("/stream/") or request.url.path in allowed_paths:
        response = await call_next(request)
        return response

    token = request.headers.get("Authorization")
    if not token:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Token required")

    response = await call_next(request)
    return response

app.include_router(basic_router)
app.include_router(vacancy_router)
app.include_router(question_router)
app.include_router(cv_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
