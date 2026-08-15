import os
import json
import re
import io

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from google import genai
from dotenv import load_dotenv

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_NAME = "gemini-3.5-flash-lite"

client = genai.Client(api_key=GEMINI_API_KEY)


def extract_aadhaar_data(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))

        image.thumbnail(
            (1024, 1024),
            Image.Resampling.LANCZOS
        )

        prompt = """
        Extract the following details from this Aadhaar card image:

        1. Name (Full name)
        2. Aadhaar Number (12-digit number)
        3. Date of Birth (in DD/MM/YYYY format)
        4. Gender (Male/Female/Other)
        5. Address (Complete address)

        Return ONLY valid JSON with these exact keys:

        {
            "name": "",
            "aadhaar_number": "",
            "date_of_birth": "",
            "gender": "",
            "address": ""
        }

        If any field is not found, leave it as an empty string.
        Do not add any additional text or explanation.
        """

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                prompt,
                image
            ]
        )

        response_text = response.text

        print(f"Gemini response: {response_text[:500]}")

        json_match = re.search(
            r'\{.*\}',
            response_text,
            re.DOTALL
        )

        if json_match:
            return json.loads(json_match.group())

        return json.loads(response_text)

    except Exception as e:
        print(f"Gemini error: {repr(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Gemini processing error: {str(e)}"
        )


@app.get("/")
async def root():
    return {
        "message": "Aadhaar Extractor API is running!",
        "model": MODEL_NAME,
        "api_key_configured": bool(GEMINI_API_KEY)
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "api_key_configured": bool(GEMINI_API_KEY)
    }


@app.post("/extract-aadhaar")
async def extract_aadhaar(
    file: UploadFile = File(...)
):
    if not file.content_type:
        raise HTTPException(
            status_code=400,
            detail="Missing content type"
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )

    contents = await file.read()

    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File size exceeds 5MB limit"
        )

    return {
        "success": True,
        "model_used": MODEL_NAME,
        "data": extract_aadhaar_data(contents)
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚀 Starting Aadhaar Extractor Server...")
    print("="*60)
    print(f"🤖 Using model: {MODEL_NAME}")
    print(f"🔑 API Key loaded: {'✅ Yes' if GEMINI_API_KEY else '❌ No'}")
    print("📍 Server will run at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("📋 List Models: http://localhost:8000/available-models")
    print("="*60)
    print("\nPress Ctrl+C to stop the server\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)