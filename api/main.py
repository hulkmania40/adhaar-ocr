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


def extract_transaction_data(file_bytes, filename):
    try:
        fname = filename.lower()

        if fname.endswith(".pdf"):
            try:
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(file_bytes))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
            except ImportError:
                raise HTTPException(
                    status_code=500,
                    detail="PDF processing requires pypdf. Install with: pip install pypdf"
                )
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to read PDF: {str(e)}"
                )
        else:
            try:
                image = Image.open(io.BytesIO(file_bytes))
                image.thumbnail(
                    (2048, 2048),
                    Image.Resampling.LANCZOS
                )
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported file format. Upload an image or PDF."
                )

            prompt = """
            Extract ALL bank transactions from this bank statement image.

            For each transaction, identify:
            - date: transaction date in YYYY-MM-DD format
            - amount: numeric amount (without currency symbols)
            - type: "credit" if money is deposited/received, "debit" if money is withdrawn/paid
            - name: payee/beneficiary name or transaction description

            Return ONLY valid JSON — a JSON array of objects:
            [
                {
                    "date": "2024-01-15",
                    "amount": 2500.00,
                    "type": "debit",
                    "name": "Amazon India"
                },
                ...
            ]

            Rules:
            - If a field cannot be determined, use an empty string or 0 for amount.
            - Include every transaction row you can read from the statement.
            - Do not include a "balance" or "closing balance" row.
            - Do not add any additional text or explanation outside the JSON array.
            """

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[prompt, image]
            )

            response_text = response.text
            print(f"Gemini response: {response_text[:500]}")

            json_match = re.search(
                r'\[.*\]',
                response_text,
                re.DOTALL
            )

            if json_match:
                return json.loads(json_match.group())

            return json.loads(response_text)

        if not text or not text.strip():
            raise HTTPException(
                status_code=400,
                detail="Could not extract any text from the PDF. It may be a scanned/image-based PDF."
            )

        prompt = f"""
        Extract ALL bank transactions from the following bank statement text.

        For each transaction, identify:
        - date: transaction date in YYYY-MM-DD format
        - amount: numeric amount (without currency symbols)
        - type: "credit" if money is deposited/received, "debit" if money is withdrawn/paid
        - name: payee/beneficiary name or transaction description

        Return ONLY valid JSON — a JSON array of objects:
        [
            {{
                "date": "2024-01-15",
                "amount": 2500.00,
                "type": "debit",
                "name": "Amazon India"
            }},
            ...
        ]

        Rules:
        - If a field cannot be determined, use an empty string or 0 for amount.
        - Include every transaction row you can read from the statement.
        - Do not include a "balance" or "closing balance" row.
        - Do not add any additional text or explanation outside the JSON array.

        Statement text:
        ---
        {text}
        ---
        """

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt]
        )

        response_text = response.text
        print(f"Gemini response: {response_text[:500]}")

        json_match = re.search(
            r'\[.*\]',
            response_text,
            re.DOTALL
        )

        if json_match:
            return json.loads(json_match.group())

        return json.loads(response_text)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Gemini error: {repr(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Gemini processing error: {str(e)}"
        )


@app.get("/")
async def root():
    return {
        "message": "Document Extractor API is running!",
        "model": MODEL_NAME,
        "api_key_configured": bool(GEMINI_API_KEY),
        "endpoints": {
            "/extract-aadhaar": "POST - upload an image to extract Aadhaar card details",
            "/extract-transactions": "POST - upload an image or PDF to extract bank transactions"
        }
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


@app.post("/extract-transactions")
async def extract_transactions(
    file: UploadFile = File(...)
):
    if not file.content_type:
        raise HTTPException(
            status_code=400,
            detail="Missing content type"
        )

    allowed_types = ["image/jpeg", "image/png", "image/webp", "image/gif", "application/pdf"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed: images and PDF."
        )

    contents = await file.read()

    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File size exceeds 10MB limit"
        )

    return {
        "success": True,
        "model_used": MODEL_NAME,
        "data": extract_transaction_data(contents, file.filename)
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚀 Starting Document Extractor Server...")
    print("="*60)
    print(f"🤖 Using model: {MODEL_NAME}")
    print(f"🔑 API Key loaded: {'✅ Yes' if GEMINI_API_KEY else '❌ No'}")
    print("📍 Server will run at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("📋 Aadhaar Endpoint: POST http://localhost:8000/extract-aadhaar")
    print("💳 Transaction Endpoint: POST http://localhost:8000/extract-transactions")
    print("="*60)
    print("\nPress Ctrl+C to stop the server\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)