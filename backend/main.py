# Image Classification + Deployment

# 1. Use Fastapi to create an API for the image classification model.
# 2. Use OpenAI's API to generate a description of the image based on the classification results.
# 3. Get structured data from the OpenAI API based on the classification results.
# 4. Make a frontend interface to display the results to the user.
# 5. Deploy the frontend on github pages and backend on _________.

import base64
import os
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from PIL import Image
from io import BytesIO

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), base_url="https://aipipe.org/openai/v1"
)

MAX_IMAGE_SIZE_BYTES = 1_000_000

def compress_image_under_limit(image_bytes: bytes, max_bytes: int = MAX_IMAGE_SIZE_BYTES) -> bytes:
    try:
        image = Image.open(BytesIO(image_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc

    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")

    # Always send JPEG to control final payload size reliably.
    if image.mode == "L":
        image = image.convert("RGB")

    quality_levels = [85, 75, 65, 55, 45, 35, 25]
    min_dimension = 128

    while image.width >= min_dimension and image.height >= min_dimension:
        for quality in quality_levels:
            buffer = BytesIO()
            image.save(buffer, format="JPEG", optimize=True, quality=quality)
            compressed = buffer.getvalue()
            if len(compressed) < max_bytes:
                return compressed

        # If quality alone is not enough, progressively shrink dimensions.
        new_width = max(min_dimension, int(image.width * 0.85))
        new_height = max(min_dimension, int(image.height * 0.85))
        if (new_width, new_height) == image.size:
            break
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    raise HTTPException(
        status_code=413,
        detail="Unable to compress image below 1 MB. Please upload a smaller image.",
    )

class ImageClassificationResponseFormat(BaseModel):
    category: str
    explanation: str
    confidence: float

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/classify")
async def classify_image(
    file: UploadFile = File(...),
    input_text: str = "Classify the image and provide a description.",
):
    original_image_bytes = await file.read()
    print(f"Original image size: {len(original_image_bytes)} bytes")
    compressed_image_bytes = compress_image_under_limit(original_image_bytes)
    print(f"Compressed image size: {len(compressed_image_bytes)} bytes")
    image = base64.b64encode(compressed_image_bytes).decode("utf-8")


    response = client.chat.completions.parse(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": "You are an image classification assistant. You will receive an image in base64 format and a prompt.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": input_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                    },
                ],
            },
        ],
        response_format=ImageClassificationResponseFormat
    )

    parsed = response.choices[0].message.parsed

    print("Category: " + parsed.category)
    print("Confidence: " + str(parsed.confidence))
    print("Explanation: " + parsed.explanation)

    return {
        "filename": file.filename,
        "classification": parsed,
    }
