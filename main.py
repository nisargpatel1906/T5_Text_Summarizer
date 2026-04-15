from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
import os
import uvicorn

# Define app
app = FastAPI(title="T5 Text Summarizer API")

# Define location context paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_summarizer_model")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Create static directory if it doesn't exist
os.makedirs(STATIC_DIR, exist_ok=True)

# Initialize FastAPI app
model = None
tokenizer = None
device = None

import re

print(f"Loading custom T5 model from {MODEL_PATH}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
    device = torch.device("cuda") # Enforcing GPU strictly
    model.to(device)
    model.eval() # Set model to evaluation mode
    print(f"Model successfully loaded on {device}!")
except Exception as e:
    print(f"Failed to load model from {MODEL_PATH}. Error: {e}")

def clean_data(text):
    text = str(text)
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.strip()
    return text

class SummaryRequest(BaseModel):
    text: str
    max_length: int = 150
    min_length: int = 40

class SummaryResponse(BaseModel):
    summary: str

@app.post("/api/summarize", response_model=SummaryResponse)
async def summarize_text(req: SummaryRequest):
    if not model or not tokenizer:
        raise HTTPException(status_code=500, detail="The T5 model is not loaded correctly.")
    
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        # Preprocess text precisely like the training loop did!
        cleaned_text = clean_data(req.text)
        
        # Prepend 'summarize: ' because T5 often requires it
        input_text = "summarize: " + cleaned_text 
        
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
        inputs = inputs.to(device)

        summary_ids = model.generate(
            inputs,
            max_length=req.max_length,
            min_length=req.min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return SummaryResponse(summary=summary)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static folder
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_home():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

if __name__ == "__main__":
    print("Starting server at http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
