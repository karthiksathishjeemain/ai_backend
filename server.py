import docx
import asyncio
import numpy as np
import json
from sentence_transformers import SentenceTransformer, util
from groq import AsyncGroq
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# -------------------------------
# Initialize Models and Groq Llama
# -------------------------------

# Load the embedding model (free model: all-MiniLM-L6-v2) for semantic similarity calculations
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Initialize the OpenAI API client with API key from environment variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------------
# FastAPI App and Request Models
# -------------------------------

# Initialize FastAPI application
app = FastAPI()

# Enable Cross-Origin Resource Sharing (CORS) for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Define request model for meeting agenda generation
class MeetingRequest(BaseModel):
    meeting_description: str

# Define request model for processing meeting transcription
class ProcessRequest(BaseModel):
    meeting_description: str
    agenda: list
    transcript: str = None  # Optional transcript input

# -------------------------------
# Helper Functions
# -------------------------------

def load_transcription(doc_path: str) -> str:
    """Load transcription text from a Word document."""
    doc = docx.Document(doc_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def chunk_text(text: str, num_chunks: int = 4) -> list:
    """Split the text into `num_chunks` parts for real-time processing simulation."""
    words = text.split()
    chunk_size = len(words) // num_chunks
    return [" ".join(words[i * chunk_size:(i + 1) * chunk_size]) for i in range(num_chunks)]

# -------------------------------
# LLM Call Functions Using Groq Llama
# -------------------------------

async def generate_agenda(meeting_description: str) -> list:
    """Generate a structured agenda based on the provided meeting description."""
    system_prompt = """
You are a meeting organizer who needs to generate an agenda based on the topics discussed in the conversation.
Identify and list only the relevant topics as a JSON array of strings.
For example:
Input: "Discuss sales strategy, new product launch, and customer feedback analysis."
Expected Output: ["Introduction", "Product Features", "Pricing"]
You need to generate between 4 to 6 agenda topics.
IMPORTANT: Respond with only the JSON array, without any extra text or explanation.
"""
    
    prompt = f'{meeting_description}'
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Using GPT-4o-mini model for efficiency
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    )
    
    # Convert response to JSON array
    agenda = json.loads(response.choices[0].message.content)
    return agenda

async def extract_topics(chunk: str) -> list:
    """Extract discussion topics from a transcript chunk."""
    system_prompt = f'''
"Based on the input text from a seller's conversation, identify and extract the discussion topics."
"The conversation may cover parts like introductions, product features, pricing, marketing strategies, customer feedback, and order processing."
"Return the topics as a JSON array of strings."

Format Requirement:

The output should be strictly in the format: [topic1, topic2, topic3]
Ensure the output is clean and machine-readable.
Examples:

1st example:
User Prompt:
"Hello, everyone! I'm excited to introduce our brand-new product line that we’ve been working on for months.
First, I'll give you an overview of what makes our product unique compared to competitors. Then, I’ll walk you through its key features, including the advanced technology behind it and how it solves common problems customers face.
Next, we’ll discuss pricing details, available discounts, and flexible payment plans. Finally, I'll touch upon our marketing strategies, partnerships, and how we plan to reach our target audience."

Expected Output from your model:
["Introduction", "Product Features", "Pricing", "Marketing Strategies"]

2nd example:
User Prompt:
"Good morning, everyone. Today’s discussion will focus on some of the most crucial aspects of patient care.
We'll start with an in-depth look at modern diagnosis techniques, including imaging, lab tests, and AI-assisted tools that improve accuracy.
Then, we’ll explore various treatment plans, ranging from medication management to surgical interventions and alternative therapies.
Another important aspect we’ll cover is how insurance procedures work, including claim filing, coverage eligibility, and patient rights.
Finally, we’ll wrap up with patient feedback mechanisms, ensuring continuous improvement in healthcare services."

Expected Output from your model:
["Patient Diagnosis", "Treatment Plans", "Insurance Procedures", "Patient Feedback"]
Please don't give any explanation or additional text in the output. Only provide the JSON array of topics.
The result generated by you, will be passed on to the later part of the code. So striclty follow the format of a JSON array.
    '''
    prompt = f'''
    {chunk}
    '''
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Using GPT-4o-mini model for efficiency
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    )
    
    topics = json.loads(response.choices[0].message.content)
    print(topics)
    return topics

def match_topics(agenda: list, extracted_topics: list) -> list:
    """Match extracted topics with agenda topics using semantic similarity."""
    matched_topics = []
    for topic in extracted_topics:
        topic_embedding = embedding_model.encode(topic, convert_to_tensor=True)
        scores = [
            util.pytorch_cos_sim(topic_embedding, embedding_model.encode(a, convert_to_tensor=True)).item()
            for a in agenda
        ]
        if max(scores) > 0.3:  # Matching threshold
            matched_topics.append(agenda[scores.index(max(scores))])
    return matched_topics

def load_transcript_from_file(file_path: str) -> str:
    """Load transcript text from a .txt file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        transcript = file.read()
    return transcript

# -------------------------------
# Process Transcription Function
# -------------------------------

async def process_transcription(meeting_description:str, agenda: list, transcript: str = None) -> list:
    """Process a meeting transcript by splitting it into chunks and analyzing each."""
    if transcript is None:
        transcript = load_transcript_from_file("transcript.txt")
    
    # Split transcript into 4 chunks for sequential processing
    chunks = chunk_text(transcript)
    discussed_topics = []
    
    for chunk in chunks:
        topics = await extract_topics(chunk)
        matched = match_topics(agenda, topics)
        discussed_topics.extend(matched)
        # await asyncio.sleep(2)  # Simulating real-time delay
    
    return list(set(discussed_topics))

# -------------------------------
# FastAPI Endpoints
# -------------------------------

@app.post("/generate-agenda")
async def generate_agenda_endpoint(request: MeetingRequest):
    """API endpoint to generate a meeting agenda."""
    try:
        agenda = await generate_agenda(request.meeting_description)
        return {"agenda": agenda}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-transcription")
async def process_transcription_endpoint(request: ProcessRequest):
    """API endpoint to process a meeting transcription."""
    try:
        discussed_topics = await process_transcription(request.meeting_description, request.agenda, request.transcript)
        return {"discussed_topics": discussed_topics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# Run the Server
# -------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
