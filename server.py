import docx
import asyncio
import numpy as np
import json
from sentence_transformers import SentenceTransformer, util
from groq import AsyncGroq
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# -------------------------------
# Initialize Models and Groq Llama
# -------------------------------

# Load the embedding model (free model: all-MiniLM-L6-v2)
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Initialize Groq Llama model (update the API key as needed)
groq = AsyncGroq(api_key="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")

# -------------------------------
# FastAPI App and Request Models
# -------------------------------

app = FastAPI()
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class MeetingRequest(BaseModel):
    meeting_description: str

class ProcessRequest(BaseModel):
    meeting_description: str
    agenda: list
    transcript: str = None  # Optional: if not provided, a default transcript is used

# -------------------------------
# Helper Functions
# -------------------------------

def load_transcription(doc_path: str) -> str:
    """Load transcription text from a Word document."""
    doc = docx.Document(doc_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def chunk_text(text: str, num_chunks: int = 4) -> list:
    """Split the text into `num_chunks` parts (to simulate real-time processing)."""
    words = text.split()
    chunk_size = len(words) // num_chunks
    return [" ".join(words[i * chunk_size:(i + 1) * chunk_size]) for i in range(num_chunks)]

# -------------------------------
# LLM Call Functions Using Groq Llama
# -------------------------------

async def generate_agenda(meeting_description: str) -> list:
    """Generate meeting agenda topics based on the meeting description."""
    system_prompt = """
You are a meeting organizer who needs to generate an agenda based on the topics discussed in the conversation.
Identify and list only the relevant topics as a JSON array of strings.
For example:
Input: "Discuss sales strategy, new product launch, and customer feedback analysis."
Expected Output: ["Introduction", "Product Features", "Pricing"]
You need to generate between 4 to 6 agenda topics based on the meeting description provided.
IMPORTANT: Respond with only the JSON array, without any extra text or explanation.
"""
    prompt = f'''
    {meeting_description}
    '''
    response = await groq.chat.completions.create(
        model="llama-3.1-8b-instant",  # Update model name if needed
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=150,
        temperature=0.7,
    )
    # print("response is", response.choices[0].message.content)
    agenda = json.loads(response.choices[0].message.content)
    # detailed_agenda = await get_agenda_details(agenda)
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
    response = await groq.chat.completions.create(
        model="llama-3.1-8b-instant",  # Update model name if needed
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=150,
        temperature=0.7,
    )
    print("response is", response.choices[0].message.content)
    topics = json.loads(response.choices[0].message.content)
    print("the chunk is ", chunk)
    print("topics are", topics)
    return topics

def match_topics(agenda: list, extracted_topics: list) -> list:
    """Match the topics extracted from a chunk with the generated agenda topics using semantic similarity."""
    matched_topics = []
    for topic in extracted_topics:
        topic_embedding = embedding_model.encode(topic, convert_to_tensor=True)
        scores = [
            util.pytorch_cos_sim(topic_embedding, embedding_model.encode(a, convert_to_tensor=True)).item()
            for a in agenda
        ]
        if max(scores) > 0.5:  # Threshold for a match
            matched_topics.append(agenda[scores.index(max(scores))])
    print("Matched topics are", matched_topics) 
    return matched_topics

# -------------------------------
# Process Transcription Function
# -------------------------------

async def process_transcription(meeting_description:str,agenda: list, transcript: str = None) -> list:
    """
    Process the meeting transcript in 4 chunks to simulate real-time analysis.
    If a transcript is not provided, a default hardcoded transcript is used.
    Returns the final list of discussed topics.
    """
    if transcript is None:
        transcript = """
1. Candidate Introduction (3 minutes)
I: Welcome! Thank you for joining us today. Could you please introduce yourself and give a brief overview of your experience in Python development?

C: Sure! My name is [Your Name], and I am a software engineer with [X] years of experience in Python development. I have worked on projects related to [mention relevant domains, e.g., web development, data science, automation, etc.]. Recently, I built [mention a significant project], where I used Python to [explain key achievements]. I have experience with frameworks like Django and Flask, as well as libraries such as NumPy and Pandas for data processing.

I: That sounds great! Could you also share what excites you the most about Python development?

C: I love Python’s simplicity and versatility. Whether it's building scalable web applications, automating tasks, or working with machine learning, Python provides powerful tools and libraries to make development efficient.

2. Python Skill Testing (7 minutes)
I: Let’s move on to some technical questions.

Q1: Can you explain the difference between deepcopy() and copy() in Python?
C: Sure! copy.copy() creates a shallow copy, meaning it only copies the references of nested objects, not the actual objects. On the other hand, copy.deepcopy() creates a deep copy, meaning it recursively copies all objects and their nested structures, ensuring that modifications to the copied object do not affect the original one.

Q2: How does Python manage memory, and what are some ways to optimize memory usage?
C: Python manages memory using reference counting and garbage collection. Reference counting keeps track of the number of references to an object, and when it drops to zero, the object is deallocated. The garbage collector handles cyclic references. To optimize memory usage, we can:

Use generators instead of lists for large data processing.
Utilize __slots__ to restrict dynamic attribute creation in classes.
Avoid unnecessary global variables and large objects in memory.
Q3: Can you write a Python function to find the second largest number in a list without using built-in sorting functions?
C: Sure! Here’s a simple implementation:

python
Copy
Edit
def second_largest(numbers):
    first, second = float('-inf'), float('-inf')
    for num in numbers:
        if num > first:
            second, first = first, num
        elif num > second and num != first:
            second = num
    return second if second != float('-inf') else None

# Example
print(second_largest([10, 20, 4, 45, 99]))  # Output: 45
I: That’s a clean and efficient approach.

Q4: Have you worked with multithreading in Python? Can you explain how the Global Interpreter Lock (GIL) affects concurrency?
C: Yes, I have worked with multithreading. The GIL in Python ensures that only one thread executes Python bytecode at a time, which can limit true parallel execution in CPU-bound tasks. However, for I/O-bound tasks like network requests and file I/O, multithreading can still be beneficial. For CPU-bound tasks, multiprocessing is a better choice since it creates separate processes, bypassing the GIL.

3. Salary Discussion (5 minutes)
I: Thanks for your answers! Now, let’s discuss salary expectations. What are you looking for in terms of compensation?

C: Based on my experience and market trends, I am expecting a package in the range of ₹X to ₹Y LPA (or $X to $Y per year if international). However, I am open to discussion based on the role’s responsibilities and growth opportunities.

I: That sounds reasonable. We have a budget within that range, and we also offer performance-based incentives and benefits. Would you be open to negotiating based on experience and project contributions?

C: Yes, I am open to discussing it further.

I: Great! Do you have any questions for me?

C: Yes, could you share some insights into the team structure and what a typical project cycle looks like?

I: Absolutely. Our team follows an Agile methodology with two-week sprints. Developers work closely with product managers and QA teams to ensure smooth deployments. You’ll have the opportunity to contribute to design discussions and optimize existing applications.

Interview Conclusion
I: Thank you for your time. We’ll review your application and get back to you soon regarding the next steps.

C: Thank you! I look forward to hearing from you.
        """
    # print("hello")
    # Split the transcript into 4 chunks
    chunks = chunk_text(transcript)
    # Generate agenda topics based on the meeting description
    # agenda = await generate_agenda(meeting_description)
    discussed_topics = []
    # Process each chunk sequentially, simulating real-time delay
    for chunk in chunks:
        # print("hii")
        chunk_embedding = embedding_model.encode(chunk, convert_to_tensor=True)
        description_embedding = embedding_model.encode(meeting_description, convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(chunk_embedding, description_embedding).item()
        print(f"Similarity score between chunk and meeting_description: {similarity_score}")
        
        # if similarity_score > 0.3:  # Threshold for similarity
        topics = await extract_topics(chunk)
        # print("topics are", topics)
        matched = match_topics(agenda, topics)
        discussed_topics.extend(matched)
        await asyncio.sleep(2)  # Non-blocking sleep
    # Return unique topics that were matched
    print("discussed topics are", discussed_topics)
    print("unique topics are", list(set(discussed_topics)))
    return list(set(discussed_topics))

# -------------------------------
# FastAPI Endpoints
# -------------------------------

@app.post("/generate-agenda")
async def generate_agenda_endpoint(request: MeetingRequest):
    try:
        agenda = await generate_agenda(request.meeting_description)
        return {"agenda": agenda}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-transcription")
async def process_transcription_endpoint(request: ProcessRequest):
    try:
        discussed_topics = await process_transcription(request.meeting_description,request.agenda, request.transcript)
        return {"discussed_topics": discussed_topics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# Run the Server
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)