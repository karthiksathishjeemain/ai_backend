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
You are an array generator.
Your task is to extract the discussion topics from a seller's conversation text.
The conversation may include subjects such as introductions, product features, pricing, and marketing strategies.
Identify and list only the relevant topics as a JSON array of strings.
For example:
Input: "Discuss sales strategy, new product launch, and customer feedback analysis."
Expected Output: ["Introduction", "Product Features", "Pricing"]
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
    print("response is", response.choices[0].message.content)
    agenda = json.loads(response.choices[0].message.content)
    # detailed_agenda = await get_agenda_details(agenda)
    return agenda

# async def get_agenda_details(agenda: list) -> list:
#     """Get details for each agenda topic."""
#     system_prompt = f'''
#     "You are a detail generator."
#     "Based on the input agenda topics, provide detailed descriptions for each topic."
#     "Return the details as a JSON array of dictionaries with 'topic' and 'detail' keys."
#     "For example:"
#     "Input: ['Introduction', 'Product Features', 'Pricing']"
#     "Output: [{{'topic': 'Introduction', 'detail': 'This is the introduction.'}}, {{'topic': 'Product Features', 'detail': 'These are the product features.'}}, {{'topic': 'Pricing', 'detail': 'This is the pricing.'}}]"
#     '''
#     prompt = f'''
#     {agenda}
#     '''
#     response = await groq.chat.completions.create(
#         model="llama-3.1-8b-instant",  # Update model name if needed
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": prompt},
#         ],
#         max_tokens=150,
#         temperature=0.7,
#     )
#     detailed_agenda = json.loads(response.choices[0].message.content)
#     return detailed_agenda

async def extract_topics(chunk: str) -> list:
    """Extract discussion topics from a transcript chunk."""
    system_prompt = f'''
    "You are an array generator."
    "Based on the input text from a seller's conversation, identify and extract the discussion topics."
    "The conversation may cover parts like introductions, product features, pricing, and marketing strategies."
    "Return the topics as a JSON array of strings."
    "For example:"
    "User prompt: 'Hello, today I will introduce our new product, discuss its features, and go over pricing details.'"
    "Output: [\\"Introduction\\", \\"Product Features\\", \\"Pricing\\"]"
    " you some times write those topics inside flower brackets, which is not required.Please follow the format as mentioned above. It should be in this format:[topic1,topic2,topic3]"
    "Don't generate anything except the JSON array as the response because it will be passed to further coding steps."
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
Time: 10:15 AM
Platform: TechSpace Online Chat Support
Alex: Hi there! I'm Alex, your TechSpace sales specialist. How can I assist you today? 💻
Customer: Hi Alex. I need help choosing a laptop for video editing. My budget is around $1500
Alex: Thanks for reaching out! I'd be happy to help you find the perfect laptop for video editing. Could you tell me what kind of video editing software you use and what type of projects you typically work on?
Customer: I use Premier Pro mainly. I edit YouTube videos, mostly 4K footage. My current laptop struggles with rendering
Alex: Thank you for those details! That helps a lot. 4K editing definitely needs some serious power. I've got a few great options in mind. Would you like me to share some links with detailed specs?
Customer: yes please
Alex: Here are three options I'd recommend:
1. MSI Creator 15 - $1,499
[Link: Details attached]
- RTX 3060 GPU
- 16GB RAM (upgradeable)
- 15.6" 4K Display
- 1TB SSD
2. Acer ConceptD 3 - $1,399
[Link: Details attached]
- RTX 3050 Ti
- 16GB RAM
- 100% sRGB Display
- 512GB SSD
3. ASUS ProArt Studiobook - $1,699
[Link: Details attached]
- RTX 3060
- 32GB RAM
- 100% DCI-P3 Display
- 1TB SSD
Which one would you like to know more about?
Customer: The MSI looks good but it's at the top of my budget. I saw it cheaper on CompetitorTech's website
Alex: I appreciate you letting me know about that! Could you share the CompetitorTech listing? We have a price match guarantee and I'd be happy to check if we can match or beat their price.
Customer: They have it for $1,399 with a free cooling pad
Alex: Let me check that offer... I can see their listing is for last quarter's model (A12 series), while ours is the latest A13 with the updated cooling system and better color accuracy. However, I want to make this work for you! Here's what I can do:
1. Match their $1,399 price
2. Include our premium cooling pad ($89 value)
3. Add our exclusive 3-year warranty upgrade ($79 value)
Would that help with your decision?
Customer: Can you throw in more RAM too? I'll need it for editing
Alex: The standard 16GB to 32GB RAM upgrade is normally $199, but let me see what I can do... 🤔
I could offer you this package:
- MSI Creator 15 (latest A13): $1,399 (matched price)
- 32GB RAM upgrade: $149 (25% off)
- Premium cooling pad: Included
- 3-year warranty: Included
Total would be $1,548. Still a bit high, but I could also...
1. Add 1TB external SSD for $79 (normally $149)
2. OR include our Pro Creator Software Bundle ($199 value)
What interests you most?
Customer: The external SSD sounds good but the total is still too high. Can you do better on the RAM upgrade?
Alex: I can see you're serious about getting the right setup. Let me check with my supervisor...
[2 minutes later]
Great news! I got special approval for this package:
- MSI Creator 15 (A13): $1,399
- 32GB RAM upgrade: $99 (50% off!)
- Premium cooling pad: Included
- 3-year warranty: Included
- 1TB external SSD: $69 (special bundle price)
Total: $1,567 - AND I'll throw in a $50 gift card for future accessories!
Is this configuration getting closer to what you're looking for?
Customer: What if I pay in full instead of financing? Any extra discount?
Alex: Absolutely! For full payment, I can offer:
1. Additional 3% off entire purchase (-$47)
2. Instant rebate of $30
3. Plus all previous offers!
New breakdown:
- MSI Creator 15 (A13): $1,399
- 32GB RAM upgrade: $99
- External SSD: $69
- All free items included
- Subtotal: $1,567
- Full payment discount: -$47
- Instant rebate: -$30
- Final total: $1,490
That brings us under your original budget! 🎉
Plus you still get:
- Premium cooling pad
- 3-year warranty
- $50 gift card
- Free next-day shipping
Should we proceed with this package?
Customer: Can you include some dongles? I'll need USB-C to HDMI at least
Alex: You drive a hard bargain! 😊 But yes, I can help with that. I'll include our Pro Connectivity Kit:
- USB-C to HDMI 2.1 dongle
- USB-C to DisplayPort
- USB-C hub with ethernet
(Total value $89 - included free!)
And since you're such a thorough negotiator, I'll even add:
- Cable management sleeve
- Screen cleaning kit
All at the same final price of $1,490. This is absolutely the best package I can offer - I can't even add a paperclip without getting fired! 😅
Ready to proceed?
Customer: Deal! Credit card please
[Rest of the conversation continues as before with payment processing and delivery details]
Customer: No that's all thanks!
Alex: Thank you for choosing TechSpace! Don't forget to:
1. Register your protection plan when your laptop arrives
2. Download our TechCare app for support
3. Use your Adobe CC activation code
4. Activate your $50 gift card (valid for 12 months)
If you need any help with setup or have questions, just start a chat or call us 24/7!
Have a great day! 👋
[Chat ended at 10:30 AM]
        """
    # print("hello")
    # Split the transcript into 4 chunks
    chunks = chunk_text(transcript)
    # Generate agenda topics based on the meeting description
    # agenda = await generate_agenda(meeting_description)
    discussed_topics = []
    # Process each chunk sequentially, simulating real-time delay
    for chunk in chunks:
        print("hii")
        chunk_embedding = embedding_model.encode(chunk, convert_to_tensor=True)
        description_embedding = embedding_model.encode(meeting_description, convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(chunk_embedding, description_embedding).item()
        print(f"Similarity score between chunk and meeting_description: {similarity_score}")
        
        if similarity_score > 0.3:  # Threshold for similarity
            topics = await extract_topics(chunk)
            print("topics are", topics)
            matched = match_topics(agenda, topics)
            discussed_topics.extend(matched)
        await asyncio.sleep(2)  # Non-blocking sleep
    # Return unique topics that were matched
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