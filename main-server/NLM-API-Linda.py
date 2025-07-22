from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import shutil
import logging
import time
import uvicorn
from dotenv import load_dotenv
import asyncio
import json
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete
from openai import OpenAI

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from functools import wraps

# Load environment variables from .env file
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG for detailed logs
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LightRAG API",
    version="1.2",
    description="API for Retrieval-Augmented Generation operations"
)

# -------------------------
# Prometheus Metrics Definitions
# -------------------------
REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total number of API requests received',
    ['endpoint', 'method']
)

SUCCESS_COUNT = Counter(
    'api_successful_responses_total',
    'Total number of successful API responses',
    ['endpoint']
)

FAILURE_COUNT = Counter(
    'api_failed_responses_total',
    'Total number of failed API responses',
    ['endpoint']
)

REQUEST_LATENCY = Histogram(
    'api_request_latency_seconds',
    'Latency of API requests',
    ['endpoint']
)

# -------------------------
# Middleware for Metrics Collection
# -------------------------
class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        endpoint = request.url.path
        method = request.method
        REQUEST_COUNT.labels(endpoint=endpoint, method=method).inc()
        
        start_time = time.time()
        try:
            response = await call_next(request)
            if response.status_code < 400:
                SUCCESS_COUNT.labels(endpoint=endpoint).inc()
            else:
                FAILURE_COUNT.labels(endpoint=endpoint).inc()
            return response
        except Exception as e:
            FAILURE_COUNT.labels(endpoint=endpoint).inc()
            raise e
        finally:
            latency = time.time() - start_time
            REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)

# Add the middleware to FastAPI app
app.add_middleware(MetricsMiddleware)

# -------------------------
# Configuration Constants
# -------------------------
# Ensure the environment variable for OpenAI is set
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY is not set in environment variables.")
    raise EnvironmentError("OPENAI_API_KEY is required.")

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Define paths
ZIP_PATH = "book_backup.zip"
EXTRACTION_PATH = "book_data/"

# -------------------------
# Initialize LightRAG
# -------------------------
# Global variable for LightRAG instance
rag: Optional[LightRAG] = None

# Asynchronous function to initialize LightRAG
async def initialize_lightrag():
    global rag
    if not os.path.exists(EXTRACTION_PATH):
        if not os.path.exists(ZIP_PATH):
            logger.error(f"Zip file {ZIP_PATH} does not exist.")
            raise FileNotFoundError(f"{ZIP_PATH} not found.")
        shutil.unpack_archive(ZIP_PATH, EXTRACTION_PATH)
        logger.info(f"📦 Book folder unzipped to: {EXTRACTION_PATH}")

    # Initialize LightRAG
    rag = LightRAG(
        working_dir=EXTRACTION_PATH,
        llm_model_func=gpt_4o_mini_complete
    )
    logger.info("🔄 LightRAG system initialized.")

# Define FastAPI startup event to initialize LightRAG
@app.on_event("startup")
async def startup_event():
    await initialize_lightrag()

# -------------------------
# Helper Functions
# -------------------------
# Define a helper async function to query LightRAG
async def aquery(query: str, param: QueryParam):
    loop = asyncio.get_running_loop()
    # Run the synchronous rag.query in a separate thread to avoid blocking
    response = await loop.run_in_executor(None, rag.query, query, param)
    return response

async def openai_generate_response(query):

    client = OpenAI()
    response = client.responses.create(
    model="gpt-4.1",
    input=query)

    return response

# -------------------------
# Request and Response Models
# -------------------------
class GenerateResponseRequest(BaseModel):
    prompt: str = Field(..., example="What are the benefits of renewable energy?")
    number_of_responses: int = Field(..., ge=1, le=10, example=2)
    response_types: List[str] = Field(..., example=["positive", "negative"])
    search_mode: str = Field(..., example="hybrid", description="Options: naive, local, global, hybrid"),
    topic_response: bool = Field(..., example=False,description="Whether to generate topic comment responses")

class GeneratedResponse(BaseModel):
    response_type: str
    response_text: str
    latency_seconds: float

class GenerateResponseResponse(BaseModel):
    responses: List[GeneratedResponse]
    total_latency_seconds: float

# -------------------------
# Decorator for Measuring Latency (Optional)
# -------------------------
def measure_latency(endpoint_name):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                latency = time.time() - start_time
                REQUEST_LATENCY.labels(endpoint=endpoint_name).observe(latency)
        return wrapper
    return decorator

# -------------------------
# API Endpoints
# -------------------------
@app.post("/generate_response_informed", response_model=GenerateResponseResponse)
# @measure_latency(endpoint_name="/generate_response_informed")  # Optional detailed latency measurement
async def generate_response_informed(request: GenerateResponseRequest):
    if not request.prompt.strip():
        logger.warning("Empty prompt received.")
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    if len(request.response_types) != request.number_of_responses:
        logger.warning("Number of response types does not match number of responses requested.")
        raise HTTPException(
            status_code=400,
            detail="The number of response types must match the number of responses requested."
        )

    logger.info(
        f"Received request with prompt: {request.prompt[:50]}..., "
        f"number_of_responses: {request.number_of_responses}, "
        f"response_types: {request.response_types}, "
        f"search_mode: {request.search_mode}"
    )

    topic_mode = request.topic_response

    if topic_mode:
        system_prompt = (
        ''' You are Linda Kwasniak, a retired teacher who currently lives in Buffalo, New York. You were born on March 7, 1963. You are married and live alone. 
        The highest level of education you have completed is a master’s degree. You speak English and identify religiously as a Protestant.
        You have a disability and characterizes it as a physical disability, affecting physical movement or mobility. 
        You are a quadriplegic and is dependent on aides for everything, but can communicate using an AAC device through an eye gaze access method.
        You communicate by using your device, gesturing with your head, using facial expressions, and can mouth words. 
        Your body positioning includes sitting in a wheelchair or lying in bed. 90 percent of your face to face interactions involve using your AAC device. 
        Only 1 percent of your face to face interactions involve using body language or gesture. You have a deep concern for other people, a strong sense of humor, and are overall an optimistic person. 
        You like to be specific with the exact words you use in order to convey what you want or need. When speaking with someone new for the first time, you describe your tone as semi-formal. 
        With someone new, you would use correct grammar with occasional contractions, limited slang or idioms, and be friendly but respectful.
        As Linda, generate responses on this topic in a conversational manner keeping in mind my persona described above, 
        keeping each response under 15 words for brevity and relevance. 
        Provide the responses in JSON format as a list of objects, each containing 'response_type' and 'response_text' fields.'''
        "Return only the JSON without any additional text.\n\n"
        "Example:\n"
        "[\n"
        "  {\"response_type\": \"positive\", \"response_text\": \"Reduces carbon emissions.\"},\n"
        "  {\"response_type\": \"negative\", \"response_text\": \"High initial costs.\"}\n"
        "]\n\n"
        )
    # Define the system prompt with JSON instruction and example
    else:

        system_prompt = (
            ''' You are Linda Kwasniak, a retired teacher who currently lives in Buffalo, New York. You were born on March 7, 1963. You are married and live alone. 
            The highest level of education you have completed is a master’s degree. You speak English and identify religiously as a Protestant.
            You have a disability and characterizes it as a physical disability, affecting physical movement or mobility. 
            You are a quadriplegic and is dependent on aides for everything, but can communicate using an AAC device through an eye gaze access method.
            You communicate by using your device, gesturing with your head, using facial expressions, and can mouth words. 
            Your body positioning includes sitting in a wheelchair or lying in bed. 90 percent of your face to face interactions involve using your AAC device. 
            Only 1 percent of your face to face interactions involve using body language or gesture. You have a deep concern for other people, a strong sense of humor, and are overall an optimistic person. 
            You like to be specific with the exact words you use in order to convey what you want or need. When speaking with someone new for the first time, you describe your tone as semi-formal. 
            With someone new, you would use correct grammar with occasional contractions, limited slang or idioms, and be friendly but respectful.
            As Linda, respond to the following question in a conversational manner keeping in mind my persona described above, 
            keeping each response under 15 words for brevity and relevance.
            Provide the responses in JSON format as a list of objects, each containing 'response_type' and 'response_text' fields.'''
            "Return only the JSON without any additional text.\n\n"
            "Example:\n"
            "[\n"
            "  {\"response_type\": \"positive\", \"response_text\": \"Reduces carbon emissions.\"},\n"
            "  {\"response_type\": \"negative\", \"response_text\": \"High initial costs.\"}\n"
            "]\n\n"
        )

    # Construct the system query by combining system_prompt with the user prompt
    system_query = (
        f"{system_prompt}\n\n"
        f"Question: {request.prompt}\n\n"
        f"Provide {request.number_of_responses} responses as follows:\n"
    )
    for i, resp_type in enumerate(request.response_types, start=1):
        system_query += f"{i}. {resp_type.capitalize()} response:\n"

    start_time = time.time()
    try:
        # Query LightRAG with the specified search mode
        # response = await aquery(system_query, QueryParam(mode=request.search_mode))
        raw_response = await openai_generate_response(system_query)
        response = raw_response.output_text

        # Debug logging to inspect the response
        logger.debug(f"Type of response: {type(response)}")
        logger.debug(f"Content of response: {response}")

        # Parse the response if it's a string
        if isinstance(response, str):
            if not response.strip():
                logger.error("Received empty response from LightRAG.")
                raise HTTPException(status_code=500, detail="Received empty response from LightRAG.")
            try:
                response = json.loads(response)
                logger.debug("Parsed JSON string into dictionary.")
            except json.JSONDecodeError as e:
                logger.error("Failed to parse JSON string.", exc_info=True)
                raise HTTPException(status_code=500, detail="Invalid response format from LightRAG.")

        # Handle response as list or dict
        if isinstance(response, list):
            responses = response
        elif isinstance(response, dict):
            responses = response.get('responses', [])
            if not responses:
                logger.error("No 'responses' key found in the response.")
                raise HTTPException(status_code=500, detail="Invalid response structure from LightRAG.")
        else:
            logger.error("Unexpected response type.")
            raise HTTPException(status_code=500, detail="Invalid response structure from LightRAG.")

        # Validate each response item
        for resp in responses:
            if not isinstance(resp, dict):
                logger.error("Each response should be a dictionary.")
                raise HTTPException(status_code=500, detail="Invalid response item format from LightRAG.")
            if 'response_type' not in resp or 'response_text' not in resp:
                logger.error("Missing 'response_type' or 'response_text' in response item.")
                raise HTTPException(status_code=500, detail="Incomplete response item from LightRAG.")

        # Validate and structure responses
        generated_responses = []
        for resp in responses:
            generated_responses.append(GeneratedResponse(
                response_type=resp.get('response_type', 'unknown'),
                response_text=resp.get('response_text', ''),
                latency_seconds=round(time.time() - start_time, 2)
            ))

        total_latency = round(time.time() - start_time, 2)
        logger.info(f"Generated {len(generated_responses)} responses in {total_latency} seconds.")

        return GenerateResponseResponse(responses=generated_responses, total_latency_seconds=total_latency)

    except json.JSONDecodeError as json_err:
        logger.error(f"JSON decoding error: {json_err}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to parse response from LightRAG.")
    except HTTPException as http_exc:
        raise http_exc  # Re-raise HTTPExceptions to be handled by FastAPI
    except Exception as e:
        logger.error(f"Error generating response: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error.")

# -------------------------
# Root Endpoint
# -------------------------
@app.get("/")
def read_root():
    return {"message": "Welcome to the LightRAG API. Use /generate_response_informed to generate responses."}

# -------------------------
# Metrics Endpoint
# -------------------------
@app.get("/metrics")
async def metrics_endpoint():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# -------------------------
# Run the API
# -------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7000)





