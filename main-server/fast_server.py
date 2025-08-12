from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, status
from fastapi.responses import FileResponse, Response
import json
import logging
import uvicorn
import csv
import httpx
import asyncio
from datetime import datetime
from dateutil.parser import parse
import os
import pytz
import time
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from collections import deque
from typing import Optional


# ------------------------ Logging Configuration ------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ------------------------ FastAPI Initialization ------------------------

app = FastAPI(title="Main Server", version="1.1")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------ Prometheus Metrics Definitions ------------------------
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from functools import wraps

# Define Prometheus metrics
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

WEBSOCKET_CONNECTIONS = Gauge(
    'websocket_connections',
    'Number of active WebSocket connections'
)

WEBSOCKET_MESSAGES_RECEIVED = Counter(
    'websocket_messages_received_total',
    'Total number of messages received via WebSocket',
    ['endpoint']
)

WEBSOCKET_MESSAGES_SENT = Counter(
    'websocket_messages_sent_total',
    'Total number of messages sent via WebSocket',
    ['endpoint']
)

CSV_FILES_CREATED = Counter(
    'csv_files_created_total',
    'Total number of CSV files created'
)

CSV_ENTRIES_APPENDED = Counter(
    'csv_entries_appended_total',
    'Total number of entries appended to CSV files'
)

# ------------------------ Middleware for Metrics Collection -------------------------
class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        endpoint = request.url.path
        if endpoint == '/metrics':
            return await call_next(request)
        
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

# ------------------------ Helper Functions for Prometheus Metrics -------------------------
def websocket_connect():
    WEBSOCKET_CONNECTIONS.inc()

def websocket_disconnect():
    WEBSOCKET_CONNECTIONS.dec()

def websocket_message_received(endpoint: str):
    WEBSOCKET_MESSAGES_RECEIVED.labels(endpoint=endpoint).inc()

def websocket_message_sent(endpoint: str):
    WEBSOCKET_MESSAGES_SENT.labels(endpoint=endpoint).inc()

def csv_file_created():
    CSV_FILES_CREATED.inc()

def csv_entry_appended():
    CSV_ENTRIES_APPENDED.inc()

# ------------------------ HTTP Client Initialization ------------------------

client = httpx.AsyncClient()
api_url = "http://localhost:7000/generate_response_informed"  # Ensure this points to the API server's internal IP if on GCP
headers = {"Content-Type": "application/json"}

# ------------------------ CSV Configuration ------------------------

session_csv_dir = "session_csv_files"
os.makedirs(session_csv_dir, exist_ok=True)

# Updated CSV headers to include the full prompt sent to the API
CSV_HEADERS = [
    'index', 'date_time', 'prompt', 'prompt_for_response', 'history', 'responses',
    'chosen_response', 'input_type', 'api_latency', 'chosen_response_latency', 'full_prompt_to_api', 'full_conversation_history'
]


# Eastern Time zone
ET = pytz.timezone('US/Eastern')


# ------- Contants ------ #
RESPONSE_TYPES = [
    "positive", "negative",
    "positive with more variation in response", "negative with more variation in response",
    "a follow-up question with positive intent", "a follow-up question with negative intent",
    "a follow-up question with positive intent and more response variation",
    "a follow-up question with positive intent and more response variation"
]

# ------------------------ Helper Functions ------------------------

def generate_csv_filename():
    timestamp = datetime.now(ET).strftime("%Y%m%d_%H%M%S")
    filename = f"conversation_history_{timestamp}.csv"
    path = os.path.join(session_csv_dir, filename)
    csv_file_created()
    return path

def initialize_csv_file(path):
    try:
        with open(path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=CSV_HEADERS)
            writer.writeheader()
        logger.info(f"Initialized CSV file at {path}")
    except Exception as e:
        logger.error(f"Failed to create CSV: {e}")

def append_to_csv_file(path, entry_dict):
    try:
        with open(path, 'a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=CSV_HEADERS)
            writer.writerow(entry_dict)
        logger.info(f"Appended entry to CSV at {path}")
        csv_entry_appended()
    except Exception as e:
        logger.error(f"Failed to append to CSV: {e}")

async def send_to_api_async(prompt, number_of_responses, response_types, search_mode, generate_topic_response):

    payload = {
        'prompt': prompt,
        'number_of_responses': number_of_responses,
        'response_types': response_types,
        'search_mode': search_mode,
        'topic_response': generate_topic_response
    }
    logger.info(f"Sending payload to API with prompt:\n{prompt}")


    # Custom timeout configuration
    timeout = httpx.Timeout(
        connect=5.0,  # Max time to establish connection
        read=10.0,    # Max time to wait for response data
        write=5.0,    # Max time to send request data
        pool=5.0      # Max time to wait for a connection from the pool
    )

    try:
        response = await client.post(api_url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        response_json = response.json()
        logger.info(f"Received response from API: {response_json}")
        return (response_json,True,None)
    
    except httpx.ReadTimeout as e:
        logger.error("Request timed out.")
        return ({"responses": []}, False, "Request timed out. Please try again.")

    except httpx.ConnectTimeout as e:
        logger.error("Connection timed out.")
        return ({"responses": []}, False, "Connection timed out. Please try again.")

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
        return ({"responses": []}, False, e.response.text)

    except httpx.RequestError as e:
        logger.error(f"Request error: {e}")
        return ({"responses": []}, False, str(e))

async def get_responses_from_api(prompt: str) -> tuple[list[dict], bool, str, float]:
    try:
        start = datetime.now(ET)
        response, ok, error = await send_to_api_async(
            prompt,
            number_of_responses=8,
            response_types=RESPONSE_TYPES,
            search_mode="naive",
            generate_topic_response=False
        )
        latency = (datetime.now(ET) - start).total_seconds()
        return response.get("responses", []), ok, error, latency
    except Exception as e: # adds an extra check to ensure errors apart from httpx errors (low level errors) are handled properly and there is a fall-back for such unexpected situations
        logger.error(f"API call failed: {e}", exc_info=True)
        return fallback_response(), False, str(e), 0.0

def check_last_entry(history):
    if history and history[-1]['user_response'] is None:
        logger.warning("Incomplete entry found in conversation history.")
        return handle_incomplete_entry(history)
    return None

def handle_incomplete_entry(history):
    incomplete_entry = history.pop()
    logger.info(f"Removed incomplete entry: {incomplete_entry['prompt']}")
    return f"Didn't choose a response; removed: {incomplete_entry['prompt']}"

def format_conversation_history_for_prompt(conversation_history):
    """
    Format the last few turns of the conversation as context:
    Partner: <partner_prompt>
    User: <chosen_response>
    """
    entries = conversation_history[-3:]
    history_str = "Conversation so far:\n"
    for h in entries:
        history_str += f"Partner: {h['prompt']}\n"
        if h['user_response']:
            history_str += f"User: {h['user_response']}\n"
    return history_str.strip()

def update_history(history, partner_prompt, user_response, model_responses, full_history, prompt_for_response, emotion, api_latency=0):
    if len(history) >= 3:
        removed = history.pop(0)
        logger.debug(f"Removed oldest entry from conversation history: {removed['prompt']}")

    history.append({
        'prompt': partner_prompt,
        'user_response': user_response,
        'prompt_for_response': prompt_for_response,
        'api_latency': api_latency
    })

    if partner_prompt is not None and model_responses is not None: # partner initiated conversation
        history_snapshot = history[-3:].copy()
        full_history.append({
            'prompt': partner_prompt,
            'responses': model_responses,
            'user_response': user_response,
            'history_snapshot': history_snapshot,
            'emotion': emotion
        })

    elif partner_prompt is None: # user initiated conversation
        history_snapshot = history[-3:].copy()
        full_history.append({
            'prompt': partner_prompt,
            'responses': model_responses,
            'user_response': user_response,
            'history_snapshot': history_snapshot,
            'emotion': emotion
        })

def update_full_history(full_history, last_convo_pair, chosen_response):
    for entry in reversed(full_history):
        #if entry['prompt'] == last_convo_pair['prompt'] and entry['user_response'] is None:
        if entry['user_response'] is None:
            entry['user_response'] = chosen_response
            break

def format_history_for_csv(history_snapshot):
    """
    Convert the history snapshot into a readable string for CSV:
    Partner: <prompt>
    User: <chosen_response>
    """
    lines = []
    for h in history_snapshot:
        lines.append(f"Partner: {h['prompt']}")
        if h['user_response']:
            lines.append(f"User: {h['user_response']}")
    return "\n".join(lines)

def format_responses_for_csv(responses_list):
    """
    Convert the responses into a readable string separated by ' || '
    """
    resp_texts = [r.get('response_text', '') for r in responses_list]
    return " || ".join(resp_texts)

def get_transcript(data):

    transcript = data.get("transcript")
    is_final   = data.get("isFinal")
    timestamp  = data.get("timestamp")

    return (transcript,is_final,timestamp)

def fallback_response(message="Sorry, no response available due to an internal error."):
    return [{'response_text': message} for _ in range(8)]

def format_responses_for_ws(partner_prompt: str, responses: list[dict], error: str | None = None) -> dict:
    response_dict = {
        "Display": partner_prompt,
        **{f"response{i+1}": responses[i].get("response_text", "") for i in range(4)},
        **{f"turnaround{i-3}": responses[i].get("response_text", "") for i in range(4, 8)}
    }
    if error:
        response_dict["error"] = error
    return response_dict


class SessionManager:
    current_session: Optional["ConversationSession"] = None

    @classmethod
    def set_session(cls, session: "ConversationSession"):
        cls.current_session = session

    @classmethod
    def get_session(cls) -> Optional["ConversationSession"]:
        return cls.current_session

class ConversationSession:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.csv_file_path = generate_csv_filename()
        self.conversation_history = []
        self.full_conversation_history = []
        self.last_full_prompt_to_api = None
        self.response_chosen_flag = False
        self.time_responses_sent = None
        self.queue = deque()
        self.input_type_flag = None
        self.partner_prompt = ""

        initialize_csv_file(self.csv_file_path)

    async def run(self):
        await self.websocket.accept()
        logger.info("WebSocket connection accepted.")
        websocket_connect()

        try:
            while True:
                data = await self.websocket.receive_text()
                if data:
                    websocket_message_received("/ws")
                    logger.info(f"Data received from OS-DPI at {datetime.now(ET)}")

                try:
                    data_json = json.loads(data)
                    await self.handle_message(data_json)
                except json.JSONDecodeError:
                    await self.send_error("Invalid JSON format.")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await self.send_error("Internal server error.")
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected.")
            websocket_disconnect()

    async def handle_message(self, data_json):
        state = data_json.get("state", {})
        prefix = state.get("$prefix", "")
        input_type = state.get("$InputType", "")
        emotion = state.get("$Style", "")

        self.input_type_flag = self.map_input_type(input_type)

        if prefix == "Chosen":
            await self.handle_chosen_response(state)
        elif prefix == "Generate Topic Comment":
            await self.handle_topic_comment(state, emotion)
        elif prefix == "new_conv":
            await self.start_new_conversation()
        else:
            await self.send_error(f"Unexpected prefix value: {prefix}")

    def map_input_type(self, input_type):
        return {
            'Typed Utterance': '[TYPED UTTERANCE]',
            'Direct Selection': '[DIRECT SELECTION]',
            'Generated': '[GENERATED]',
            'Topic Comment': '[TOPIC COMMENT GENERATION]'
        }.get(input_type, None)

    async def handle_chosen_response(self, state):
        chosen_response = state.get("$socket", "")
        self.response_chosen_flag = True
        time_chosen_response_received = datetime.now(ET)
        chosen_response_latency = (time_chosen_response_received - self.time_responses_sent).total_seconds() if self.time_responses_sent else 0.0

        if not chosen_response:
            await self.send_error("Chosen response is empty.")
            return

        logger.info(f"Received chosen response: {chosen_response}")

        if self.queue:
            entire_partner_prompt = ' '.join(self.queue)
            self.queue.clear()
            if self.conversation_history:
                self.conversation_history[-1]['prompt'] = entire_partner_prompt
                self.full_conversation_history[-1]['prompt'] = entire_partner_prompt

        if self.conversation_history:
            await self.update_conversation_with_chosen_response(chosen_response, chosen_response_latency)
        else:
            await self.handle_initial_chosen_response(chosen_response, chosen_response_latency, state.get("$Style", ""))

    async def update_conversation_with_chosen_response(self, chosen_response, latency):
        entry = self.conversation_history[-1]
        full_entry = self.full_conversation_history[-1]
        timestamp = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S")

        if entry['user_response']:
            full_response = " ".join([entry['user_response'], chosen_response])
            entry['user_response'] = full_response
            full_entry['user_response'] = full_response
        else:
            entry['user_response'] = chosen_response
            full_entry['user_response'] = chosen_response

        await self.append_csv_entry(entry, full_entry, chosen_response, latency, timestamp)

    async def handle_initial_chosen_response(self, chosen_response, latency, emotion):
        update_history(
            self.conversation_history,
            None,
            chosen_response,
            None,
            self.full_conversation_history,
            None,
            emotion,
            None
        )
        update_full_history(self.full_conversation_history, self.conversation_history[-1], chosen_response)
        timestamp = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S")
        await self.append_csv_entry(self.conversation_history[-1], self.full_conversation_history[-1], chosen_response, latency, timestamp)

    async def handle_topic_comment(self, state, emotion):
        topic_comment = state.get("$socket", "")
        history_context = format_conversation_history_for_prompt(self.conversation_history)
        final_prompt_to_api = f"{history_context}\nTopic: {topic_comment}\n\nPlease respond accordingly."
        self.last_full_prompt_to_api = final_prompt_to_api

        responses_list, is_response_ok, error_text, api_latency = await get_responses_from_api(final_prompt_to_api)

        while len(responses_list) < 8:
            responses_list.append({'response_text': 'No response available.'})

        responses_dict = format_responses_for_ws(self.partner_prompt, responses_list, error_text if not is_response_ok else None)
        self.time_responses_sent = datetime.now(ET)

        await self.websocket.send_text(json.dumps(responses_dict))
        websocket_message_sent("/ws")
        logger.info("Generated responses sent to OS-DPI")

        update_history(
            self.conversation_history,
            self.partner_prompt,
            None,
            responses_list,
            self.full_conversation_history,
            emotion,
            api_latency
        )

    async def start_new_conversation(self):
        logger.info("Received 'new_conv' prefix, starting a new conversation.")
        self.conversation_history.clear()
        self.full_conversation_history.clear()
        self.last_full_prompt_to_api = None
        self.queue.clear()
        self.csv_file_path = generate_csv_filename()
        initialize_csv_file(self.csv_file_path)
        self.time_responses_sent = None
        logger.info(f"New CSV file created at {self.csv_file_path} for the new conversation.")
        await self.websocket.send_text(json.dumps({'state': {"$Info": "New conversation started."}}))
        websocket_message_sent("/ws")

    async def append_csv_entry(self, entry, full_entry, chosen_response, latency, timestamp):
        if not self.full_conversation_history:
            await self.send_error("Conversation history is empty.")
            return

        formatted_history = format_history_for_csv(full_entry.get('history_snapshot', []))
        formatted_responses = format_responses_for_csv(full_entry.get('responses', [])) if full_entry.get('responses') else None
        entire_conversation = format_history_for_csv(self.full_conversation_history)

        csv_entry = {
            'index': len(self.conversation_history),
            'date_time': timestamp,
            'prompt': entry['prompt'],
            'prompt_for_response': entry['prompt_for_response'],
            'history': formatted_history,
            'responses': formatted_responses,
            'chosen_response': chosen_response,
            'input_type': self.input_type_flag,
            'api_latency': entry['api_latency'],
            'chosen_response_latency': latency,
            'full_prompt_to_api': self.last_full_prompt_to_api or "",
            'full_conversation_history': entire_conversation
        }
        append_to_csv_file(self.csv_file_path, csv_entry)

    async def send_error(self, message):
        logger.error(message)
        await self.websocket.send_text(json.dumps({'error': message}))
        websocket_message_sent("/ws")

# ------------------------ WebSocket Endpoint ------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    session = ConversationSession(websocket)
    SessionManager.set_session(session)
    await session.run()

# ------------------------ CSV Download Endpoint ------------------------

@app.get("/download_csv")
async def download_csv():
    # global csv_file_path
    session = SessionManager.get_session()
    if session.csv_file_path and os.path.exists(session.csv_file_path):
        logger.info(f"CSV file found at {session.csv_file_path}. Preparing for download.")
        return FileResponse(session.csv_file_path, media_type='text/csv', filename=os.path.basename(session.csv_file_path))
    else:
        logger.error("CSV file does not exist.")
        raise HTTPException(status_code=404, detail="CSV file does not exist.")

# ------------------------ Metrics Endpoint ------------------------

@app.get("/metrics")
async def metrics_endpoint():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ------------------------ Root Endpoint ------------------------

@app.get("/")
def read_root():
    return {"message": "Welcome to the Main Server. Use appropriate endpoints to interact."}

@app.api_route("/receive_transcript", methods=["GET", "POST", "OPTIONS"])
async def receive_transcript_proxy_temp(request: Request):
    session = SessionManager.get_session()

    if not session:
        return JSONResponse(content={"status": "error", "error": "No active WebSocket session"}, status_code=400)

    if request.method == "OPTIONS":
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    if request.method == "GET":
        return PlainTextResponse("ASR test server is running.", status_code=200)

    if request.method == "POST":
        try:
            data = await request.json()
            source = data.get("source", "asr")

            if source == "asr":
                if not session.queue:
                    session.response_chosen_flag = False
                    emotion = 'friendly'
                    partner_prompt, is_final_transcript, time_dG_received = get_transcript(data)

                    if is_final_transcript:
                        logger.info(f"Partner prompt received: {partner_prompt}")
                        session.queue.append(partner_prompt)
                        prompt_for_response = partner_prompt

                        history_context = format_conversation_history_for_prompt(session.conversation_history)
                        final_prompt_to_api = f"{history_context}\nPartner: {partner_prompt}\n\nPlease respond accordingly."
                        session.last_full_prompt_to_api = final_prompt_to_api

                        responses_list, is_response_ok, error_text, api_latency = await get_responses_from_api(final_prompt_to_api)

                        while len(responses_list) < 8:
                            responses_list.append({'response_text': 'No response available.'})

                        responses_dict = format_responses_for_ws(partner_prompt, responses_list, error_text if not is_response_ok else None)

                        session.time_responses_sent = datetime.now(ET)
                        await session.websocket.send_text(json.dumps(responses_dict))
                        websocket_message_sent("/ws")
                        logger.info("Generated responses sent to OS-DPI")

                        update_history(
                            session.conversation_history,
                            partner_prompt,
                            None,
                            responses_list,
                            session.full_conversation_history,
                            prompt_for_response,
                            emotion,
                            api_latency
                        )
                else:
                    if not session.response_chosen_flag:
                        partner_prompt, is_final_transcript, time_dG_received = get_transcript(data)
                        if is_final_transcript:
                            session.queue.append(partner_prompt)
                            logger.info(f"Partner prompt received: {partner_prompt}")

            elif source == "prompt":
                rows = data.get("rows", [])
                for row in rows:
                    key = row.get("key")
                    prompt = row.get("prompt")
                    timestamp = row.get("timestamp")
                    print(f"[PROMPT] [{timestamp}] {key} → {prompt}")

            else:
                print(f"[WARN] Unknown source: {source!r}")

            return JSONResponse(content={"status": "ok"}, status_code=200)

        except Exception as e:
            logger.error(f"An error occurred while processing the transcript: {e}")
            await session.websocket.send_text(json.dumps({'error': str(e)}))
            websocket_message_sent("/ws")
            return JSONResponse(content={"status": "error", "error": str(e)}, status_code=400)


# ------------------------ Graceful Shutdown ------------------------

@app.on_event("shutdown")
async def shutdown_event():
    await client.aclose()
    logger.info("httpx client closed.")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5678, log_level="info")




