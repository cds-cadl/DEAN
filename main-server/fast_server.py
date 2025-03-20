from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
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
rasp_pi_api_url = "https://humane-marmot-entirely.ngrok-free.app"
headers = {"Content-Type": "application/json"}

# ------------------------ CSV Configuration ------------------------

session_csv_dir = "session_csv_files"
os.makedirs(session_csv_dir, exist_ok=True)

# Updated CSV headers to include the full prompt sent to the API
CSV_HEADERS = [
    'index', 'date_time', 'prompt', 'history', 'responses',
    'chosen_response', 'server_to_pi_latency',
    'pi_to_server_latency', 'api_latency', 'chosen_response_latency', 'full_prompt_to_api'
]

# ------------------------ In-Memory Conversation Histories ------------------------

conversation_history = []       # Stores recent conversation states (up to last 3)
full_conversation_history = []  # Stores all conversation states

# Global variables
csv_file_path = None
time_responses_sent = None

# We will store the last full prompt to the API in conversation_history or a separate variable
last_full_prompt_to_api = None

# Eastern Time zone
ET = pytz.timezone('US/Eastern')

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

async def get_speech_to_text():
    try:
        response = await client.get(f'{rasp_pi_api_url}/get_audio_transcription', timeout=10)
        response.raise_for_status()
        data_json = response.json()
        logger.debug(f"RPi API Response: {data_json}")
        return data_json
    except httpx.RequestError as e:
        logger.error(f"Error fetching speech-to-text: {e}")
        return {}
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error while fetching speech-to-text: {e}")
        return {}

async def send_to_api_async(prompt, number_of_responses, response_types, search_mode, generate_topic_response):
    payload = {
        'prompt': prompt,
        'number_of_responses': number_of_responses,
        'response_types': response_types,
        'search_mode': search_mode,
        'topic_response': generate_topic_response
    }
    logger.info(f"Sending payload to API with prompt:\n{prompt}")
    try:
        response = await client.post(api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        response_json = response.json()
        logger.info(f"Received response from API: {response_json}")
        return response_json
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
        return {"responses": []}
    except httpx.RequestError as e:
        logger.error(f"Error sending request to API: {e}")
        return {"responses": []}

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

def update_history(history, partner_prompt, user_response, model_responses, full_history, emotion, server_to_pi_latency, pi_to_server_latency, api_latency):
    if len(history) >= 3:
        removed = history.pop(0)
        logger.debug(f"Removed oldest entry from conversation history: {removed['prompt']}")

    history.append({
        'prompt': partner_prompt,
        'user_response': user_response,
        'server_to_pi_latency': server_to_pi_latency,
        'pi_to_server_latency': pi_to_server_latency,
        'api_latency': api_latency
    })

    if model_responses is not None:
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
        if entry['prompt'] == last_convo_pair['prompt'] and entry['user_response'] is None:
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

# ------------------------ WebSocket Endpoint ------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global csv_file_path, conversation_history, full_conversation_history, time_responses_sent
    global last_full_prompt_to_api

    # Start a new conversation session: generate and initialize CSV
    csv_file_path = generate_csv_filename()
    conversation_history = []
    full_conversation_history = []
    time_responses_sent = None
    last_full_prompt_to_api = None
    initialize_csv_file(csv_file_path)

    # Accept the WebSocket connection
    await websocket.accept()
    logger.info("WebSocket connection accepted.")
    websocket_connect()

    try:
        while True:
            data = await websocket.receive_text()
            if data:
                websocket_message_received("/ws")
                time_received_osdpi = datetime.now(ET)
                logger.info(f"Data received from OS-DPI at {time_received_osdpi}")

            try:
                data_json = json.loads(data)
                state = data_json.get("state", {})
                prefix = state.get("$prefix", "")
                emotion = state.get("$Style", "")

                if prefix == 'prompt':
                    # Check for incomplete entry
                    incomplete_message = check_last_entry(conversation_history)

                    # Send request to RPi for speech-to-text
                    time_server_sent_to_rasp_pi = datetime.now(ET)
                    rasp_pi_data = await get_speech_to_text()
                    time_server_received_from_rasp_pi = datetime.now(ET)

                    # Extract prompt from partner
                    partner_prompt = rasp_pi_data.get('transcript', '').strip()
                    if not partner_prompt:
                        logger.error("No prompt text received from RPi API.")
                        await websocket.send_text(json.dumps({'error': 'No prompt text received.'}))
                        websocket_message_sent("/ws")
                        continue

                    # Parse timestamps from RPi API
                    try:
                        time_rasp_pi_received_from_server = parse(rasp_pi_data.get('time_received', datetime.now().isoformat())).astimezone(ET)
                    except Exception:
                        time_rasp_pi_received_from_server = datetime.now(ET)

                    try:
                        time_rasp_pi_sent_to_server = parse(rasp_pi_data.get('time_processed', datetime.now().isoformat())).astimezone(ET)
                    except Exception:
                        time_rasp_pi_sent_to_server = datetime.now(ET)

                    server_to_pi_latency = (time_rasp_pi_received_from_server - time_server_sent_to_rasp_pi).total_seconds()
                    pi_to_server_latency = (time_server_received_from_rasp_pi - time_rasp_pi_sent_to_server).total_seconds()

                    logger.info(f"Partner prompt received: {partner_prompt}")
                    logger.debug(f"Latencies - Server to Pi: {server_to_pi_latency}s, Pi to Server: {pi_to_server_latency}s")

                    # Echo prompt back to UI
                    await websocket.send_text(json.dumps({'state': {"$Display": partner_prompt}}))
                    websocket_message_sent("/ws")

                    # Add conversation history to the prompt for context
                    history_context = format_conversation_history_for_prompt(conversation_history)
                    final_prompt_to_api = f"{history_context}\nPartner: {partner_prompt}\n\nPlease respond accordingly."
                    # Store it globally so we can use it later when chosen response is picked
                    last_full_prompt_to_api = final_prompt_to_api

                    # Send prompt to LightRAG API
                    api_request_start_time = datetime.now(ET)
                    response = await send_to_api_async(
                        final_prompt_to_api,
                        number_of_responses=8,
                        response_types=["positive", 
                                        "negative", 
                                        "positive with more variation in response", 
                                        "negative with more variation in response",
                                        "a follow-up question with positive intent",
                                        "a follow-up question with negative intent",
                                        "a follow-up question with positive intent and more response variation",
                                        "a follow-up question with positive intent and more response variation"],
                        search_mode="naive",
                        generate_topic_response=False
                    )
                    api_request_end_time = datetime.now(ET)
                    api_latency = (api_request_end_time - api_request_start_time).total_seconds()

                    responses_list = response.get('responses', [])
                    # Ensure at least 2 responses
                    while len(responses_list) < 2:
                        responses_list.append({'response_text': 'No response available.'})

                    # Construct response dictionary
                    responses_dict = {
                        'Display': partner_prompt,
                        'response1': responses_list[0].get('response_text', ''),
                        'response2': responses_list[1].get('response_text', ''),
                        'response3': responses_list[2].get('response_text', ''),
                        'response4': responses_list[3].get('response_text', ''),
                        'turnaround1': responses_list[4].get('response_text', ''),
                        'turnaround2': responses_list[5].get('response_text', ''),
                        'turnaround3': responses_list[6].get('response_text', ''),
                        'turnaround4': responses_list[7].get('response_text', '')
                    }

                    if incomplete_message:
                        responses_dict['warning'] = incomplete_message

                    time_responses_sent = datetime.now(ET)
                    await websocket.send_text(json.dumps(responses_dict))
                    websocket_message_sent("/ws")

                    # Update conversation histories with partner prompt and no chosen response yet
                    update_history(
                        conversation_history,
                        partner_prompt,
                        None,
                        responses_list,
                        full_conversation_history,
                        emotion,
                        server_to_pi_latency,
                        pi_to_server_latency,
                        api_latency
                    )

                elif prefix == 'Chosen':
                    chosen_response = state.get("$socket", "")
                    time_chosen_response_received = datetime.now(ET)
                    chosen_response_latency = (time_chosen_response_received - time_responses_sent).total_seconds() if time_responses_sent else 0.0

                    if chosen_response:
                        logger.info(f"Received chosen response: {chosen_response}")

                        if conversation_history and conversation_history[-1]['user_response'] is not None: #turnaround
                            # Update the last conversation pair with user's chosen response
                            answer = conversation_history[-1]['user_response']
                            question = chosen_response
                            conversation_history[-1]['user_response'] = " ".join([answer,question])
                            update_full_history(full_conversation_history, conversation_history[-1], chosen_response)
                            timestamp = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S")

                            if not full_conversation_history:
                                logger.error("Full conversation history is empty. Cannot append to CSV.")
                                await websocket.send_text(json.dumps({'error': 'Conversation history is empty.'}))
                                websocket_message_sent("/ws")
                                continue

                            latest_full_entry = full_conversation_history[-1]

                            # Format history and responses for CSV
                            formatted_history = format_history_for_csv(latest_full_entry.get('history_snapshot', []))
                            formatted_responses = format_responses_for_csv(latest_full_entry.get('responses', []))

                            csv_entry = {
                                'index': len(conversation_history),
                                'date_time': timestamp,
                                'prompt': conversation_history[-1]['prompt'],  # Partner prompt
                                'history': formatted_history,
                                'responses': formatted_responses,
                                'chosen_response': chosen_response,
                                'server_to_pi_latency': conversation_history[-1]['server_to_pi_latency'],
                                'pi_to_server_latency': conversation_history[-1]['pi_to_server_latency'],
                                'api_latency': conversation_history[-1]['api_latency'],
                                'chosen_response_latency': chosen_response_latency,
                                'full_prompt_to_api': last_full_prompt_to_api if last_full_prompt_to_api else ""
                            }
                            append_to_csv_file(csv_file_path, csv_entry)

                        elif conversation_history and conversation_history[-1]['user_response'] is None:
                            # Update the last conversation pair with user's chosen response
                            conversation_history[-1]['user_response'] = chosen_response
                            update_full_history(full_conversation_history, conversation_history[-1], chosen_response)
                            timestamp = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S")

                            if not full_conversation_history:
                                logger.error("Full conversation history is empty. Cannot append to CSV.")
                                await websocket.send_text(json.dumps({'error': 'Conversation history is empty.'}))
                                websocket_message_sent("/ws")
                                continue

                            latest_full_entry = full_conversation_history[-1]

                            # Format history and responses for CSV
                            formatted_history = format_history_for_csv(latest_full_entry.get('history_snapshot', []))
                            formatted_responses = format_responses_for_csv(latest_full_entry.get('responses', []))

                            csv_entry = {
                                'index': len(conversation_history),
                                'date_time': timestamp,
                                'prompt': conversation_history[-1]['prompt'],  # Partner prompt
                                'history': formatted_history,
                                'responses': formatted_responses,
                                'chosen_response': chosen_response,
                                'server_to_pi_latency': conversation_history[-1]['server_to_pi_latency'],
                                'pi_to_server_latency': conversation_history[-1]['pi_to_server_latency'],
                                'api_latency': conversation_history[-1]['api_latency'],
                                'chosen_response_latency': chosen_response_latency,
                                'full_prompt_to_api': last_full_prompt_to_api if last_full_prompt_to_api else ""
                            }
                            append_to_csv_file(csv_file_path, csv_entry)
                        else:
                            logger.error("Chosen response received without a corresponding prompt.")
                            await websocket.send_text(json.dumps({'error': 'No corresponding prompt for chosen response.'}))
                            websocket_message_sent("/ws")
                    else:
                        logger.error("No chosen response found in the received data.")
                        await websocket.send_text(json.dumps({'error': 'Chosen response is empty.'}))
                        websocket_message_sent("/ws")

                elif prefix == 'Generate Topic Comment':
                    
                    topic_comment = state.get("$socket", "")

                    # Add conversation history to the prompt for context
                    history_context = format_conversation_history_for_prompt(conversation_history)
                    final_prompt_to_api = f"{history_context}\nTopic: {topic_comment}\n\nPlease respond accordingly."
                    # Store it globally so we can use it later when chosen response is picked
                    last_full_prompt_to_api = final_prompt_to_api

                    # Send prompt to LightRAG API
                    api_request_start_time = datetime.now(ET)
                    response = await send_to_api_async(
                        final_prompt_to_api,
                        number_of_responses=4,
                        response_types=["positive", "negative", "a positive response on the given topic along with an additional question as a follow up", "a negative response on the given topic along with an additional question as a follow up"],
                        search_mode="naive",
                        generate_topic_response=True
                    )
                    api_request_end_time = datetime.now(ET)
                    api_latency = (api_request_end_time - api_request_start_time).total_seconds()

                    responses_list = response.get('responses', [])
                    # Ensure at least 2 responses
                    while len(responses_list) < 2:
                        responses_list.append({'response_text': 'No response available.'})

                    # Construct response dictionary
                    responses_dict = {
                        'Display': partner_prompt,
                        'response1': responses_list[0].get('response_text', ''),
                        'response2': responses_list[1].get('response_text', ''),
                        'response3': responses_list[2].get('response_text', ''),
                        'response4': responses_list[3].get('response_text', '')
                    }

                    if incomplete_message:
                        responses_dict['warning'] = incomplete_message

                    time_responses_sent = datetime.now(ET)
                    await websocket.send_text(json.dumps(responses_dict))
                    websocket_message_sent("/ws")

                    # Update conversation histories with partner prompt and no chosen response yet
                    update_history(
                        conversation_history,
                        partner_prompt,
                        None,
                        responses_list,
                        full_conversation_history,
                        emotion,
                        server_to_pi_latency,
                        pi_to_server_latency,
                        api_latency
                    )

                elif prefix == 'new_conv':
                    logger.info("Received 'new_conv' prefix, starting a new conversation.")
                    conversation_history.clear()
                    full_conversation_history.clear()
                    last_full_prompt_to_api = None

                    # Reinitialize CSV file for a new conversation session
                    csv_file_path = generate_csv_filename()
                    initialize_csv_file(csv_file_path)
                    time_responses_sent = None
                    logger.info(f"New CSV file created at {csv_file_path} for the new conversation.")
                    await websocket.send_text(json.dumps({'state': {"$Info": "New conversation started."}}))
                    websocket_message_sent("/ws")

                else:
                    logger.error(f"Unexpected prefix value: {prefix}")
                    await websocket.send_text(json.dumps({'error': f"Unexpected prefix value: {prefix}"}))
                    websocket_message_sent("/ws")

            except json.JSONDecodeError:
                logger.error("Invalid JSON received.")
                await websocket.send_text(json.dumps({'error': 'Invalid JSON format.'}))
                websocket_message_sent("/ws")
            except Exception as e:
                logger.error(f"An error occurred while processing the message: {e}")
                await websocket.send_text(json.dumps({'error': 'Internal server error.'}))
                websocket_message_sent("/ws")
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
        websocket_disconnect()

# ------------------------ CSV Download Endpoint ------------------------

@app.get("/download_csv")
async def download_csv():
    global csv_file_path
    if csv_file_path and os.path.exists(csv_file_path):
        logger.info(f"CSV file found at {csv_file_path}. Preparing for download.")
        return FileResponse(csv_file_path, media_type='text/csv', filename=os.path.basename(csv_file_path))
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

# ------------------------ Graceful Shutdown ------------------------

@app.on_event("shutdown")
async def shutdown_event():
    await client.aclose()
    logger.info("httpx client closed.")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5678, log_level="info")




