#!/usr/bin/env python3
import os
import json
import asyncio
import signal
import logging
import pyaudio
import queue
import threading
from datetime import datetime
from dotenv import load_dotenv
import pytz
import webrtcvad
import websockets
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
import time

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# -------------------------
# Load Environment Variables
# -------------------------
load_dotenv()
DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY')
if not DEEPGRAM_API_KEY:
    raise ValueError("DEEPGRAM_API_KEY is not set in environment variables.")

# -------------------------
# Configuration Constants
# -------------------------
DEEPGRAM_WS_URL = (
    'wss://api.deepgram.com/v1/listen'
    '?encoding=linear16&sample_rate=16000&channels=1&model=nova-2&interim_results=true'
)
RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_DURATION_MS = 30
CHUNK = int(RATE * CHUNK_DURATION_MS / 1000)
VAD_AGGRESSIVENESS = 2
SILENCE_THRESHOLD = 5.0

MAX_RETRIES = 5
BACKOFF_FACTOR = 1

ET = pytz.timezone('US/Eastern')

# -------------------------
# Logging Configuration
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -------------------------
# Prometheus Metrics Definitions
# -------------------------
TRANSCRIPT_COUNT = Counter('transcripts_processed_total', 'Total number of transcripts processed')
DEEPGRAM_API_SUCCESS = Counter('deepgram_api_call_success_total', 'Total successful Deepgram API calls')
DEEPGRAM_API_FAILURE = Counter('deepgram_api_call_failure_total', 'Total failed Deepgram API calls')
PROCESSING_LATENCY = Histogram('transcript_processing_latency_seconds', 'Latency for processing transcripts')
QUEUE_LENGTH = Gauge('audio_queue_length', 'Number of audio chunks waiting to be processed')
WEBSOCKET_STATUS = Gauge('websocket_connection_status', 'Status of WebSocket connection (1=Up, 0=Down)')

# -------------------------
# Global Variables
# -------------------------
audio_queue = queue.Queue()
shutdown_event = threading.Event()

speech_start_time = None
speech_end_time = None
in_speech = False
in_processing = False
transcript = ""

vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
silence_frame = (b'\x00' * (2 * CHUNK))

utterance_done_event = asyncio.Event()

audio = None
stream = None
ws_global = None

final_results = {}

def reset_utterance_state():
    global speech_start_time, speech_end_time, in_speech, in_processing, transcript
    speech_start_time = None
    speech_end_time = None
    in_speech = False
    in_processing = False
    transcript = ""

def start_new_utterance():
    reset_utterance_state()
    global in_speech, in_processing, speech_start_time
    in_speech = True
    in_processing = False
    speech_start_time = time.perf_counter()

def end_utterance():
    global in_speech, in_processing, speech_end_time
    in_speech = False
    in_processing = True
    speech_end_time = time.perf_counter()

def finalize_utterance():
    global in_speech, in_processing, speech_start_time, speech_end_time, transcript, final_results

    final_received_time = time.perf_counter()
    if speech_end_time is not None and final_received_time < speech_end_time:
        final_received_time = speech_end_time

    if speech_start_time is not None and speech_end_time is not None:
        processing_latency = final_received_time - speech_end_time
        total_latency = final_received_time - speech_start_time
    else:
        processing_latency = 0.0
        total_latency = 0.0

    if processing_latency < 0:
        processing_latency = 0.0

    logger.info(f"Processing Latency (end->final): {processing_latency:.6f} s")
    logger.info(f"Total Latency (start->final): {total_latency:.6f} s")
    logger.info(f"Final Transcript: {transcript}")

    # Update Prometheus metrics
    TRANSCRIPT_COUNT.inc()
    PROCESSING_LATENCY.observe(processing_latency)
    QUEUE_LENGTH.set(audio_queue.qsize())

    final_results.clear()
    final_results["transcript"] = transcript
    final_results["processing_latency"] = processing_latency
    final_results["total_latency"] = total_latency

    reset_utterance_state()
    loop = asyncio.get_running_loop()
    loop.call_soon_threadsafe(utterance_done_event.set)

async def send_to_deepgram(ws, data):
    """
    Sends audio data to Deepgram via WebSocket and updates Prometheus metrics based on success or failure.
    """
    try:
        await ws.send(data)
        DEEPGRAM_API_SUCCESS.inc()
    except Exception as e:
        DEEPGRAM_API_FAILURE.inc()
        WEBSOCKET_STATUS.set(0)  # WebSocket is down due to failure
        logger.error(f"Deepgram API call failed: {e}")
        raise e

def audio_callback(in_data, frame_count, time_info, status_flags):
    if shutdown_event.is_set():
        return (in_data, pyaudio.paComplete)
    audio_queue.put(in_data)
    return (in_data, pyaudio.paContinue)

async def send_audio(ws):
    silence_start = None

    while not shutdown_event.is_set():
        data = await asyncio.get_running_loop().run_in_executor(None, audio_queue.get)
        if data is None:
            break

        is_speech = vad.is_speech(data, RATE)

        if is_speech:
            if not in_speech:
                start_new_utterance()
            silence_start = None
            await send_to_deepgram(ws, data)
        else:
            if in_speech:
                if silence_start is None:
                    silence_start = time.perf_counter()
                if (time.perf_counter() - silence_start) >= SILENCE_THRESHOLD:
                    end_utterance()
                    await send_to_deepgram(ws, silence_frame)
                else:
                    await send_to_deepgram(ws, silence_frame)
            else:
                await send_to_deepgram(ws, silence_frame)

    try:
        await ws.send(json.dumps({"type": "CloseStream"}))
    except Exception:
        pass

async def receive_transcript(ws):
    global transcript, in_speech, in_processing

    async for message in ws:
        if shutdown_event.is_set():
            break

        msg = json.loads(message)
        if 'request_id' in msg:
            continue

        if 'channel' in msg and 'alternatives' in msg['channel']:
            alternative = msg['channel']['alternatives'][0]
            transcript = alternative.get('transcript', '').strip()
            is_final = msg.get('is_final', False)

            if transcript and is_final:
                if in_speech and not in_processing:
                    end_utterance()
                finalize_utterance()

async def maintain_connection():
    retry_count = 0
    global ws_global

    while not shutdown_event.is_set():
        try:
            async with websockets.connect(
                DEEPGRAM_WS_URL,
                extra_headers={'Authorization': f'Token {DEEPGRAM_API_KEY}'},
                ping_interval=20,
                ping_timeout=20
            ) as ws:
                ws_global = ws
                WEBSOCKET_STATUS.set(1)  # WebSocket is up
                logger.info("Connected to Deepgram WebSocket (persistent).")

                send_task = asyncio.create_task(send_audio(ws))
                receive_task = asyncio.create_task(receive_transcript(ws))

                done, pending = await asyncio.wait(
                    [send_task, receive_task],
                    return_when=asyncio.FIRST_EXCEPTION
                )

                for task in pending:
                    task.cancel()

        except Exception as e:
            WEBSOCKET_STATUS.set(0)  # WebSocket is down due to failure
            if not shutdown_event.is_set():
                logger.error(f"WebSocket connection error: {e}")

        if shutdown_event.is_set():
            break

        retry_count += 1
        if retry_count > MAX_RETRIES:
            logger.error("Max reconnection attempts reached. Exiting maintain_connection.")
            break

        delay = BACKOFF_FACTOR * (2 ** (retry_count - 1))
        logger.info(f"Reconnecting in {delay} seconds...")
        await asyncio.sleep(delay)

async def wait_for_utterance():
    utterance_done_event.clear()
    await utterance_done_event.wait()

async def handle_request():
    start = datetime.now(ET)
    final_results.clear()
    await wait_for_utterance()

    end = datetime.now(ET)
    total_time = (end - start).total_seconds()

    final_results["time_received"] = str(start)
    final_results["time_processed"] = str(end)
    final_results["total_request_time_s"] = total_time

    return final_results

# -------------------------
# FastAPI Application Setup
# -------------------------
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global audio, stream
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        stream_callback=audio_callback
    )
    stream.start_stream()
    logger.info("Audio started at startup.")
    asyncio.create_task(maintain_connection())
    logger.info("Background task for Deepgram connection started.")

@app.on_event("shutdown")
def shutdown_event_handler():
    logger.info("Shutdown event triggered.")
    shutdown_event.set()
    audio_queue.put(None)

    if stream is not None:
        stream.stop_stream()
        stream.close()
    if audio is not None:
        audio.terminate()
    logger.info("Audio resources released on shutdown.")

def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}. Initiating shutdown...")
    shutdown_event.set()
    audio_queue.put(None)

for s in (signal.SIGINT, signal.SIGTERM):
    try:
        signal.signal(s, signal_handler)
    except AttributeError:
        pass

@app.get("/")
async def hello():
    return "hello"

@app.get("/get_audio_transcription")
async def get_text_from_audio():
    results = await handle_request()
    return JSONResponse(content=results)

# Metrics Endpoint
@app.get("/metrics")
async def metrics_endpoint():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
