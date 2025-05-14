import torch, time, numpy as np, sounddevice as sd, atexit, json
from TTS.api import TTS
from flask import Flask, request, jsonify
from flask_cors import CORS
from threading import Thread, Lock
import itertools

from kokoro import KPipeline

# Define global variables used for synthesis and concurrency safety
synthesis_results = {} # dictionary to store the wav audio produced by the TTS model
synthesis_lock = Lock()
synthesis_counter = itertools.count(start=1)
next_to_play = 1
text_buffer = {"text": "", "start": None, "end": None}
pending_ids = []
buffer_lock = Lock()
skipped_ids = set()
# Dictionary to store testing logs
testing_logs = {
    "transcription time": {},
    "synthesis time": {},
    "transmission time": [],
    "sampling time": [],
    "playback time": [],
    "system latency": []
}

# Initialize Flask app and enables CORS
app = Flask(__name__)
CORS(app)

# Global variables for TTS model and pipeline
tts_model = None
kokoro_pipeline = None
model_type = None  # 'coqui' or 'kokoro'
total_queued_audio_duration = 0  # in seconds
voice = "af_heart"  # Default voice

sample_rate = 24000
device = "cuda" if torch.cuda.is_available() else "cpu"

# Allows users to load the model they want to use
@app.route("/load_model", methods=["POST"])
def load_model():
    global tts_model, kokoro_pipeline, model_type
    model_name = request.json.get("model", "kokoro")

    if "kokoro" in model_name.lower():
        kokoro_pipeline = KPipeline(lang_code='a')
        tts_model = None
        model_type = "kokoro"
        # Warm the model to avoid latency on first request
        warmup_text = "Warm up model. This is a warm up setence to initialize the model and weights. It should help reduce the latency for the first request later."
        generator = kokoro_pipeline(warmup_text, voice='af_heart', speed=1)
        _, _, _ = next(generator)
        return jsonify({"status": "success", "message": "Kokoro model loaded and warmed up."})
    else:
        if tts_model is not None and model_name == getattr(tts_model, 'model_name', None):
            return jsonify({"status": "success", "message": f"Model {model_name} already loaded."})
        try:
            tts_model = TTS(model_name).to(device)
            kokoro_pipeline = None
            model_type = "coqui"
            return jsonify({"status": "success", "message": f"Model {model_name} loaded."})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

# Worker to read synthesizied audio and play it in order
def playback_worker():
    global next_to_play, total_queued_audio_duration
    while True:
        with synthesis_lock:
            while next_to_play in skipped_ids:
                skipped_ids.remove(next_to_play)
                next_to_play += 1

            if next_to_play in synthesis_results:
                synthesis_result = synthesis_results.pop(next_to_play)
                wav = synthesis_result["wav"]
                audio_start_time = synthesis_result["audio_start_time"]
                audio_end_time = synthesis_result["audio_end_time"]
                duration_sec = len(wav) / sample_rate
                total_queued_audio_duration = max(0, total_queued_audio_duration - duration_sec)
                next_to_play += 1
            else:
                wav = None

        if wav is not None:
            try:
                playback_start_time = time.time() * 1000
                sd.play(wav, samplerate=sample_rate)
                sd.wait()
                playback_end_time = time.time() * 1000

                average_delay = ((playback_start_time - audio_start_time) + (playback_end_time - audio_end_time)) / 2
                testing_logs["playback time"].append(playback_end_time - playback_start_time)
                testing_logs["system latency"].append(average_delay)
            except Exception as e:
                print("Playback error:", e)
        else:
            time.sleep(0.05)

# Helper function to split text into sentences
def split_sentence(text):
    first_idx = -1
    for punct in (".", "!", "?"):
        idx = text.find(punct)
        if idx != -1 and (first_idx == -1 or idx < first_idx):
            first_idx = idx

    if first_idx != -1:
        return text[:first_idx+1], text[first_idx+1:].lstrip()
    return None, text

# Synthesis thread to create audio from text and add it to the queue
def synthesize(text, audio_start_time, audio_end_time, sid, connection_start):
    global total_queued_audio_duration
    print(f"SID: {sid} | Start: {audio_start_time} | End: {audio_end_time} | Text: {text}")
    try:
        if model_type == "coqui":
            wav = tts_model.tts(text=text, speaker_wav="./audio/johns_voice.wav", language="en")
        else:
            # Choose speed and voice based on queue length
            with synthesis_lock:
                queued_seconds = total_queued_audio_duration
            speed = 1.3 if queued_seconds > 10 else 1.1

            start_time = time.time() * 1000
            generator = kokoro_pipeline(text, voice=voice, speed=speed)
            _, _, audio = next(generator)
            end_time = time.time() * 1000
            testing_logs["synthesis time"][text] = (end_time - start_time, speed)

            wav = (audio.detach().cpu().numpy() * 32767).astype(np.int16)
            audio_duration_sec = len(wav) / sample_rate

        with synthesis_lock:
            total_queued_audio_duration += audio_duration_sec
            synthesis_results[sid] = {
                "wav": wav,
                "audio_start_time": audio_start_time + connection_start,
                "audio_end_time": audio_end_time + connection_start
            }
    except Exception as e:
        print(f"Synthesis error for seq {sid}:", e)

# Endpoint to handle synthesis requests and set complete sentences for synthesis
@app.route("/synthesis", methods=["POST"])
def synthesis():
    # Check if the model is loaded
    if tts_model is None and kokoro_pipeline is None:
        return jsonify({"status": "error", "message": "No model loaded."}), 400
    data = request.json
    text = data.get("transcript", "").strip()
    if not text:
        return jsonify({"status": "error", "message": "No text provided."}), 400
    
    # transmission time - how long it takes the transcription to get from the caller to the recipient
    transmission_time = data.get("recipient-posted-at") - (data.get("caller-posted-at") - 700)
    testing_logs["transmission time"].append(transmission_time)
    # transcription time - how long it takes for audio to be transcribed and get to the callers web app
    transcription_time = data.get("caller-posted-at") - (data.get("connection start") + data.get("end"))
    testing_logs["transcription time"][text] = transcription_time

    global text_buffer
    with buffer_lock:
        if text_buffer["text"] == "":
            text_buffer["start"] = data.get("start")
        text_buffer["text"] += " " + text
        text_buffer["end"] = data.get("end")
        text_buffer["text"] = text_buffer["text"].strip()

        while True:
            sentence, rest = split_sentence(text_buffer["text"])
            if not sentence:
                break

            duration_ms = text_buffer["end"] - text_buffer["start"]
            total_chars = len(text_buffer["text"])
            sentence_chars = len(sentence)

            estimated_sentence_end = text_buffer["start"] + int((sentence_chars / total_chars) * duration_ms)

            sequence_id = next(synthesis_counter)
            Thread(
                target=synthesize,
                args=(sentence, text_buffer["start"], estimated_sentence_end, sequence_id, request.json.get("connection start")-700),
                daemon=True
            ).start()

            if not rest.strip():
                text_buffer["text"] = ""
                text_buffer["start"] = None
                break
            else:
                text_buffer["text"] = rest.strip()
                text_buffer["start"] = estimated_sentence_end

    return jsonify({"status": "success", "message": "Text buffered, synthesis triggered if sentence complete."})

Thread(target=playback_worker, daemon=True).start()

# Endpoint set the voice for the TTS model
@app.route("/set_voice", methods=["POST"])
def set_voice():
    global voice
    allowed_voices = {"af_heart", "af_bella", "am_fenrir", "am_michael"}  # Example list
    requested_voice = request.json.get("voice", "af_heart")

    if requested_voice in allowed_voices:
        voice = requested_voice
    else:
        voice = "af_heart"
    return jsonify({"status": "success", "selected_voice": voice})

def save_testing_logs():
    if testing_logs:  # Only if there are logs
        avg_transcription = np.mean(list(testing_logs["transcription time"].values())) if testing_logs["transcription time"] else 0
        avg_synthesis = np.mean(list(testing_logs["synthesis time"].values())) if testing_logs["synthesis time"] else 0
        avg_transmission = np.mean(testing_logs["transmission time"]) if testing_logs["transmission time"] else 0
        avg_playback = np.mean(testing_logs["playback time"]) if testing_logs["playback time"] else 0
        avg_latency = np.mean(testing_logs["system latency"]) if testing_logs["system latency"] else 0

        print("\n=== Test Results Summary ===")
        print(f"Average Transcription Time: {avg_transcription:.3f} ms")
        print(f"Average Synthesis Time: {avg_synthesis:.3f} ms")
        print(f"Average Transmission Time: {avg_transmission:.3f} ms")
        print(f"Average Playback Time: {avg_playback:.3f} ms")
        print(f"Average System Latency: {avg_latency:.3f} ms")

        # Save to file
        with open("testing_logs.json", "w") as f:
            json.dump(testing_logs, f, indent=4)
        print("Testing logs saved to testing_logs.json")

if __name__ == "__main__":
    atexit.register(save_testing_logs)
    app.run(debug=False)

