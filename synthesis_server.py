import torch, time, numpy as np, sounddevice as sd, atexit, json
from TTS.api import TTS
from flask import Flask, request, jsonify
from flask_cors import CORS
from threading import Thread, Lock
import itertools

from kokoro import KPipeline

synthesis_results = {} # dictionary to store the wav audio produced by the TTS model
synthesis_lock = Lock()
synthesis_counter = itertools.count(start=1)
next_to_play = 1
text_buffer = {"text": "", "start": None, "end": None}
pending_ids = []
buffer_lock = Lock()
skipped_ids = set()
testing_logs = {
    "transcription time": [],
    "synthesis time": [],
    "transmission time": [],
    "sampling time": [],
    "playback time": [],
    "system latency": []
}

app = Flask(__name__)
CORS(app)

tts_model = None
kokoro_pipeline = None
model_type = None  # 'coqui' or 'kokoro'

sample_rate = 24000
device = "cuda" if torch.cuda.is_available() else "cpu"

@app.route("/load_model", methods=["POST"])
def load_model():
    global tts_model, kokoro_pipeline, model_type

    model_name = request.json.get("model", "kokoro")
    # model_name = request.json.get("model", "tts_models/multilingual/multi-dataset/xtts_v2")

    if "kokoro" in model_name.lower():
        kokoro_pipeline = KPipeline(lang_code='a')
        tts_model = None
        model_type = "kokoro"
        generator = kokoro_pipeline("Warm up model.", voice='af_heart', speed=1)
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

def playback_worker():
    global next_to_play
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

def split_sentence(text):
    last_idx = -1
    for punct in (".", "!", "?"):
        idx = text.rfind(punct)
        if idx > last_idx:
            last_idx = idx

    if last_idx != -1:
        return text[:last_idx+1], text[last_idx+1:].lstrip()
    return None, text

@app.route("/synthesis", methods=["POST"])
def synthesis():
    if tts_model is None and kokoro_pipeline is None:
        return jsonify({"status": "error", "message": "No model loaded."}), 400
    data = request.json
    text = data.get("transcript", "").strip()
    if not text:
        return jsonify({"status": "error", "message": "No text provided."}), 400
    
    # transmission time - how long it takes the transcription to get from the caller to the recipient
    transmission_time = data.get("recipient-posted-at") - data.get("caller-posted-at")
    testing_logs["transmission time"].append(transmission_time)
    # transcription time - how long it takes for audio to be transcribed and get to the callers web app
    transcription_time = data.get("caller-posted-at") - (data.get("connection start") + data.get("end"))
    testing_logs["transcription time"].append(transcription_time)

    global text_buffer
    with buffer_lock:
        if text_buffer["text"] == "":
            text_buffer["start"] = data.get("start")
        text_buffer["text"] += " " + text
        text_buffer["end"] = data.get("end")
        text_buffer["text"] = text_buffer["text"].strip()

        sentence, rest = split_sentence(text_buffer["text"])
        print(f"Sentence: {sentence}, Rest: {rest}")
        if sentence and (rest.strip() == ""):
            # Only synthesize if no rest text, so that we ensure we know the start/end time of the sentence
            sequence_id = next(synthesis_counter)

            sentence_start = text_buffer["start"]
            sentence_end = text_buffer["end"]

            def synthesize(text, audio_start_time, audio_end_time, sid, connection_start):
                print(f"SID: {sid} | Start: {audio_start_time} | End: {audio_end_time} | Text: {text}")
                try:
                    if model_type == "coqui":
                        wav = tts_model.tts(text=text, speaker_wav="./audio/johns_voice.wav", language="en")
                    else:
                        start_time = time.time() * 1000
                        generator = kokoro_pipeline(text, voice='af_heart', speed=1)
                        _, _, audio = next(generator)
                        end_time = time.time() * 1000
                        testing_logs["synthesis time"].append(end_time - start_time)
                        wav = (audio.detach().cpu().numpy() * 32767).astype(np.int16)
                    with synthesis_lock:
                        synthesis_results[sid] = {"wav": wav, "audio_start_time": audio_start_time + connection_start, "audio_end_time": audio_end_time+ connection_start}
                except Exception as e:
                    print(f"Synthesis error for seq {sid}:", e)

            Thread(target=synthesize, args=(sentence, sentence_start, sentence_end, sequence_id, request.json.get("connection start")), daemon=True).start()

            text_buffer["text"] = ""
            text_buffer["start"] = None
            text_buffer["end"] = None
    return jsonify({"status": "success", "message": "Text buffered, synthesis triggered if sentence complete."})

Thread(target=playback_worker, daemon=True).start()

def save_testing_logs():
    if testing_logs:  # Only if there are logs
        avg_transcription = np.mean(testing_logs["transcription time"]) if testing_logs["transcription time"] else 0
        avg_synthesis = np.mean(testing_logs["synthesis time"]) if testing_logs["synthesis time"] else 0
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

