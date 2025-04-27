import torch, time, numpy as np, sounddevice as sd
from TTS.api import TTS
from flask import Flask, request, jsonify
from flask_cors import CORS
from threading import Thread, Lock
import itertools

from kokoro import KPipeline  # Import Kokoro

synthesis_results = {}
synthesis_lock = Lock()
synthesis_counter = itertools.count(start=1)
next_to_play = 1
text_buffer = ""
pending_ids = []
buffer_lock = Lock()
skipped_ids = set()

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
        return jsonify({"status": "success", "message": "Kokoro model loaded."})
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
                wav = synthesis_results.pop(next_to_play)
                next_to_play += 1
            else:
                wav = None

        if wav is not None:
            try:
                sd.play(wav, samplerate=sample_rate)
                sd.wait()
            except Exception as e:
                print("Playback error:", e)
        else:
            time.sleep(0.01)

def split_first_sentence(text):
    for punct in (".", "!", "?"):
        idx = text.find(punct)
        if idx != -1:
            return text[:idx+1], text[idx+1:].lstrip()
    return None, text  # No complete sentence yet

@app.route("/synthesis", methods=["POST"])
def synthesis():
    if tts_model is None and kokoro_pipeline is None:
        return jsonify({"status": "error", "message": "No model loaded."}), 400

    text = request.json.get("transcript", "").strip()
    if not text:
        return jsonify({"status": "error", "message": "No text provided."}), 400

    global text_buffer
    with buffer_lock:
        text_buffer += " " + text
        text_buffer = text_buffer.strip()

        while True:
            sentence, rest = split_first_sentence(text_buffer)
            if sentence:
                sequence_id = next(synthesis_counter)

                def synthesize(text, sid):
                    print(f"SID: {sid} - Starting synthesis for text: {text}")
                    try:
                        if model_type == "coqui":
                            wav = tts_model.tts(text=text, speaker_wav="./audio/johns_voice.wav", language="en")
                        else:
                            start_time = time.time()
                            generator = kokoro_pipeline(text, voice='af_heart', speed=1)
                            _, _, audio = next(generator)
                            end_time = time.time()
                            print(f"Kokoro synthesis time: {end_time - start_time:.2f} seconds")
                            wav = (audio.detach().cpu().numpy() * 32767).astype(np.int16)
                        with synthesis_lock:
                            synthesis_results[sid] = wav
                    except Exception as e:
                        print(f"Synthesis error for seq {sid}:", e)

                Thread(target=synthesize, args=(sentence, sequence_id), daemon=True).start()

                text_buffer = rest
            else:
                break

    return jsonify({"status": "success", "message": "Text buffered, synthesis triggered if sentence complete."})


Thread(target=playback_worker, daemon=True).start()

if __name__ == "__main__":
    app.run(debug=True)