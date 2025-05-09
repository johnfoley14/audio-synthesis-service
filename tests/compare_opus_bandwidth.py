import subprocess
from pydub import AudioSegment
import threading
import io

print("loading audio...")
# Load MP3 audio
audio = AudioSegment.from_mp3("../../transcription-service/tests/test_audio.mp3")

print(f"Loaded audio: {len(audio) / 1000:.2f} seconds, "
      f"{audio.frame_rate} Hz, {audio.channels} channels")

frame_duration_ms = 20
frame_size = int(audio.frame_rate * (frame_duration_ms / 1000)) * audio.frame_width * audio.channels
pcm_data = audio.raw_data
total_frames = len(pcm_data) // frame_size
print(f"Total frames: {total_frames}, Frame size: {frame_size} bytes")

# Start ffmpeg
print("Starting ffmpeg encoder...")
ffmpeg_proc = subprocess.Popen(
    [
        "ffmpeg",
        "-f", "s16le",
        "-ar", str(audio.frame_rate),
        "-ac", str(audio.channels),
        "-i", "pipe:0",
        "-c:a", "libopus",
        "-b:a", "24k",
        "-f", "ogg",
        "pipe:1"
    ],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL
)

# Thread to drain stdout
compressed_chunks = []

def read_output():
    while True:
        chunk = ffmpeg_proc.stdout.read(4096)
        if not chunk:
            break
        compressed_chunks.append(chunk)

print("Starting output reader thread...")
reader_thread = threading.Thread(target=read_output)
reader_thread.start()

# Stream audio frames
print("Streaming audio chunks into ffmpeg...")

for i in range(total_frames):
    start = i * frame_size
    end = start + frame_size
    chunk = pcm_data[start:end]

    if len(chunk) < frame_size:
        chunk += b'\x00' * (frame_size - len(chunk))

    ffmpeg_proc.stdin.write(chunk)
    ffmpeg_proc.stdin.flush()

# Clean up
print("Finished streaming. Closing ffmpeg input...")
ffmpeg_proc.stdin.close()
reader_thread.join()
ffmpeg_proc.stdout.close()
ffmpeg_proc.wait()

# Report result
compressed_data = b''.join(compressed_chunks)
print("Encoding complete.")
print(f"Simulated streaming compressed size: {len(compressed_data)} bytes")

folders = ['run1_tiny', 'run2_tiny', 'run3_tiny', 'run1_small', 'run2_small', 'run3_small']

for folder in folders:

    with open(f"./test_results/{folder}/output_transcript.txt", 'r', encoding='utf-8') as f:
        content = f.read()
        char_count = len(content)
        print(f"{folder}: {char_count} characters")
