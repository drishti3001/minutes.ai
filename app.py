from flask import Flask, render_template, request
import os
import whisper
from dotenv import load_dotenv
import google.generativeai as genai
from pydub import AudioSegment
import warnings
import json

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found. Check your .env file.")
genai.configure(api_key=api_key) #type: ignore

warnings.filterwarnings("ignore")

app = Flask(__name__)
if not os.path.exists('uploads'):
    os.makedirs('uploads')

print("Loading Whisper 'medium' model (CPU)...")
whisper_model = whisper.load_model("medium", device="cpu")
print("Whisper model ready!")

def split_audio(file_path, chunk_length_ms=5*60*1000):
    audio = AudioSegment.from_file(file_path)
    chunks = []
    base_name = os.path.splitext(file_path)[0]
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i+chunk_length_ms]
        chunk_name = f"{base_name}_chunk{i//chunk_length_ms}.wav"
        chunk.export(chunk_name, format="wav")
        chunks.append(chunk_name)
    return chunks

def transcribe_audio(file_path):
    print(f"Transcribing in chunks: {file_path}")
    transcript = ""
    chunks = split_audio(file_path)
    print(f"Total chunks: {len(chunks)}")
    for idx, chunk_path in enumerate(chunks):
        print(f"Transcribing chunk {idx+1}/{len(chunks)} ...")
        result = whisper_model.transcribe(chunk_path)
        chunk_text = result.get("text", "")
        transcript += str(chunk_text).strip() + " "
    print("Transcription complete.")
    return transcript.strip(), chunks

def get_summary(transcript):
    print("Generating summary with Gemini...")
    model = genai.GenerativeModel("gemini-pro-latest") #type: ignore

    prompt = f"""
    Your task is to act as a professional meeting assistant. Analyze the following meeting transcript and extract specific information into a structured JSON format. Follow these instructions carefully:
    1.  Create a Summary: Write a detailed, one-paragraph summary. It must cover the main purpose of the meeting, the key problems or topics discussed, and the overall resolution or outcome.
    2.  Extract Key Decisions: Identify and list all formal decisions that were finalized. A key decision involves a concrete resolution, such as an approval, a rejection, a budget allocation, or a final policy choice. Do not include topics that were only discussed but not resolved. Each decision should be a complete sentence.
    3.  Extract Action Items: Identify all distinct, actionable tasks that were assigned to a person or a group. Each action item must clearly state WHO is responsible and WHAT they need to do.
    The final output must be a single, clean JSON object with three keys: "summary", "decisions", and "action_items". Do not include any text or markdown before or after the JSON object.
    Transcript:
    ---
    {transcript}
    ---
    """
    try:
        response = model.generate_content(prompt)
        clean_response_text = response.text.strip().replace("```json", "").replace("```", "")
        summary_data = json.loads(clean_response_text)
        print("Summary generation and JSON parsing complete.")
        return summary_data
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error: Failed to parse JSON or generate summary. Error: {e}")
        return {"summary": "Could not generate a summary for this transcript.", "decisions": [], "action_items": []}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audio_file' not in request.files:
        return "Error: No file part in request."
    file = request.files['audio_file']
    if not file.filename:
        return "Error: No file selected."

    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)
    
    created_chunks = []
    try:
        transcript, created_chunks = transcribe_audio(filepath)
        summary_data = get_summary(transcript) 
        return render_template('index.html', transcript=transcript, summary_data=summary_data)
    finally:
        print("Cleaning up temporary files...")
        for chunk in created_chunks:
            try:
                os.remove(chunk)
            except OSError as e:
                print(f"Error removing chunk {chunk}: {e}")
        try:
            os.remove(filepath)
            print(f"Removed original file: {filepath}")
        except OSError as e:
            print(f"Error removing original file {filepath}: {e}")

if __name__ == '__main__':
    app.run(debug=True)