from flask import Flask, render_template, request, jsonify
import os
import whisper
import ffmpeg
from dotenv import load_dotenv
import google.generativeai as genai
from pydub import AudioSegment
import warnings
import json
from pyannote.audio import Pipeline
import numpy as np
import re
import time 

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
hf_token = os.getenv("HF_TOKEN")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")
if not hf_token:
    raise ValueError("HF_TOKEN not found in .env file (for speaker diarization).")

genai.configure(api_key=api_key) #type:ignore
warnings.filterwarnings("ignore")

app = Flask(__name__, static_folder='static') 

if not os.path.exists('uploads'):
    os.makedirs('uploads')

print("Loading Whisper 'medium' model (CPU)...")
whisper_model = whisper.load_model("medium", device="cpu")
print("Whisper model ready!")


def prepare_audio_for_diarization(input_path):
    """
    Standardizes any audio/video file to 16kHz, 1-channel (mono) WAV
    required by the pyannote.audio model.
    """
    print(f"Standardizing audio from: {input_path}")
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    audio_filename = f"{base_name}_16khz_mono.wav"
    audio_filepath = os.path.join("uploads", audio_filename)
    
    try:
        (
            ffmpeg
            .input(input_path)
            .output(audio_filepath, acodec='pcm_s16le', ac=1, ar='16000') # Force 16kHz, mono
            .run(overwrite_output=True, quiet=True)
        )
        print(f"Standardized audio created at: {audio_filepath}")
        return audio_filepath
    except Exception as e:
        print(f"Error standardizing audio: {e}")
        return None

def diarize_audio(file_path):
    """
    Perform speaker diarization using pyannote.audio.
    Returns list of segments with speaker labels and timestamps.
    """
    print("Running speaker diarization...")
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        ) #type:ignore 
        
        diarization = pipeline(file_path)
        segments = []
        unique_speakers = set()
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "speaker": speaker,
                "start": turn.start,
                "end": turn.end
            })
            unique_speakers.add(speaker)
        print(f"Diarization done. Speakers detected: {len(unique_speakers)}")
        return segments, len(unique_speakers) # Return segments and speaker count
    except Exception as e:
        print(f"Speaker diarization failed: {e}")
        return [], 0

def clean_transcript_text(text):
    """
    Cleans raw transcribed text by removing artifacts, filler words,
    and normalizing whitespace.
    """
    if not isinstance(text, str):
        return ""

    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    filler_words = r'\b(um|uh|hmm|mhm)\b'
    text = re.sub(filler_words, '', text, flags=re.IGNORECASE)
    text = text.replace('...', ' ')
    text = text.replace('*', '')
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def transcribe_audio(file_path):
    """
    Transcribes audio file with speaker attribution.
    Returns the final transcript string and a dictionary of metrics.
    """
    print(f"Starting optimized transcription for: {file_path}")
    
    metrics = {
        "speaker_count": 0,
        "raw_segment_count": 0,
        "merged_segment_count": 0,
        "segment_reduction_percent": 0.0,
        "raw_text_length": 0,
        "clean_text_length": 0
    }

    segments, speaker_count = diarize_audio(file_path)
    transcript = ""
    metrics["speaker_count"] = speaker_count
    metrics["raw_segment_count"] = len(segments)

    if not segments:
        print("No diarization output; performing full audio transcription.")
        result = whisper_model.transcribe(file_path)
        raw_text = result.get("text", "")
        metrics["raw_text_length"] = len(raw_text)
        
        transcript = clean_transcript_text(raw_text)
        metrics["clean_text_length"] = len(transcript)
        return transcript, metrics 

    print(f"Diarization found {len(segments)} raw segments. Merging...")
    merged_segments = []
    if segments:
        current_segment = segments[0]
        for next_segment in segments[1:]:
            gap = next_segment["start"] - current_segment["end"]
            if next_segment["speaker"] == current_segment["speaker"] and gap < 1.5:
                current_segment["end"] = next_segment["end"]
            else:
                merged_segments.append(current_segment)
                current_segment = next_segment
        merged_segments.append(current_segment)
    
    metrics["merged_segment_count"] = len(merged_segments)
    if metrics["raw_segment_count"] > 0:
        reduction = (metrics["raw_segment_count"] - metrics["merged_segment_count"]) / metrics["raw_segment_count"]
        metrics["segment_reduction_percent"] = round(reduction * 100, 2)
    
    print(f"Merged into {len(merged_segments)} segments for transcription.")

    print("Loading audio file into memory for slicing...")
    try:
        audio_segment = AudioSegment.from_file(file_path)
    except Exception as e:
        print(f"Failed to load audio file {file_path}: {e}")
        return "", metrics 

    for idx, seg in enumerate(merged_segments):
        print(f"Processing merged segment {idx+1}/{len(merged_segments)} - {seg['speaker']} ({seg['start']:.2f}s → {seg['end']:.2f}s)")
        try:
            chunk = audio_segment[seg["start"] * 1000: seg["end"] * 1000]

            if len(chunk) == 0:
                print(f"Segment {idx+1} is empty (duration 0ms), skipping.")
                continue
            
            samples = np.array(chunk.get_array_of_samples()).astype(np.float32)
            samples /= chunk.max_possible_amplitude 

            result = whisper_model.transcribe(samples, language="en") 
            
            text_value = result.get("text", "")
            raw_text = " ".join(text_value) if isinstance(text_value, list) else str(text_value)
            metrics["raw_text_length"] += len(raw_text)
            
            text = clean_transcript_text(raw_text)
            metrics["clean_text_length"] += len(text)
            
            if text:
                transcript += f"{seg['speaker']}: {text}\n"

        except Exception as e:
            print(f"Error in merged segment {idx+1}: {e}")

    print("Transcription complete with speakers.")
    return transcript.strip(), metrics 


def get_summary(transcript, metrics): 
    """
    Use Gemini model to summarize and extract key details from any transcript.
    """
    print("Generating summary using Gemini...")
    model = genai.GenerativeModel("gemini-pro-latest")  # type:ignore

    prompt = f"""
    You are an expert summarization assistant. Analyze the following transcript.
    Your goal is to provide a clear and concise overview of the content.

    Please return a valid JSON object with the following structure:
    1. "summary": A concise, one-paragraph summary of the main topics.
    2. "decisions": A list of any significant conclusions, agreements, or final decisions 
       identified in the text. If there are no clear decisions, return an empty list.
    3. "action_items": A list of specific, actionable tasks or next steps mentioned. 
       If there are no clear action items, return an empty list.

    Transcript:
    ---
    {transcript}
    ---
    """

    try:
        response = model.generate_content(prompt)
        clean_response_text = response.text.strip().replace("```json", "").replace("```", "")
        summary_data = json.loads(clean_response_text)
        print("Summary generated successfully.")
        
        summary_text = summary_data.get("summary", "")
        metrics["summary_length"] = len(summary_text)
        if metrics["clean_text_length"] > 0:
            ratio = (metrics["clean_text_length"] - metrics["summary_length"]) / metrics["clean_text_length"]
            metrics["compression_ratio_percent"] = round(max(0, ratio) * 100, 2)
        else:
            metrics["compression_ratio_percent"] = 0.0
        
        return summary_data
    except Exception as e:
        print(f"Error generating summary: {e}")
        return {"summary": "Summary could not be generated.", "decisions": [], "action_items": []}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload_audio', methods=['POST'])
def upload_audio_file():
    start_time = time.time() 
    
    if 'audio_file' not in request.files:
        return "Error: No audio file found."
    file = request.files['audio_file']
    if not file.filename:
        return "Error: No file selected."

    original_filepath = os.path.join('uploads', file.filename)
    file.save(original_filepath)

    standardized_audio_path = None 
    try:
        standardized_audio_path = prepare_audio_for_diarization(original_filepath)
        if not standardized_audio_path:
            return "Error: Could not process and standardize audio file."

        transcript, metrics = transcribe_audio(standardized_audio_path)
        summary_data = get_summary(transcript, metrics)
        
        end_time = time.time()
        metrics["total_processing_time_sec"] = round(end_time - start_time, 2)
        
        return render_template(
            'index.html',
            transcript=transcript, 
            summary_data=summary_data, 
            metrics=metrics, 
            default_tab='audio'
        )
    finally:
        if standardized_audio_path and os.path.exists(standardized_audio_path):
            os.remove(standardized_audio_path)
        if os.path.exists(original_filepath):
            os.remove(original_filepath)


@app.route('/upload_video', methods=['POST'])
def upload_video_file():
    start_time = time.time() 
    
    if 'video_file' not in request.files:
        return "Error: No video file found."
    file = request.files['video_file']
    if not file.filename:
        return "Error: No file selected."

    video_filepath = os.path.join('uploads', file.filename)
    file.save(video_filepath)

    standardized_audio_path = None
    try:
        standardized_audio_path = prepare_audio_for_diarization(video_filepath)
        if not standardized_audio_path:
            return "Error: Could not extract audio from video."

        transcript, metrics = transcribe_audio(standardized_audio_path)
        summary_data = get_summary(transcript, metrics)

        end_time = time.time()
        metrics["total_processing_time_sec"] = round(end_time - start_time, 2)

        return render_template(
            'index.html', 
            transcript=transcript, 
            summary_data=summary_data, 
            metrics=metrics, 
            default_tab='video'
        )
    finally:
        if standardized_audio_path and os.path.exists(standardized_audio_path):
            os.remove(standardized_audio_path)
        if os.path.exists(video_filepath):
            os.remove(video_filepath)


@app.route('/summarize_text', methods=['POST'])
def summarize_text():
    start_time = time.time() 
    
    text_input = request.form.get('text_input')
    if not text_input:
        return "Error: No text provided."

    metrics = {
        "speaker_count": "N/A",
        "raw_segment_count": "N/A",
        "merged_segment_count": "N/A",
        "segment_reduction_percent": "N/A",
        "raw_text_length": len(text_input),
        "clean_text_length": len(text_input), 
    }
    
    transcript = text_input 
    summary_data = get_summary(transcript, metrics) 
    
    end_time = time.time()
    metrics["total_processing_time_sec"] = round(end_time - start_time, 2)
    
    return render_template(
        'index.html', 
        transcript=transcript, 
        summary_data=summary_data, 
        metrics=metrics, 
        default_tab='text'
    )
    
@app.route('/ask_question', methods=['POST'])
def ask_question():
    data = request.get_json()
    if not data or 'transcript' not in data or 'question' not in data:
        return jsonify({"error": "Missing transcript or question"}), 400

    transcript = data['transcript']
    question = data['question']

    model = genai.GenerativeModel("gemini-pro-latest") #type:ignore
    
    prompt = f"""
    You are a helpful assistant. Answer the following question based *only* on the provided transcript.
    Do not use any external knowledge.
    If the answer is not found in the transcript, state clearly: "I'm sorry, that information is not available in the transcript."

    Transcript:
    ---
    {transcript}
    ---

    Question: {question}

    Answer:
    """

    try:
        response = model.generate_content(prompt)
        answer = response.text.strip()
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"Error asking question: {e}")
        return jsonify({"error": "An error occurred while generating the answer."}), 500


if __name__ == '__main__':
    app.run(debug=True)
