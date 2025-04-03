import os
import json
import time
import numpy as np
import tempfile
import wave
import re
import difflib
import traceback
import logging
import sys
from io import BytesIO
import base64
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, Response
import google.generativeai as genai
import random
USE_AI_API = True
GEMINI_API_KEY = "AIzaSyADIlKIg1fOMoy083UzHOrKqkgHADk__A8"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("karaoke_app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("KaraokeApp")

app = Flask(__name__)

# Configurations - use absolute path
LYRICS_DIR = r"D:\KARAPP\synchronized_lyrics"
AUDIO_DIR = r"D:\KARAPP\audio_files"

logger.info(f"Looking for lyrics in: {LYRICS_DIR}")
logger.info(f"Audio directory: {AUDIO_DIR}")

# Initialize speech recognition model
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Load Wav2Vec2 model and processor
logger.info("Loading speech recognition model (this may take a moment)...")
try:
    wav2vec2_model_name = "facebook/wav2vec2-large-960h-lv60-self"
    wav2vec2_processor = Wav2Vec2Processor.from_pretrained(wav2vec2_model_name)
    wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(wav2vec2_model_name).to(device)
    wav2vec2_model.eval()
    logger.info("Speech recognition model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading speech recognition model: {e}")
    logger.error("Speech evaluation features will not be available")
    wav2vec2_processor = None
    wav2vec2_model = None


# Main routes
@app.route('/')
def index():
    """Serve the main karaoke web application page"""
    return render_template('index.html')


@app.route('/evaluation')
def evaluation_page():
    """Serve the singing evaluation page"""
    return render_template('evaluation.html')


@app.route('/songs')
def get_songs():
    """Get the list of available synchronized songs"""
    logger.info("GET /songs endpoint called")
    songs = []

    try:
        # Create directories if they don't exist
        os.makedirs(LYRICS_DIR, exist_ok=True)

        if not os.path.exists(LYRICS_DIR):

            logger.warning("Lyrics directory does not exist!")
            return jsonify([])

        # List all JSON files in the lyrics directory
        for filename in os.listdir(LYRICS_DIR):
            if filename.endswith('.json'):
                file_path = os.path.join(LYRICS_DIR, filename)

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        song_data = json.load(f)

                    # Extract basic song info
                    if 'title' in song_data and 'artist' in song_data:
                        # Check if audio file exists
                        audio_file = song_data.get('audio_file', '')
                        audio_exists = os.path.exists(audio_file) if audio_file else False

                        songs.append({
                            'id': filename.replace('.json', ''),
                            'title': song_data.get('title', 'Unknown Title'),
                            'artist': song_data.get('artist', 'Unknown Artist'),
                            'audio_file': audio_file if audio_exists else '',
                            'has_audio': audio_exists
                        })
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")

        # Sort by artist and title
        songs.sort(key=lambda x: (x['artist'], x['title']))
        logger.info(f"Returning {len(songs)} songs")
        return jsonify(songs)

    except Exception as e:
        logger.error(f"Error listing songs: {e}")
        return jsonify([])


@app.route('/lyrics/<song_id>')
def get_lyrics(song_id):
    """Get the lyrics data for a specific song"""
    file_path = os.path.join(LYRICS_DIR, f"{song_id}.json")

    if not os.path.exists(file_path):
        return jsonify({'error': 'Song not found'}), 404

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            song_data = json.load(f)

        # Handle the synchronized_lyrics format
        if 'synchronized_lyrics' in song_data and not 'lyrics' in song_data:
            song_data['lyrics'] = song_data['synchronized_lyrics']

        # Fix audio path if needed
        if 'audio_file' in song_data and song_data['audio_file']:
            audio_file = song_data['audio_file']

            # Fix relative paths
            if not os.path.isabs(audio_file):
                audio_filename = os.path.basename(audio_file.replace('\\', '/'))
                absolute_audio_path = os.path.join(AUDIO_DIR, audio_filename)

                if os.path.exists(absolute_audio_path):
                    song_data['audio_file'] = absolute_audio_path
                else:
                    # Try to find the file in the audio directory
                    for root, dirs, files in os.walk(AUDIO_DIR):
                        for file in files:
                            if audio_filename in file:
                                song_data['audio_file'] = os.path.join(root, file)
                                break

        return jsonify(song_data)

    except Exception as e:
        logger.error(f"Error getting lyrics: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/audio/<song_id>')
def get_audio(song_id):
    """Get the audio file for a specific song"""
    file_path = os.path.join(LYRICS_DIR, f"{song_id}.json")

    if not os.path.exists(file_path):
        logger.warning(f"Song not found: {song_id}")
        return jsonify({'error': 'Song not found'}), 404

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            song_data = json.load(f)

        audio_file = song_data.get('audio_file', '')
        # Get the filename from the path
        audio_filename = os.path.basename(audio_file.replace('\\', '/'))

        # Define possible paths where the audio might be
        possible_paths = [
            audio_file,
            os.path.join(os.getcwd(), audio_file.replace('\\', '/')),
            os.path.join(os.getcwd(), 'audio_files', audio_filename),
            os.path.join(os.path.dirname(LYRICS_DIR), audio_file.replace('\\', '/')),
            os.path.join(AUDIO_DIR, audio_filename),
            os.path.join(r"D:\KARAPP", audio_file.replace('\\', '/'))
        ]

        # Try all possible paths
        audio_path = None
        for path in possible_paths:
            if os.path.exists(path):
                audio_path = path
                break

        if not audio_path:
            # Last resort: search for the file in the project directory tree
            project_root = os.getcwd()
            for root, dirs, files in os.walk(project_root):
                for file in files:
                    if file == audio_filename:
                        audio_path = os.path.join(root, file)
                        break
                if audio_path:
                    break

        if not audio_path:
            logger.warning(f"Audio file not found: {audio_file}")
            return jsonify({'error': 'Audio file not found'}), 404

        # Return the audio file
        return send_file(audio_path)

    except Exception as e:
        logger.error(f"Error serving audio: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files like CSS and JavaScript"""
    return send_from_directory('static', filename)


@app.route('/api/lyrics/<song_id>', methods=['GET'])
def get_song_lyrics_text(song_id):
    """Get plain text lyrics for a specific song"""
    file_path = os.path.join(LYRICS_DIR, f"{song_id}.json")

    if not os.path.exists(file_path):
        return jsonify({'error': 'Song not found'}), 404

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            song_data = json.load(f)

        # Handle different lyric formats - extract just the text
        lyrics_text = ""
        if 'lyrics' in song_data:
            if isinstance(song_data['lyrics'], list):
                # Handle list format with timestamps
                lyrics_text = ' '.join([line.get('text', '') for line in song_data['lyrics']])
            else:
                # Handle plain text format
                lyrics_text = song_data['lyrics']
        elif 'synchronized_lyrics' in song_data:
            if isinstance(song_data['synchronized_lyrics'], list):
                lyrics_text = ' '.join([line.get('text', '') for line in song_data['synchronized_lyrics']])
            else:
                lyrics_text = song_data['synchronized_lyrics']
        else:
            return jsonify({'error': 'No lyrics found in song data'}), 404

        return jsonify({
            'id': song_id,
            'title': song_data.get('title', 'Unknown Title'),
            'artist': song_data.get('artist', 'Unknown Artist'),
            'lyrics': lyrics_text
        })

    except Exception as e:
        logger.error(f"Error getting song lyrics: {e}")
        return jsonify({'error': str(e)}), 500


def compare_lyrics(sung_text, original_lyrics):
    """Comparison between sung text and original lyrics checking correct order"""
    print("\n==== LYRICS COMPARISON ====")
    print(f"SUNG: {sung_text}")
    print(f"ORIG: {original_lyrics}")

    # Normalize both texts - lowercase and remove punctuation
    def normalize_text(text):
        text = text.lower()
        for char in ".,!?;:\"'()[]{}-_":
            text = text.replace(char, ' ')
        return ' '.join(text.split())

    # Normalize texts
    sung_text = normalize_text(sung_text)
    original_lyrics = normalize_text(original_lyrics)

    print(f"\nNormalized sung text: {sung_text}")
    print(f"Normalized original: {original_lyrics}")

    # Split into words
    sung_words = sung_text.split()
    original_words = original_lyrics.split()

    print(f"\nSung words: {len(sung_words)}")
    print(f"Original words: {len(original_words)}")

    # Initialize results
    correctly_sung = []
    missed_words = []
    mispronounced_words = []
    extra_words = []

    # Use sequence matcher to find matches in correct order
    matcher = difflib.SequenceMatcher(None, sung_words, original_words)

    # Process match blocks (these are sequences that match in order)
    for block in matcher.get_matching_blocks():
        i, j, n = block
        if n > 0:  # If there's a match
            for k in range(n):
                correctly_sung.append(original_words[j + k])
                print(f"ORDERED MATCH: '{sung_words[i + k]}' matches '{original_words[j + k]}' at position {j + k}")

    # Find missed words (words in original that aren't in correctly_sung)
    for word in original_words:
        if word not in correctly_sung:
            missed_words.append(word)

    # Find potential mispronunciations among the missed words
    sung_used = [False] * len(sung_words)
    for i, sung_word in enumerate(sung_words):
        if sung_word in correctly_sung:
            sung_used[i] = True

    # Check remaining sung words for similarity to missed words
    for missed_word in missed_words[:]:  # Copy to allow modification while iterating
        best_match = None
        best_idx = -1
        best_similarity = 0

        for i, sung_word in enumerate(sung_words):
            if not sung_used[i]:
                similarity = difflib.SequenceMatcher(None, missed_word, sung_word).ratio()
                if similarity > 0.7 and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = sung_word
                    best_idx = i

        if best_match:
            print(f"SIMILAR WORD: '{best_match}' is {best_similarity:.2f} similar to '{missed_word}'")
            mispronounced_words.append({
                'sung': best_match,
                'original': missed_word,
                'similarity': best_similarity
            })
            sung_used[best_idx] = True

    # Find extra words (words sung but not in original or mispronounced)
    for i, word in enumerate(sung_words):
        if not sung_used[i]:
            extra_words.append(word)
            print(f"EXTRA WORD: '{word}'")

    # Calculate statistics
    total_original_words = len(original_words)
    correctly_sung_count = len(correctly_sung)
    mispronounced_count = len(mispronounced_words)

    # Accuracy: percentage of sung words that exactly match original words in the right order
    accuracy = 0
    if len(sung_words) > 0:
        accuracy = (correctly_sung_count / len(sung_words)) * 100

    # Coverage: percentage of original words that were sung correctly in order
    coverage = 0
    if total_original_words > 0:
        coverage = (correctly_sung_count / total_original_words) * 100

    # Create word status list for visualization
    word_statuses = []
    mispronounced_originals = [item['original'] for item in mispronounced_words]

    for word in original_words:
        if word in correctly_sung:
            word_statuses.append({'word': word, 'status': 'correct'})
        elif word in mispronounced_originals:
            word_statuses.append({'word': word, 'status': 'mispronounced'})
        else:
            word_statuses.append({'word': word, 'status': 'missed'})

    # Create results dictionary
    results = {
        'missed_words': missed_words,
        'mispronounced_words': mispronounced_words,
        'extra_words': extra_words,
        'correctly_sung': correctly_sung,
        'accuracy_score': accuracy,
        'coverage_percentage': coverage,
        'word_statuses': word_statuses
    }

    print("\n==== COMPARISON RESULTS ====")
    print(f"Correctly sung: {correctly_sung_count} words")
    print(f"Mispronounced: {mispronounced_count} words")
    print(f"Missed words: {len(missed_words)} words")
    print(f"Extra words: {len(extra_words)} words")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Coverage: {coverage:.1f}%")

    return results


# This function checks for common phrases and returns templated responses
def get_templated_response(question, title=""):
    """
    Returns a templated response for common phrases without calling the AI API.
    Returns None if no template matches, indicating the AI should be called.
    """
    question = question.lower().strip()

    # Dictionary of common messages grouped by type
    templates = {
        # Greetings
        "greetings": {
            "matches": ["hi", "hello", "hey", "greetings", "howdy", "hi there", "hello there", "sup", "yo"],
            "responses": [
                f"Hello! I'm your lyrics coach for this song. How can I help you learn it better?",
                f"Hi there! Ready to help you master these lyrics. What specific part are you struggling with?",
                f"Hey! I'm here to help you memorize these lyrics. What would you like to know?"
            ]
        },

        # Thank you / Appreciation
        "thanks": {
            "matches": ["thank you", "thanks", "thank you so much", "thanks a lot", "appreciate it", "thx", "ty"],
            "responses": [
                "You're welcome! Feel free to ask if you need more help with these lyrics.",
                "Happy to help! Keep practicing and you'll memorize these lyrics in no time.",
                "Glad I could help! Is there anything else about the lyrics you'd like to know?"
            ]
        },

        # Farewells
        "farewell": {
            "matches": ["bye", "goodbye", "see you", "farewell", "see ya", "cya", "gotta go"],
            "responses": [
                "Goodbye! Good luck with your singing practice!",
                "See you later! Keep practicing those lyrics!",
                "Take care! Remember, regular practice makes perfect with lyrics memorization."
            ]
        },

        # Help requests
        "help": {
            "matches": ["help", "help me", "i need help", "can you help", "how do i", "how can i"],
            "responses": [
                f"I'd be happy to help! You can ask me about specific techniques for memorizing lyrics, understanding difficult phrases, or breaking down complex sections.",
                f"Sure thing! Try asking me about memory techniques, pronunciation tips, or ways to understand the meaning behind these lyrics.",
                f"I'm here to help! Ask me about memorization strategies, practice routines, or any challenging parts of the lyrics."
            ]
        },

        # About the assistant
        "about": {
            "matches": ["who are you", "what are you", "what do you do", "how do you work"],
            "responses": [
                "I'm your lyrics coach assistant! I can help you learn lyrics more effectively by providing memorization techniques, explaining meanings, and offering practice strategies.",
                "I'm an AI designed to help you memorize and understand lyrics better. I can provide tips, explain meanings, and suggest memory techniques specific to the song you're learning.",
                "I'm your personal lyrics coach! I can help with memorization techniques, line-by-line explanations, and practice strategies tailored to these lyrics."
            ]
        }
    }

    # Check if question matches any template
    for category, data in templates.items():
        for match in data["matches"]:
            if question == match or question.startswith(match + " ") or question.endswith(" " + match):
                return random.choice(data["responses"])

    # No match found, should use AI
    return None


# Helper function to list available Gemini models - useful for debugging
def list_available_models():
    """Lists all available Gemini models - helpful for debugging"""
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        models = genai.list_models()
        available_models = [model.name for model in models]
        print("Available Gemini models:", available_models)
        return available_models
    except Exception as e:
        print(f"Error listing models: {e}")
        return []


# The complete lyrics assistant function
@app.route('/api/lyrics-assistant', methods=['POST'])
def lyrics_assistant():
    """Handle requests to the lyrics AI assistant"""
    try:
        data = request.json
        question = data.get('question', '')
        lyrics = data.get('lyrics', '')
        title = data.get('title', '')
        artist = data.get('artist', '')

        print(f"Lyrics assistant received question: {question}")

        # Check for templated responses first
        templated_response = get_templated_response(question, title)
        if templated_response:
            print("Using templated response")
            return jsonify({"response": templated_response})

        # If not using API, return sample responses
        if not USE_AI_API:
            responses = [
                f"To learn these lyrics for '{title}', try breaking them into chunks of 2-3 lines and mastering each section before moving to the next.",
                "Look for patterns and rhymes in these lyrics - they can serve as memory hooks.",
                "Try singing along with just the first word of each line to build momentum.",
                f"For these lyrics, create a visual story in your mind to help remember the sequence.",
                "Connect the emotions in these lyrics to your own experiences to make them more memorable.",
                "Try writing out these lyrics by hand a few times - this engages different memory systems.",
                "Record yourself singing these lyrics and listen to it several times.",
                "Focus on the parts that tell a story - narrative elements are easier to remember.",
                "Try explaining what these lyrics mean to you in your own words."
            ]
            return jsonify({"response": random.choice(responses)})

        # Using Google Gemini API
        try:
            # Configure the Gemini API
            genai.configure(api_key=GEMINI_API_KEY)

            # Create the model with the updated model name
            model = genai.GenerativeModel('gemini-1.5-pro')

            # Handle very long lyrics - truncate if needed to avoid token limit errors
            if len(lyrics) > 5000:
                lyrics = lyrics[:5000] + "...[lyrics truncated for length]"

            # Create the prompt for the AI
            prompt = f"""You are a lyrics coach helping someone learn the lyrics to '{title}' by {artist}'.

LYRICS:
{lyrics}

USER QUESTION: {question}

Give a helpful, concise answer to help them learn these lyrics better. 
Include memorization techniques when relevant. Keep your response under 150 words.
            """

            # Call the Gemini API
            response = model.generate_content(prompt)

            # Extract the response text
            ai_response = response.text

            return jsonify({"response": ai_response})

        except Exception as e:
            print(f"Gemini API error: {e}")
            # Fallback to sample responses if API fails
            fallback_responses = [
                "I'm having trouble connecting to the AI service. Try breaking the lyrics into small chunks to memorize them more easily.",
                "Sorry, I couldn't process your question right now. Try creating a mental story from these lyrics to remember them better.",
                "There seems to be a connection issue. Focus on the rhyming patterns in these lyrics as memory hooks."
            ]
            return jsonify({"response": f"{random.choice(fallback_responses)} (Error: {str(e)})"})

    except Exception as e:
        print(f"Error in lyrics assistant: {e}")
        return jsonify({"response": "Sorry, I encountered an error. Please try again."})



def transcribe_audio_data(audio_data, sample_rate=16000):
    """Transcribe audio data to text with improved processing"""
    if wav2vec2_model is None or wav2vec2_processor is None:
        logger.error("Speech recognition model not available")
        return "Speech recognition model not available"

    try:
        logger.info(f"Starting transcription: audio shape={audio_data.shape}, sample_rate={sample_rate}")

        # Normalize audio (important for consistent recognition)
        if np.max(np.abs(audio_data)) > 0:  # Check if there is any sound
            audio_data = audio_data / np.max(np.abs(audio_data))
            logger.info("Audio normalized")
        else:
            logger.warning("Audio contains no data (silence)")
            return "No sound detected in the recording"

        # Resample if needed
        if sample_rate != 16000:
            logger.info(f"Resampling from {sample_rate}Hz to 16000Hz")
            # Convert to tensor for torchaudio resampling
            waveform = torch.FloatTensor(audio_data)
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)  # Add channel dimension if needed

            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            audio_data = waveform.squeeze().numpy()
            sample_rate = 16000
            logger.info(f"Resampled audio shape: {audio_data.shape}")

        # Apply some basic noise reduction (simple high-pass filter)
        try:
            from scipy import signal
            # Simple high-pass filter to reduce background noise (cutoff at 100Hz)
            b, a = signal.butter(4, 100 / (sample_rate / 2), 'highpass')
            audio_data = signal.filtfilt(b, a, audio_data)
            logger.info("Applied noise reduction filter")
        except ImportError:
            logger.warning("scipy not available, skipping noise reduction")

        # Process audio with Wav2Vec2
        logger.info("Processing audio with Wav2Vec2")
        input_values = wav2vec2_processor(
            audio_data,
            return_tensors="pt",
            sampling_rate=sample_rate,
            padding=True
        )["input_values"].to(device)

        # Get logits with error handling
        try:
            with torch.no_grad():
                logits = wav2vec2_model(input_values)["logits"]

            # Get predictions
            predicted_ids = torch.argmax(logits, dim=-1)

            # Decode to text
            transcription = wav2vec2_processor.batch_decode(predicted_ids)[0]
            logger.info(f"Raw transcription: {transcription}")

            # Clean up transcription
            transcription = transcription.lower()

            # If transcription is empty, it could mean no speech was detected
            if not transcription.strip():
                logger.warning("Empty transcription - no speech detected")
                return "No words detected in the recording. Please try again and sing louder."

            return transcription

        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                logger.error("CUDA out of memory - trying with CPU")
                # Try again with CPU
                with torch.no_grad():
                    input_values = input_values.cpu()
                    wav2vec2_model.cpu()
                    logits = wav2vec2_model(input_values)["logits"]
                    wav2vec2_model.to(device)

                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = wav2vec2_processor.batch_decode(predicted_ids)[0]
                return transcription.lower()
            else:
                raise e

    except Exception as e:
        logger.error(f"Error in transcription: {e}")
        logger.error(traceback.format_exc())
        return f"Error transcribing audio: {str(e)}"


@app.route('/api/evaluate', methods=['POST'])
def evaluate_singing():
    """Process recorded audio and compare with original lyrics"""
    if wav2vec2_model is None or wav2vec2_processor is None:
        return jsonify({'error': 'Speech recognition model not available'}), 500

    try:
        # Get data from the request
        data = request.json
        if not data or 'audio' not in data or 'song_id' not in data:
            return jsonify({'error': 'Missing audio data or song_id'}), 400

        # Log basic information
        logger.info(f"Processing evaluation for song_id: {data['song_id']}")

        # Decode base64 audio data
        try:
            # Remove header if present
            audio_base64 = data['audio'].split(',')[1] if ',' in data['audio'] else data['audio']
            audio_bytes = base64.b64decode(audio_base64)
            logger.info(f"Decoded audio: {len(audio_bytes)} bytes")
        except Exception as e:
            logger.error(f"Error decoding audio: {e}")
            return jsonify({'error': f'Audio data decoding failed: {str(e)}'}), 400

        # Save audio to a temporary file
        temp_file = None
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_file.write(audio_bytes)
            temp_file.close()
            temp_path = temp_file.name
            logger.info(f"Saved audio to temporary file: {temp_path}")
        except Exception as e:
            logger.error(f"Error saving temporary file: {e}")
            return jsonify({'error': f'Could not save audio data: {str(e)}'}), 500

        # Process the saved audio file
        try:
            # Try to create a proper WAV file
            output_path = temp_path + ".processed.wav"

            # Try using ffmpeg if available
            try:
                import subprocess
                logger.info("Converting audio using ffmpeg")
                subprocess.call([
                    'ffmpeg', '-i', temp_path,
                    '-acodec', 'pcm_s16le',
                    '-ar', '16000',
                    '-ac', '1',
                    output_path
                ], stderr=subprocess.DEVNULL)

                # Check if file was created
                if os.path.exists(output_path):
                    logger.info("Audio converted successfully with ffmpeg")
                    # Clean up original file
                    os.unlink(temp_path)
                    temp_path = output_path
                else:
                    logger.warning("ffmpeg did not produce output file")
            except Exception as e:
                logger.warning(f"ffmpeg conversion failed: {e}")

            # Load the audio file with torchaudio
            logger.info(f"Loading audio from: {temp_path}")
            waveform, sample_rate = torchaudio.load(temp_path)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                logger.info("Converted stereo to mono")

            # Resample if needed
            if sample_rate != 16000:
                logger.info(f"Resampling from {sample_rate}Hz to 16000Hz")
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                sample_rate = 16000

            # Normalize audio
            audio_data = waveform.squeeze().numpy()
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
                logger.info("Audio normalized")
            else:
                logger.warning("Audio contains no data (silence)")
                return jsonify({'error': 'No sound detected in the recording. Please try again.'}), 400

            # Clean up
            os.unlink(temp_path)

        except Exception as e:
            # Clean up temp file if it exists
            if temp_file and os.path.exists(temp_path):
                os.unlink(temp_path)

            logger.error(f"Error processing audio file: {e}")
            logger.error(traceback.format_exc())
            return jsonify({'error': f'Audio processing failed: {str(e)}'}), 500

        # Get the song lyrics
        song_id = data['song_id']
        logger.info(f"Getting lyrics for song_id: {song_id}")

        # Get the song lyrics text
        response = get_song_lyrics_text(song_id)
        if isinstance(response, tuple):
            logger.error("Failed to get song lyrics")
            return jsonify({'error': 'Failed to get song lyrics'}), 500

        song_data = json.loads(response.get_data(as_text=True))
        if 'error' in song_data:
            logger.error(f"Error in song data: {song_data['error']}")
            return jsonify({'error': f"Error getting song data: {song_data['error']}"}), 404

        original_lyrics = song_data['lyrics']
        logger.info(f"Retrieved lyrics with {len(original_lyrics.split())} words")

        # Transcribe audio
        logger.info("Starting transcription...")
        transcribed_text = transcribe_audio_data(audio_data, sample_rate)
        logger.info(f"Transcribed text: {transcribed_text}")

        # Compare with original lyrics
        logger.info("Comparing lyrics...")
        comparison_results = compare_lyrics(transcribed_text, original_lyrics)
        logger.info(f"Comparison results: {len(comparison_results['correctly_sung'])} words correct, " +
                    f"{len(comparison_results['missed_words'])} words missed, " +
                    f"{len(comparison_results['mispronounced_words'])} words mispronounced")

        # Prepare report

        report = {
            'title': f"Singing Evaluation: {song_data['title']}",
            'accuracy': round(comparison_results['accuracy_score'], 1),
            'coverage': round(comparison_results['coverage_percentage'], 1),
            'missed_words': comparison_results['missed_words'],
            'mispronounced_words': comparison_results['mispronounced_words'],
            'extra_words': comparison_results['extra_words'],
            'words_correct': len(comparison_results['correctly_sung']),
            'words_missed': len(comparison_results['missed_words']),
            'words_mispronounced': len(comparison_results['mispronounced_words']),
            'words_extra': len(comparison_results['extra_words']),
            'transcription': transcribed_text,
            'word_statuses': comparison_results['word_statuses']  # Add this line
        }
        # Performance rating
        if comparison_results['coverage_percentage'] >= 90:
            report['rating'] = "Excellent! 🌟"
        elif comparison_results['coverage_percentage'] >= 75:
            report['rating'] = "Very Good! 👏"
        elif comparison_results['coverage_percentage'] >= 60:
            report['rating'] = "Good effort 👍"
        elif comparison_results['coverage_percentage'] >= 40:
            report['rating'] = "Keep practicing 🎤"
        else:
            report['rating'] = "More practice needed 📝"

        logger.info(f"Evaluation complete - accuracy: {report['accuracy']}%, coverage: {report['coverage']}%")
        return jsonify(report)

    except Exception as e:
        logger.error(f"Error evaluating singing: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Check if audio files exist
    audio_files = []
    if os.path.exists(AUDIO_DIR):
        for root, dirs, files in os.walk(AUDIO_DIR):
            for file in files:
                if file.endswith(('.wav', '.mp3')):
                    audio_files.append(os.path.join(root, file))

    logger.info(f"Found {len(audio_files)} audio files")

    # Create directories if they don't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)

    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)