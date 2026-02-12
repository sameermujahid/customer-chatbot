import os
import webrtcvad
import collections
import speech_recognition as sr
from pydub import AudioSegment
from happytransformer import HappyTextToText, TTSettings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VADAudio:
    def __init__(self, aggressiveness=3):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = 16000
        self.frame_duration_ms = 30

    def frame_generator(self, audio, frame_duration_ms, sample_rate):
        n = int(sample_rate * (frame_duration_ms / 1000.0))
        offset = 0
        while offset + n < len(audio):
            yield audio[offset:offset + n]
            offset += n

    def vad_collector(self, audio, sample_rate, frame_duration_ms, padding_duration_ms=300, aggressiveness=3):
        vad = webrtcvad.Vad(aggressiveness)
        num_padding_frames = int(padding_duration_ms / frame_duration_ms)
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False

        for frame in self.frame_generator(audio, frame_duration_ms, sample_rate):
            is_speech = vad.is_speech(frame, sample_rate)
            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > 0.9 * ring_buffer.maxlen:
                    triggered = True
                    for f, s in ring_buffer:
                        yield f
                    ring_buffer.clear()
            else:
                yield frame
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > 0.9 * ring_buffer.maxlen:
                    triggered = False
                    yield b''.join([f for f in ring_buffer])
                    ring_buffer.clear()

def transcribe_with_vad(audio_file):
    try:
        vad_audio = VADAudio()
        audio = AudioSegment.from_file(audio_file)
        audio = audio.set_frame_rate(vad_audio.sample_rate).set_channels(1)
        raw_audio = audio.raw_data

        frames = vad_audio.vad_collector(raw_audio, vad_audio.sample_rate, vad_audio.frame_duration_ms)
        for frame in frames:
            if len(frame) > 0:
                recognizer = sr.Recognizer()
                recognizer.energy_threshold = 300
                recognizer.dynamic_energy_threshold = True
                recognizer.pause_threshold = 0.8
                
                audio_data = sr.AudioData(frame, vad_audio.sample_rate, audio.sample_width)
                try:
                    text = recognizer.recognize_google(audio_data, language="en-US")
                    if text.strip():
                        print(f"Transcription: {text}")
                        return text
                except sr.UnknownValueError:
                    logger.warning("Google Speech Recognition could not understand the audio")
                except sr.RequestError as e:
                    logger.error(f"Could not request results from Google Speech Recognition service; {e}")
        return ""
    except Exception as e:
        logger.error(f"Error in transcribe_with_vad: {str(e)}")
        return ""

def process_audio_file(audio_file):
    if 'audio' not in audio_file:
        return {"error": "No audio file provided"}, 400

    # Ensure the file has an allowed extension
    allowed_extensions = {'wav', 'mp3', 'ogg', 'webm'}
    if '.' not in audio_file.filename or \
       audio_file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return {"error": "Invalid audio file format"}, 400

    temp_path = None
    wav_path = None
    try:
        # Save the uploaded file temporarily
        temp_dir = os.path.join(os.getcwd(), 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, 'temp_audio.' + audio_file.filename.rsplit('.', 1)[1].lower())

        audio_file.save(temp_path)

        # Convert audio to proper format if needed
        audio = AudioSegment.from_file(temp_path)
        audio = audio.set_channels(1)  # Convert to mono
        audio = audio.set_frame_rate(16000)  # Set sample rate to 16kHz
        audio = audio.normalize()  # Normalize audio levels

        # Save as WAV for speech recognition
        wav_path = os.path.join(temp_dir, 'temp_audio.wav')
        audio.export(wav_path, format="wav", parameters=["-ar", "16000", "-ac", "1"])

        # Try VAD-based transcription first
        text = transcribe_with_vad(wav_path)
        
        # If VAD fails, try direct recognition
        if not text:
            recognizer = sr.Recognizer()
            recognizer.energy_threshold = 300
            recognizer.dynamic_energy_threshold = True
            recognizer.pause_threshold = 0.8

            with sr.AudioFile(wav_path) as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data, language="en-US")

        if not text.strip():
            return {"error": "No speech detected. Please try speaking again."}, 400

        # Grammar correction
        try:
            happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
            settings = TTSettings(do_sample=True, top_k=50, temperature=0.7)
            corrected_text = happy_tt.generate_text(f"grammar: {text}", args=settings)
            corrected_text = corrected_text.text
        except Exception as e:
            logger.error(f"Grammar correction failed: {str(e)}")
            corrected_text = text  # Fall back to original text

        print(f"Original Transcription: {text}")
        print(f"Corrected Transcription: {corrected_text}")

        return {
            "transcription": corrected_text,
            "original": text
        }

    except sr.UnknownValueError:
        return {"error": "Could not understand audio. Please speak more clearly."}, 400
    except sr.RequestError as e:
        return {"error": f"Speech recognition service error: {str(e)}"}, 500
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        return {"error": f"Audio processing error: {str(e)}"}, 500
    finally:
        # Ensure temp files are cleaned up even if an error occurs
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.error(f"Error removing temp file: {str(e)}")
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception as e:
                logger.error(f"Error removing wav file: {str(e)}") 