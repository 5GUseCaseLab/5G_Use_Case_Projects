import os
import io
import re
import sys
import time
import numpy as np
import speech_recognition as sr
try:
    from .campus_snapping import snap_to_campus_landmark
except ImportError:
    from campus_snapping import snap_to_campus_landmark

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Lazy-loaded Multilingual Whisper model instance (cached in RAM)
_whisper_model = None

def get_whisper_model():
    """
    Loads or returns the cached local Multilingual Whisper model.
    Uses 'small' (244M parameters) with INT8 quantization:
    High acoustic resolution for distinguishing subtle phonemes and acronyms.
    """
    global _whisper_model
    if _whisper_model is None:
        try:
            from faster_whisper import WhisperModel
            print("⏳ Loading local High-Precision Whisper AI model ('small')...")
            # compute_type='int8' runs fast on CPU with 3x higher parameter precision
            _whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
            print("✅ Multilingual Whisper AI ('small') ready for high-accuracy recognition!")
        except ImportError:
            _whisper_model = False
            print("⚠️ 'faster-whisper' not installed. Falling back to Google Web Speech API.")
        except Exception as e:
            _whisper_model = False
            print(f"⚠️ Error initializing Whisper: {e}. Falling back to Google Web Speech API.")
    return _whisper_model

def clean_destination_query(raw_text):
    """
    Intelligently extracts the core destination from natural spoken commands.
    e.g. 'take me to Brahmaputra' -> 'Brahmaputra'
         'where is lecture hall complex' -> 'lecture hall complex'
         'go to Core 2' -> 'Core 2'
    Zero hardcoded locations: dynamically parses natural language grammar.
    """
    if not raw_text:
        return ""
    cleaned = raw_text.strip()
    prefixes = [
        r"^(please\s+)?(navigate\s+me\s+(to\s+)?)",
        r"^(please\s+)?(take\s+me\s+to\s+)",
        r"^(please\s+)?(can\s+you\s+take\s+me\s+to\s+)",
        r"^(please\s+)?(go\s+to\s+)",
        r"^(please\s+)?(navigate\s+to\s+)",
        r"^(please\s+)?(guide\s+me\s+to\s+)",
        r"^(please\s+)?(where\s+is\s+)",
        r"^(i\s+want\s+to\s+go\s+to\s+)",
        r"^(head\s+to\s+)",
        r"^(lead\s+me\s+to\s+)",
        r"^(walk\s+to\s+)",
    ]
    for p in prefixes:
        cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned.strip(". ,!?:;\"'")

def normalize_audio(audio_data):
    """
    Converts speech recognition audio to a normalized 16kHz float32 NumPy array.
    Applies automatic digital gain (pre-amp) so quiet Bluetooth earphone audio
    is brought to optimal volume without clipping or distortion.
    """
    raw_bytes = audio_data.get_raw_data(convert_rate=16000, convert_width=2)
    audio_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    
    # Calculate peak volume
    peak = np.max(np.abs(audio_np))
    if peak > 0.01:
        target_peak = 0.85
        # Apply up to 4.0x digital gain boost (12 dB) to quiet earphone audio
        gain = min(target_peak / peak, 4.0)
        audio_np = np.clip(audio_np * gain, -1.0, 1.0)
    
    return audio_np

def decode_audio_to_np(raw_bytes, sample_rate=16000, fmt=None):
    """
    Decodes binary audio payload (WAV, MP4, WebM, AAC, PyAV container, or raw PCM) into a
    normalized 16kHz mono float32 NumPy array ready for Whisper AI.
    """
    if not raw_bytes or len(raw_bytes) == 0:
        return None
        
    audio_np = None
    
    # 1. PyAV high-fidelity decoder & resampler (FFmpeg sinc filter with zero aliasing)
    try:
        import av
        container = None
        try:
            container = av.open(io.BytesIO(raw_bytes))
        except Exception as open_err:
            if fmt:
                try:
                    container = av.open(io.BytesIO(raw_bytes), format=fmt)
                except Exception:
                    pass
            if container is None:
                raise open_err

        if container and container.streams.audio:
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(format='fltp', layout='mono', rate=16000)
            frames = []
            for frame in container.decode(stream):
                for r in resampler.resample(frame):
                    frames.append(r.to_ndarray())
            for r in resampler.resample(None):
                frames.append(r.to_ndarray())
            if frames:
                audio_np = np.concatenate(frames, axis=1).squeeze().astype(np.float32)
    except Exception:
        pass

    # 2. Standard wave module fallback
    if audio_np is None:
        try:
            import wave
            with wave.open(io.BytesIO(raw_bytes), "rb") as wf:
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                n_channels = wf.getnchannels()
                pcm_data = wf.readframes(n_frames)
            if n_channels > 1:
                audio_np = np.frombuffer(pcm_data, dtype=np.int16).reshape(-1, n_channels).mean(axis=1).astype(np.float32) / 32768.0
            else:
                audio_np = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
            if framerate != 16000 and len(audio_np) > 0:
                target_len = int(len(audio_np) * 16000 / framerate)
                audio_np = np.interp(
                    np.linspace(0, len(audio_np), target_len, endpoint=False),
                    np.arange(len(audio_np)),
                    audio_np
                ).astype(np.float32)
        except Exception:
            pass

    # 3. Raw 16-bit PCM fallback
    if audio_np is None and len(raw_bytes) > 200 and len(raw_bytes) % 2 == 0:
        try:
            pcm_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if len(pcm_np) > 1600:
                if sample_rate != 16000:
                    target_len = int(len(pcm_np) * 16000 / sample_rate)
                    audio_np = np.interp(
                        np.linspace(0, len(pcm_np), target_len, endpoint=False),
                        np.arange(len(pcm_np)),
                        pcm_np
                    ).astype(np.float32)
                else:
                    audio_np = pcm_np
        except Exception:
            pass

    if audio_np is not None and len(audio_np) > 0:
        peak = np.max(np.abs(audio_np))
        if peak > 0.01:
            gain = min(0.85 / peak, 4.0)
            audio_np = np.clip(audio_np * gain, -1.0, 1.0)
            
    return audio_np

def transcribe_audio_np(audio_np):
    """
    Directly transcribes a 16kHz float32 NumPy array using local Whisper AI.
    Applies campus context prompt, cleaning, and fuzzy landmark snapping.
    """
    whisper = get_whisper_model()
    if whisper:
        campus_context_prompt = (
            "Campus navigation at IIT Guwahati: New SAC, Old SAC, Student Activity Centre, "
            "Central Library, Lecture Hall Complex, LHC, Academic Complex, Core 1, Core 2, "
            "Core 3, Core 4, Brahmaputra, Dihing, Kameng, Subansiri, Manas, Umiam, Barak, "
            "Kapili, Siang, Lohit, Dibang, Dhansiri, Disang, Gaurang, Married Scholars Hostel, "
            "Food Court, Khokha Market, Sports Complex, Auditorium, Health Centre, Hospital, Main Gate."
        )
        try:
            segments, info = whisper.transcribe(
                audio_np, 
                language="en", 
                task="transcribe", 
                beam_size=3, 
                vad_filter=True,
                initial_prompt=campus_context_prompt
            )
            raw_text = " ".join([segment.text for segment in segments]).strip()
            
            # If vad_filter=True filtered out faint or single-word speech, retry without vad_filter
            if not raw_text:
                segments_fallback, _ = whisper.transcribe(
                    audio_np,
                    language="en",
                    task="transcribe",
                    beam_size=3,
                    vad_filter=False,
                    initial_prompt=campus_context_prompt
                )
                raw_text = " ".join([s.text for s in segments_fallback]).strip()

            cleaned_dest = clean_destination_query(raw_text)
            
            # Apply Intelligent Campus Landmark Fuzzy Snapping
            snapped_dest, conf, was_corrected = snap_to_campus_landmark(cleaned_dest)
            if was_corrected:
                print(f"⚡ [CAMPUS SNAPPED]: '{cleaned_dest}' -> '{snapped_dest}' ({conf}% match)")
                return snapped_dest
            elif cleaned_dest:
                print(f"✅ Recognized Destination: '{cleaned_dest}' (raw: '{raw_text}')")
                return cleaned_dest
            else:
                print("❌ Whisper detected only silence/noise.")
                return None
        except Exception as e:
            print(f"⚠️ Whisper transcription error: {e}")
            return None
    return None

def get_voice_command():
    """
    Activates the microphone, listens for a user command via Bluetooth or wired earphones, 
    normalizes the audio volume, and converts speech to text using offline Whisper AI.
    """
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 1.0
    recognizer.phrase_threshold = 0.2
    recognizer.non_speaking_duration = 0.4

    with sr.Microphone() as source:
        try:
            recognizer.adjust_for_ambient_noise(source, duration=0.4)
        except Exception:
            pass
        # The microphone hardware stream is now fully open and recording!
        print("\n🟢 Microphone LIVE! Speak your destination through your earphones...", flush=True)
        
        # Quick non-blocking chime right as mic goes live
        try:
            import winsound
            winsound.Beep(950, 100)
        except Exception:
            pass

        try:
            # Immediate capture with zero calibration delay
            audio = recognizer.listen(source, timeout=6, phrase_time_limit=10)
            
            # --- TEMPORARY DEBUG: Save captured voice to test audio quality ---
            debug_wav_path = "captured_voice_debug.wav"
            try:
                with open(debug_wav_path, "wb") as f:
                    f.write(audio.get_wav_data())
                print(f"💾 [DEBUG] Saved captured voice to: {os.path.abspath(debug_wav_path)}")
            except Exception:
                pass
            # ------------------------------------------------------------------
            
            # Pre-process: Normalize audio volume for consistent Bluetooth reception
            audio_np = normalize_audio(audio)
            
            # Attempt transcription with offline Multilingual Whisper AI
            whisper_result = transcribe_audio_np(audio_np)
            if whisper_result:
                return whisper_result

            # Fallback to Google Web Speech if Whisper returned silence/noise
            try:
                print("🔄 Processing speech with Google Speech API (Fallback)...")
                norm_pcm = (np.clip(audio_np, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                norm_audio = sr.AudioData(norm_pcm, 16000, 2)
                raw_text = recognizer.recognize_google(norm_audio)
                cleaned_dest = clean_destination_query(raw_text)
                snapped_dest, conf, was_corrected = snap_to_campus_landmark(cleaned_dest)
                if was_corrected:
                    print(f"⚡ [CAMPUS SNAPPED]: '{cleaned_dest}' -> '{snapped_dest}' ({conf}% match)")
                    return snapped_dest
                if cleaned_dest:
                    print(f"✅ Recognized Destination (Fallback): '{cleaned_dest}'")
                    return cleaned_dest
            except Exception as fb_err:
                print(f"⚠️ Google Speech fallback notice: {fb_err}")
            
        except sr.WaitTimeoutError:
            print("❌ Timeout: No voice sound detected. Please speak at normal volume through your earphones.")
            return None
        except sr.UnknownValueError:
            print("❌ Sound heard, but speech was unclear. Please try speaking again.")
            return None
        except sr.RequestError as e:
            print(f"❌ Network Error: Could not connect to Google Speech service; {e}")
            return None

if __name__ == "__main__":
    # --- LOCAL TESTING BLOCK ---
    print("--- Testing Speech-to-Text Module ---")
    get_whisper_model()  # Preload Whisper AI into RAM so recognition is instant
    destination = get_voice_command()
    
    if destination:
        print(f"\n🎉 Test Successful! The OSRM routing module will now search for: [{destination}]")
    else:
        print("\n⚠️ Test Failed. Please try running it again.")
