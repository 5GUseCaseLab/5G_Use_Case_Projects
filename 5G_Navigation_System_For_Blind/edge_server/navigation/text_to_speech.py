import os
import threading
import subprocess

def _speak_worker(text):
    """
    Background worker that speaks text out loud using Windows SAPI Speech engine.
    Fast, completely offline, zero-latency, and doesn't conflict with audio locks.
    """
    try:
        # Sanitize text for Windows PowerShell SAPI voice command
        clean_text = text.replace("'", "").replace('"', "")
        cmd = f"powershell -Command \"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{clean_text}')\""
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"⚠️ Speech Error: {e}")

def speak_text(text, async_mode=True):
    """
    Speaks the given text into the computer speakers / earphones.
    If async_mode is True, speaks in a background thread so it never freezes AI vision.
    """
    print(f"🔊 [AUDIO]: \"{text}\"")
    if async_mode:
        t = threading.Thread(target=_speak_worker, args=(text,), daemon=True)
        t.start()
    else:
        _speak_worker(text)

if __name__ == "__main__":
    print("Testing Text-to-Speech Engine...")
    speak_text("Five G Navigation system active. Path is clear.", async_mode=False)
