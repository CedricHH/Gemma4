"""
Gemma 4 E2B-it Audio-Inferenz Client
====================================
Autorisiert für professionelle Deployments. Erfasst Audio, konvertiert die Tensoren
in Base64 Data-URIs und sendet eine ASR-Anfrage an die lokale vLLM Instanz.
"""

import sys
import io
import base64
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from openai import OpenAI

# -----------------------------------------------------------------------------
# Modellspezifische physikalische und architektonische Konstanten
# -----------------------------------------------------------------------------
SAMPLE_RATE = 16000  # Zwingende 16 kHz Spezifikation des Conformer-Encoders
CHANNELS = 1         # Reduktion auf Mono-Signalpfad
DTYPE = 'float32'    # Normalisierung im Bereich [-1.0, 1.0] für optimale FFT
RECORD_SECONDS = 10  # Testdauer (Limit von 30s strikt beachten)

API_BASE_URL = "http://localhost:8000/v1"
# API Key ist für vLLM lokal nicht verifikationspflichtig, aber syntaktisch gefordert.
API_KEY = "local-server-key"
MODEL_ID = "google/gemma-4-E2B-it"


def capture_audio_to_base64_uri(duration_sec: int, sample_rate: int, channels: int, dtype: str) -> str:
    """
    Kapselt die Interaktion mit PortAudio. Nimmt ein blockierendes Audiosignal auf,
    baut einen virtuellen RIFF/WAV-Header im RAM und kodiert die Binärdaten in Base64.
    """
    print(f"\n[MIC] Initialisiere Hardware-Aufnahme für {duration_sec} Sekunden...")
    print(f"[MIC] Format: {sample_rate} Hz, {channels} Kanal, {dtype}")
    print(f"[MIC] >>> Bitte sprechen Sie jetzt <<<")

    try:
        # sd.rec extrahiert direkt in den Arbeitsspeicher als NumPy Tensor
        audio_tensor = sd.rec(
            int(duration_sec * sample_rate),
            samplerate=sample_rate,
            channels=channels,
            dtype=dtype
        )
        sd.wait() # Synchrone Blockierung bis zur Fertigstellung
        print("[MIC] Aufnahme beendet. Initialisiere Vektor-Transformation...")

        # Transformation des flüchtigen Tensurs in eine persistente, formatierte Byte-Struktur
        wav_buffer = io.BytesIO()
        wavfile.write(wav_buffer, sample_rate, audio_tensor)

        # Extraktion und Base64 Codierung (33% Netzwerk-Overhead-Generierung)
        binary_wav_data = wav_buffer.getvalue()
        base64_encoded_string = base64.b64encode(binary_wav_data).decode('utf-8')

        # Konstruktion der Data-URI gemäß W3C und OpenAI API Standards
        data_uri = f"data:audio/wav;base64,{base64_encoded_string}"
        return data_uri

    except Exception as hardware_err:
        print(f" Hardware-Fehler während der Mikrofon-Kommunikation: {hardware_err}")
        sys.exit(1)


def transmit_to_gemma_api(base64_uri: str):
    """
    Konstruiert das OpenAI-kompatible JSON-Payload unter strikter Einhaltung
    der 'Modality Order' und übermittelt die Inferenzanfrage an vLLM.
    """
    print(f" Initialisiere TCP-Verbindung zu {API_BASE_URL}...")

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    # Rigides Prompt-Engineering zur Vermeidung von Generierungs-Halluzinationen
    system_instruction = (
        "Transcribe the following speech in German into German text. "
        "Follow these specific instructions for formatting the answer: "
        "* Only output the transcription, with no newlines. "
        "* When transcribing numbers, write the digits (e.g., 1.7 instead of one point seven)."
    )

    # Strukturierung der JSON Payload
    # WICHTIG: Das Audio-Objekt muss den Index 0 (erste Position) in der Content-Liste haben.
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": base64_uri
                    }
                },
                {
                    "type": "text",
                    "text": system_instruction
                }
            ]
        }
    ]

    try:
        print(" Payload übermittelt. Warte auf PagedAttention und Conformer-Inferenz...")

        # API Kommunikation. Hier könnte optional 'chat_template_kwargs'
        # via vLLM-spezifischen extra-Body-Parametern übergeben werden, um
        # den Thinking Mode (enable_thinking=False) zu deaktivieren.
        completion = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_tokens=512,
            temperature=0.1, # Deterministische Abtastung (Greedy Decoding)
        )

        model_response = completion.choices.message.content

        print("\n" + "="*70)
        print(" TRANSKRIPTIONS-ERGEBNIS (Gemma 4 E2B-it)")
        print("="*70)
        print(model_response)
        print("="*70 + "\n")

    except Exception as network_err:
        print(f" Netzwerk- oder Inferenzfehler: {network_err}")


if __name__ == "__main__":
    # Sequentielle Ausführung der Pipeline
    audio_data_uri = capture_audio_to_base64_uri(
        duration_sec=RECORD_SECONDS,
        sample_rate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE
    )

    transmit_to_gemma_api(audio_data_uri)
