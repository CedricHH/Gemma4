import base64
import io
import gradio as gr
from scipy.io import wavfile
from openai import OpenAI

API_BASE_URL = "http://localhost:8000/v1"
API_KEY = "local-server-key"
MODEL_ID = "google/gemma-4-E2B-it"

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

system_instruction = (
    "Transcribe the following speech in German into German text. "
    "Follow these specific instructions for formatting the answer: "
    "* Only output the transcription, with no newlines. "
    "* When transcribing numbers, write the digits (e.g., 1.7 instead of one point seven)."
)

def process_audio(audio_data):
    if audio_data is None:
        return "Bitte zuerst eine Audioaufnahme machen."

    # Gradio passes audio as a tuple: (sample_rate, data)
    sample_rate, audio_numpy = audio_data

    try:
        # Convert audio to wav format in memory
        wav_buffer = io.BytesIO()
        wavfile.write(wav_buffer, sample_rate, audio_numpy)
        binary_wav_data = wav_buffer.getvalue()

        # Encode to base64
        base64_encoded_string = base64.b64encode(binary_wav_data).decode('utf-8')
        data_uri = f"data:audio/wav;base64,{base64_encoded_string}"

        # Structure payload
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {
                            "url": data_uri
                        }
                    },
                    {
                        "type": "text",
                        "text": system_instruction
                    }
                ]
            }
        ]

        # Call API
        completion = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_tokens=512,
            temperature=0.1,
        )

        return completion.choices.message.content

    except Exception as e:
        return f"Fehler bei der Verarbeitung: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Gemma 4 E2B-it Audio Transkription") as demo:
    gr.Markdown("# Gemma 4 E2B-it Audio Transkription")
    gr.Markdown("Nimm Audio mit deinem Mikrofon auf, um es vom lokalen vLLM-Server transkribieren zu lassen.")

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["microphone"], type="numpy", label="Audio Aufnahme")
            submit_btn = gr.Button("Transkribieren", variant="primary")

        with gr.Column():
            text_output = gr.Textbox(label="Transkriptions-Ergebnis", lines=10)

    submit_btn.click(fn=process_audio, inputs=audio_input, outputs=text_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
