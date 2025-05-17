import gradio as gr
import whisper

# Load the Whisper model
model = whisper.load_model("base")

# Define the function to process the audio input
def transcribe_audio(audio_file_path):
    if audio_file_path is None:
        return "No audio input provided."
    # Transcribe the audio
    result = model.transcribe(audio_file_path)
    return result["text"]

# Create a Gradio interface
interface = gr.Interface(
    fn=transcribe_audio,  # Function that will process the input
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath"),  # Input type: microphone or file upload
    outputs="text",  # Output: Transcribed text
    title="Whisper Speech-to-Text",  # Interface title
    description="🎤 Speak or upload audio to convert speech to text with Whisper.",  # Description text
    live=True  # Enable real-time interaction
)

# Launch the Gradio interface with a public URL
interface.launch(share=True)