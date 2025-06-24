from fastapi import FastAPI, UploadFile, File
import whisper
import tempfile

app = FastAPI()
model = whisper.load_model("base")

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_audio.write(await file.read())
        temp_audio.flush()
        result = model.transcribe(temp_audio.name)
    return {"transcription": result["text"]}


