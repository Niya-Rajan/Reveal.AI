import os
import subprocess
import tempfile
import numpy as np
import torch
import torch.nn as nn
import librosa

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# ================== FASTAPI ==================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================== MODEL ==================
class MFCC_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = MFCC_CNN().to(DEVICE)
model.load_state_dict(
    torch.load("mfcc_audio_detector.pth", map_location=DEVICE)
)
model.eval()

# ================== AUDIO UTILS ==================
def extract_audio_from_video(video_path, out_wav):
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ac", "1",
        "-ar", "16000",
        out_wav
    ]
    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return out_wav


def extract_mfcc(audio_path, sr=16000, n_mfcc=40, max_len=300):
    audio, _ = librosa.load(audio_path, sr=sr)

    if len(audio) < sr:
        raise ValueError("Audio too short")

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc
    )

    # Pad / truncate (EXACT AS TRAINING)
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(
            mfcc,
            ((0, 0), (0, max_len - mfcc.shape[1]))
        )
    else:
        mfcc = mfcc[:, :max_len]

    # Normalize (EXACT AS TRAINING)
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)

    return mfcc


# ================== API ==================
@app.post("/predict")
async def predict_video(file: UploadFile = File(...)):
    # Save uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await file.read())
        video_path = temp_video.name

    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

    try:
        # 1. Extract audio
        extract_audio_from_video(video_path, temp_audio)

        # 2. Extract MFCC
        mfcc = extract_mfcc(temp_audio)

        # 3. Prepare tensor (1, 1, 40, 300)
        x = (
            torch.tensor(mfcc)
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .to(DEVICE)
        )

        # 4. Predict
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)

        real_prob = float(probs[0][0])
        ai_prob = float(probs[0][1])

        prediction = "AI-GENERATED" if ai_prob > real_prob else "REAL"
        confidence = round(max(real_prob, ai_prob) * 100, 2)

        return {
            "prediction": prediction,
            "confidence": confidence,
            "real_probability": round(real_prob, 4),
            "ai_probability": round(ai_prob, 4)
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
