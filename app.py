import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_avg_spectrum(video_path, num_frames=10, resize=(256,256)):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, total-1, num_frames, dtype=int)

    spectra = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, resize)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1)
        spectra.append(magnitude)
    cap.release()
    return np.mean(spectra, axis=0)

def radial_profile(data):
    y, x = np.indices((data.shape))
    center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])
    r = np.hypot(x - center[0], y - center[1])
    r = r.astype(np.int32)
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    return tbin / np.maximum(nr, 1)

# --- STEP 1: Load reference videos ---
real_ref = "real (15).mp4"     # a real video sample
ai_ref   = "A_closeup_handheld_202508232113.mp4"       # an AI-generated sample

real_profile = radial_profile(get_avg_spectrum(real_ref))
ai_profile   = radial_profile(get_avg_spectrum(ai_ref))
real_profile /= np.max(real_profile)
ai_profile   /= np.max(ai_profile)

# --- STEP 2: Analyze new video ---
test_video = "A_4k_ultrarealistic_202508232204.mp4"
test_profile = radial_profile(get_avg_spectrum(test_video))
test_profile /= np.max(test_profile)

# --- STEP 3: Compare to references ---
diff_real = np.mean(np.abs(test_profile - real_profile))
diff_ai   = np.mean(np.abs(test_profile - ai_profile))

if diff_real < diff_ai:
    label = "Real Video"
    reason = ("Its frequency spectrum resembles real footage — "
              "broader spread of mid/high-frequency energy and sensor-like noise.")
else:
    label = "AI-Generated Video"
    reason = ("Its spectrum is smoother and more low-frequency dominated, "
              "lacking natural sensor noise typical of real cameras.")

# --- STEP 4: Visualization ---
plt.figure(figsize=(8,4))
plt.plot(test_profile, label='Test Video', linewidth=2)
plt.plot(real_profile, label='Real Ref', linestyle='--')
plt.plot(ai_profile, label='AI Ref', linestyle='--')
plt.title(f"Frequency Energy Profile – Prediction: {label}")
plt.xlabel("Spatial Frequency Radius →")
plt.ylabel("Normalized Energy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"\nPredicted label: {label}")
print(f"Reason: {reason}")
print(f"Difference vs Real: {diff_real:.4f}")
print(f"Difference vs AI: {diff_ai:.4f}")
