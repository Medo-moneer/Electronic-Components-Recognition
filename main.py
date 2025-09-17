import os
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageTk
import tkinter as tk

# ---------------- Settings ----------------
MODEL_PATH = os.path.join("src", "model", "model.pth")
IMG_SIZE = 224
CLASS_NAMES = [
    "0","1","potentiometer","Ultrasonic","Please show me the pice","13","14","15","Resistor","17","18","19","Resistor","LED","Resistor",
    "22","23","Ultrasonic","25","26","27","28","29","Push botton","Push botton","31","32","Resistor","34","35",
    "potentiometer","37","38","Push botton","Arduino","40","Push botton","potentiometer","43","44","45","46","LED","48","49",
    "5","Resistor","51","potentiometer","Please show me the pice","Arduino","Ultrasonic","Resistor","57","58","59","6","LED","7","8","9"
]
NUM_CLASSES = len(CLASS_NAMES)

# ---------------- Load Model ----------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")

state_dict = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# ---------------- Image Transform ----------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------- GUI ----------------
root = tk.Tk()
root.title("AI Camera Classifier - Google Lens Style")
root.configure(bg="black")

# Camera Preview (small window)
camera_label = tk.Label(root, bg="black")
camera_label.pack(side="left", padx=10, pady=10)

# Prediction Text Area
result_label = tk.Label(root, text="Waiting for camera...", font=("Arial", 16), fg="white", bg="black", justify="left", anchor="nw")
result_label.pack(side="right", fill="both", expand=True, padx=10, pady=10)

# ---------------- Camera ----------------
cap = cv2.VideoCapture(0)

def update_frame():
    ret, frame = cap.read()
    if not ret:
        result_label.config(text="❌ Cannot access camera")
        return
    
    # Resize preview (Google Lens small style)
    frame_small = cv2.resize(frame, (300, 220))  
    img_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    # Convert to Tkinter Image
    imgtk = ImageTk.PhotoImage(image=img_pil)
    camera_label.imgtk = imgtk
    camera_label.configure(image=imgtk)

    # Prediction
    img_input = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(img_input).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    prediction_text = f"Prediction: {CLASS_NAMES[pred.item()]}\nConfidence: {conf.item()*100:.1f}%"
    result_label.config(text=prediction_text)

    # Repeat after 50 ms
    root.after(50, update_frame)

# Start updating
update_frame()

# Exit on close
def on_closing():
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
