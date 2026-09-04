from google import genai

client = genai.Client(api_key="your_api")
from torchvision import models, transforms
import torch.nn as nn
import torch
descriere=input()
model = models.efficientnet_b3(weights=None)
num_classes = 10
num_intrari = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(num_intrari, num_classes)
)
state_dict = torch.load("classifier_weights.pth", map_location='cpu')
model.load_state_dict(state_dict)
model.eval()
import tkinter as tk
from tkinter import filedialog

def select_image():
    root = tk.Tk()
    root.withdraw()  

    
    root.attributes('-topmost', True)
    root.focus_force()
    
    image_path = filedialog.askopenfilename(
        title="Alege o imagine",
        filetypes=[("Imagini", "*.jpg *.jpeg *.png *.bmp *.webp")]
    )
    
    root.destroy()
    
    return image_path


calea_imaginii = select_image()
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

def predict_single_image(image_path, model, class_names=None, device='cpu'):
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=-1)
        
        top2_probs, top2_indices = torch.topk(probs, k=2, dim=-1)

    results = []
    for prob, idx in zip(top2_probs[0], top2_indices[0]):
        class_id = idx.item()
        label = class_id
        results.append({
            'clasa': label,
            'probabilitate': round(prob.item() * 100, 2)
        })

    return results

top_2 = predict_single_image(calea_imaginii, model)
dictionar_curatat = {
    "nv": "Nev melanocitar (aluniță benignă)",
    "bcc": "Carcinom bazocelular",
    "mel": "Melanom malign",
    "ack": "Keratoză actinică",
    "akiec": "Keratoză actinică / Carcinom in situ",
    "sek": "Keratoză seboreică",
    "bkl": "Keratoză benignă generală",
    "scc": "Carcinom spinocelular",
    "vasc": "Leziune vasculară",
    "df": "Dermatofibrom",
}
import pandas as pd 
data= pd.DataFrame(top_2)
import joblib

le = joblib.load("label_encoder.joblib")

clase_prescurtate = le.inverse_transform(data['clasa'].values)

data['clasa_abreviere'] = clase_prescurtate

data['nume_diagnostic'] = data['clasa_abreviere'].map(dictionar_curatat)

print(data)
dat= data[['nume_diagnostic','probabilitate']]
rezultat_text = " | ".join([f"{rand['nume_diagnostic']}: {rand['probabilitate']}%" for index, rand in dat.iterrows()])
prompt = f"""Ești un asistent virtual empatic și responsabil. 

Aici sunt cele mai probabile două rezultate generate de modelul de clasificare pentru afecțiunea pielii utilizatorului:
{rezultat_text}

Context și descriere suplimentară a problemei:
{descriere}

Sarcina ta:
Scrie un mesaj scurt (2-3 paragrafe) adresat utilizatorului în care să incluzi următoarele:
1. Prezintă-i, pe un ton calm și informativ, cele două probabilități de mai sus și integrează natural detaliile din contextul suplimentar.
2. Explică foarte clar și ferm că acesta NU este un diagnostic medical, ci doar o estimare oferită de o aplicație AI.
3. Sfătuiește utilizatorul ca pasul următor și cel mai important este să consulte un medic dermatolog pentru un diagnostic precis și tratament.
"""
response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt,
)

print(response.text)