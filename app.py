

import os
import io
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import streamlit as st
from PIL import Image
from torchvision import models, transforms
from google import genai
from google.genai import types


st.set_page_config(
    page_title="DermAI · Pre-diagnosticare",
    page_icon="🩺",
    layout="centered",
)

st.markdown(
    """
    <style>
        .main { background-color: #fafbfc; }
        .stApp { max-width: 900px; margin: 0 auto; }
        .result-card {
            background: white;
            border-radius: 14px;
            padding: 1rem 1.3rem;
            margin-bottom: 0.7rem;
            border: 1px solid #e6e8eb;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }
        .result-title { font-weight: 600; font-size: 1.05rem; color: #1f2933; }
        .result-bar-bg {
            background: #eef1f4;
            border-radius: 6px;
            height: 10px;
            margin-top: 6px;
            overflow: hidden;
        }
        .result-bar-fill {
            background: linear-gradient(90deg, #4f8ff0, #6fb1ff);
            height: 100%;
            border-radius: 6px;
        }
        .disclaimer {
            background: #fff7e6;
            border-left: 4px solid #f0ad4e;
            padding: 0.8rem 1rem;
            border-radius: 8px;
            font-size: 0.92rem;
            color: #6b4d0a;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🩺 DermAI — Pre-diagnosticare dermatologică")
st.caption(
    "Încarcă o fotografie a leziunii cutanate și primești o estimare orientativă, "
    "generată de un model AI — NU un diagnostic medical."
)

st.markdown(
    """
    <div class="disclaimer">
    ⚠️ Această aplicație <b>nu oferă diagnostice medicale</b>. Rezultatele sunt estimări
    generate automat și trebuie confirmate întotdeauna de un medic dermatolog.
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

DICTIONAR_CLASE = {
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


@st.cache_resource(show_spinner="Se încarcă modelul AI...")
def load_model(weights_path: str = "classifier_weights.pth", num_classes: int = 10):
    model = models.efficientnet_b3(weights=None)
    num_intrari = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_intrari, num_classes),
    )
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


@st.cache_resource(show_spinner=False)
def load_label_encoder(path: str = "label_encoder.joblib"):
    return joblib.load(path)


def predict_top2(image: Image.Image, model, device: str = "cpu"):
    transform = transforms.Compose(
        [transforms.Resize((300, 300)), transforms.ToTensor()]
    )
    input_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    model.to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=-1)
        top2_probs, top2_indices = torch.topk(probs, k=2, dim=-1)

    results = []
    for prob, idx in zip(top2_probs[0], top2_indices[0]):
        results.append({"clasa": idx.item(), "probabilitate": round(prob.item() * 100, 2)})
    return results


def build_prompt(rezultat_text: str, descriere: str) -> str:
    return f"""Ești un asistent virtual empatic și responsabil.

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


with st.sidebar:
    st.header("⚙️ Configurare")
    api_key = st.text_input(
        "Cheie API Gemini",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        help="Cheia ta nu este stocată — se folosește doar pentru sesiunea curentă.",
    )
    st.divider()
    st.markdown(
        "**Cum funcționează:**\n"
        "1. Încarci o poză cu leziunea\n"
        "2. Adaugi o scurtă descriere\n"
        "3. Modelul AI estimează cele mai probabile 2 afecțiuni\n"
        "4. Poți pune întrebări suplimentare în chat"
    )



if "chat" not in st.session_state:
    st.session_state.chat = None
if "history" not in st.session_state:
    st.session_state.history = []
if "analizat" not in st.session_state:
    st.session_state.analizat = False



col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader(
        "📷 Încarcă o fotografie a leziunii",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )
    if uploaded_file is not None:
        image = Image.open(io.BytesIO(uploaded_file.read()))
        st.image(image, caption="Imagine încărcată", use_container_width=True)

with col2:
    descriere = st.text_area(
        "📝 Descrie simptomele / contextul",
        placeholder=(
            "Ex: leziunea a apărut acum 3 luni, s-a mărit ușor, "
            "nu doare, dar uneori mă mănâncă..."
        ),
        height=180,
    )

analizeaza = st.button("🔍 Analizează", type="primary", use_container_width=True)


#
if analizeaza:
    if uploaded_file is None:
        st.warning("Te rog încarcă o imagine înainte de a analiza.")
    elif not api_key:
        st.warning("Te rog introdu cheia API Gemini în bara laterală.")
    elif not os.path.exists("classifier_weights.pth") or not os.path.exists("label_encoder.joblib"):
        st.error(
            "Nu găsesc fișierele `classifier_weights.pth` și/sau `label_encoder.joblib` "
            "în directorul aplicației. Adaugă-le și repornește."
        )
    else:
        with st.spinner("Se analizează imaginea..."):
            model = load_model()
            le = load_label_encoder()

            top_2 = predict_top2(image, model)
            data = pd.DataFrame(top_2)
            data["clasa_abreviere"] = le.inverse_transform(data["clasa"].values)
            data["nume_diagnostic"] = data["clasa_abreviere"].map(DICTIONAR_CLASE)

        st.session_state.analizat = True
        st.session_state.data = data

        st.subheader("📊 Rezultate estimare")
        for _, rand in data.iterrows():
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-title">{rand['nume_diagnostic']}</div>
                    <div class="result-bar-bg">
                        <div class="result-bar-fill" style="width:{rand['probabilitate']}%;"></div>
                    </div>
                    <div style="text-align:right; font-size:0.85rem; color:#555; margin-top:2px;">
                        {rand['probabilitate']}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        rezultat_text = " | ".join(
            f"{rand['nume_diagnostic']}: {rand['probabilitate']}%"
            for _, rand in data.iterrows()
        )

        with st.spinner("Se generează explicația AI..."):
            try:
                client = genai.Client(api_key=api_key)
                prompt = build_prompt(rezultat_text, descriere or "Nu a fost oferită o descriere.")
                chat = client.chats.create(
                    model="gemini-3.6-flash",
                    config=types.GenerateContentConfig(system_instruction=prompt),
                )
                response = chat.send_message(prompt)
                st.session_state.chat = chat
                st.session_state.history = [("assistant", response.text)]
            except Exception as e:
                st.error(f"Eroare la apelul API: {e}")


if st.session_state.history:
    st.subheader("💬 Explicație și discuție")

    for role, text in st.session_state.history:
        with st.chat_message(role):
            st.write(text)

    intrebare = st.chat_input("Scrie o întrebare despre rezultate...")
    if intrebare:
        st.session_state.history.append(("user", intrebare))
        with st.chat_message("user"):
            st.write(intrebare)

        with st.chat_message("assistant"):
            with st.spinner("..."):
                try:
                    raspuns = st.session_state.chat.send_message(intrebare)
                    st.write(raspuns.text)
                    st.session_state.history.append(("assistant", raspuns.text))
                except Exception as e:
                    st.error(f"Eroare: {e}")

st.divider()
st.caption(
    "DermAI este un instrument informativ și nu înlocuiește consultul medical de specialitate."
)
