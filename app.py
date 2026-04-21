import gradio as gr
import pandas as pd
from pycaret.classification import load_model, predict_model
import torch
import torch.nn as nn
import pickle
import re
import os
import subprocess
import sys

# ==========================================
# BAGIAN 1: MODEL MACHINE LEARNING (PYCARET)
# ==========================================
model_ml = load_model("model_ml/sentiment_ml_model")

def predict_sentiment_ml(text):
    text_processed = text.lower()
    
    # Aturan Deteksi Cepat Berdasarkan Keyword Baru (Lexicon-based override)
    negative_words = [
        "nyawit", "mbgnya mana wok", "wowok jelek", "jelek", "buruk", "parah", "kecewa", "hancur", 
        "gagal", "tolol", "bodoh", "gila", "korup", "ngawur", "hoaks", "pembohong", "dungu", 
        "sombong", "bocil", "buzzer", "cebong", "kampret", "kadrun", "penipu", "sampah", "najis", 
        "munafik", "pecundang", "cacat", "goblok", "bego", "tolak", "anti", "makar", "pengkhianat", 
        "berengsek", "brengsek", "fufufafa", "curang", "licik", "nepotisme", "dinasti", "kkn"
    ]
    positive_words = [
        "keren", "hebat", "bagus", "mantap", "terbaik", "luar biasa", "salut", "top", "good", 
        "oke", "mantul", "cerdas", "pintar", "bijak", "maju", "sukses", "bangga", "wibawa", 
        "merakyat", "jujur", "amanah", "gas", "dukung", "setuju", "lanjutkan", "menyala", 
        "gacor", "cinta", "sayang", "harapan", "pro"
    ]
    
    for word in negative_words:
        if word in text_processed:
            return "Negatif"
            
    for word in positive_words:
        if word in text_processed:
            return "Positif"

    text_length = len(text_processed)
    word_count = len(text_processed.split())
    
    df_input = pd.DataFrame({
        'text_processed': [text_processed],
        'text_length': [text_length],
        'word_count': [word_count]
    })
    
    predictions = predict_model(model_ml, data=df_input)
    if 'prediction_label' in predictions.columns:
        result = predictions['prediction_label'].iloc[0]
    else:
        result = predictions['Label'].iloc[0]
        
    return result


# ==========================================
# BAGIAN 2: MODEL DEEP LEARNING (PYTORCH)
# ==========================================
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        output = self.fc(hidden[-1])
        return output

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

vocab_path = "model_dl/vocab.pkl"
model_path = "model_dl/lstm_model.pt"

# Auto-train if missing
if not (os.path.exists(vocab_path) and os.path.exists(model_path)):
    print("Model DL belum ada. Melakukan training otomatis...")
    subprocess.run([sys.executable, "src/train_dl.py"], check=True)

if os.path.exists(vocab_path) and os.path.exists(model_path):
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
        
    device = torch.device('cpu')
    model_dl = LSTMClassifier(len(vocab), embed_dim=64, hidden_dim=128, output_dim=3)
    model_dl.load_state_dict(torch.load(model_path, map_location=device))
    model_dl.eval()
else:
    vocab = {}
    model_dl = None

label_mapping = {0: "Negatif", 1: "Netral", 2: "Positif"}

def predict_sentiment_dl(text):
    if model_dl is None:
        return "Error: Model DL belum dilatih atau tidak ditemukan."
        
    text_processed = text.lower()
    
    # Lexicon-based override
    negative_words = [
        "nyawit", "mbgnya mana wok", "wowok jelek", "jelek", "buruk", "parah", "kecewa", "hancur", 
        "gagal", "tolol", "bodoh", "gila", "korup", "ngawur", "hoaks", "pembohong", "dungu", 
        "sombong", "bocil", "buzzer", "cebong", "kampret", "kadrun", "penipu", "sampah", "najis", 
        "munafik", "pecundang", "cacat", "goblok", "bego", "tolak", "anti", "makar", "pengkhianat", 
        "berengsek", "brengsek", "fufufafa", "curang", "licik", "nepotisme", "dinasti", "kkn"
    ]
    positive_words = [
        "keren", "hebat", "bagus", "mantap", "terbaik", "luar biasa", "salut", "top", "good", 
        "oke", "mantul", "cerdas", "pintar", "bijak", "maju", "sukses", "bangga", "wibawa", 
        "merakyat", "jujur", "amanah", "gas", "dukung", "setuju", "lanjutkan", "menyala", 
        "gacor", "cinta", "sayang", "harapan", "pro"
    ]
    
    for word in negative_words:
        if word in text_processed:
            return "Negatif"
            
    for word in positive_words:
        if word in text_processed:
            return "Positif"

    tokens = tokenize(text_processed)
    indices = [vocab.get(t, vocab.get('<UNK>', 1)) for t in tokens]
    
    max_len = 50
    if len(indices) < max_len:
        indices = indices + [vocab.get('<PAD>', 0)] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
        
    tensor_input = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        output = model_dl(tensor_input)
        _, predicted = torch.max(output, 1)
        pred_label_idx = predicted.item()
        
    return label_mapping.get(pred_label_idx, "Netral")

# ==========================================
# BAGIAN 3: MEMBANGUN ANTARMUKA GRADIO (TABS)
# ==========================================
iface_ml = gr.Interface(
    fn=predict_sentiment_ml,
    inputs=gr.Textbox(lines=3, placeholder="Masukkan komentar di sini..."),
    outputs=gr.Text(label="Hasil Prediksi Sentimen"),
    title="Analisis Sentimen Politik di X (Twitter)",
    description="""Aplikasi ini memproses *dataset tweet* dari X (Twitter) terkait panggung perpolitikan Indonesia untuk diklasifikasikan ke dalam sentimen Positif, Netral, maupun Negatif.

🤖 **Tentang Model (Machine Learning):**
Versi ini menggunakan kerangka **PyCaret AutoML** yang menyeleksi dan melatih algoritme ML konvensional terbaik secara otomatis. Selain pemodelan statistik, kami juga menyematkan aturan baca (*Lexicon / Keyword filter*) untuk mendeteksi *slang* media sosial dengan presisi tinggi."""
)

iface_dl = gr.Interface(
    fn=predict_sentiment_dl,
    inputs=gr.Textbox(lines=3, placeholder="Masukkan komentar di sini..."),
    outputs=gr.Text(label="Hasil Prediksi Sentimen (Deep Learning PyTorch)"),
    title="Analisis Sentimen Politik di X (Twitter) - Versi Deep Learning",
    description="""Aplikasi ini memproses *dataset tweet* dari X (Twitter) terkait panggung perpolitikan Indonesia untuk diklasifikasikan ke dalam sentimen Positif, Netral, maupun Negatif.

🧠 **Tentang Model (Deep Learning):**
Berbeda dengan versi standar, versi ini sangat mutakhir karena digerakkan oleh algoritma **Deep Learning**. Kami mengimplementasikan arsitektur **LSTM (Long Short-Term Memory)** via **PyTorch** yang memiliki daya ingat tinggi terhadap posisi kata penentu pada kalimat panjang, dibantu filter leksikon (*Lexicon*) terpadu."""
)

# Integrasikan Keduanya Menjadi Sistem Multi-Tab
app = gr.TabbedInterface(
    [iface_ml, iface_dl],
    ["✨ Machine Learning (PyCaret)", "🧠 Deep Learning (PyTorch)"]
)

if __name__ == "__main__":
    app.launch()
