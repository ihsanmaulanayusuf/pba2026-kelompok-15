import gradio as gr
import torch
import torch.nn as nn
import pickle
import re
import os

# 1. Definisikan Kelas Model LSTM (Harus sama persis dengan yang dilatih)
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

# 2. Sederhana tokenizer
def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

# 3. Load Vocab and Model
vocab_path = "model_dl/vocab.pkl"
model_path = "model_dl/lstm_model.pt"

# Auto-train if missing (Berguna saat di-deploy ke Hugging Face)
if not (os.path.exists(vocab_path) and os.path.exists(model_path)):
    print("Model belum ada. Melakukan training otomatis...")
    import subprocess
    import sys
    # Run training
    subprocess.run([sys.executable, "src/train_dl.py"], check=True)

if os.path.exists(vocab_path) and os.path.exists(model_path):
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
        
    device = torch.device('cpu') # Deployment HF space gratis menggunakan CPU
    model = LSTMClassifier(len(vocab), embed_dim=64, hidden_dim=128, output_dim=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
else:
    vocab = {}
    model = None

label_mapping = {0: "Negatif", 1: "Netral", 2: "Positif"}

def predict_sentiment_dl(text):
    if model is None:
        return "Error: Model DL belum dilatih atau tidak ditemukan."
        
    text_processed = text.lower()
    
    # 1. Lexicon-based override (Aturan Deteksi Cepat Berdasarkan Keyword Baru)
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

    # 2. Proses Teks menggunakan Vocab dan Model DL
    tokens = tokenize(text_processed)
    indices = [vocab.get(t, vocab.get('<UNK>', 1)) for t in tokens]
    
    # Pad to match max_len = 50 (seperti saat training)
    max_len = 50
    if len(indices) < max_len:
        indices = indices + [vocab.get('<PAD>', 0)] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
        
    tensor_input = torch.tensor(indices, dtype=torch.long).unsqueeze(0) # Tambah batch dimension

    # 3. Prediksi menggunakan model PyTorch
    with torch.no_grad():
        output = model(tensor_input)
        _, predicted = torch.max(output, 1)
        pred_label_idx = predicted.item()
        
    return label_mapping.get(pred_label_idx, "Netral")

# 4. Bangun UI dengan Gradio
iface = gr.Interface(
    fn=predict_sentiment_dl,
    inputs=gr.Textbox(lines=3, placeholder="Masukkan komentar di sini..."),
    outputs=gr.Text(label="Hasil Prediksi Sentimen (Deep Learning PyTorch)"),
    title="Analisis Sentimen Politik di X (Twitter) - Versi Deep Learning",
    description="""Aplikasi ini memproses dataset tweet dari X (Twitter) terkait panggung perpolitikan Indonesia untuk diklasifikasikan ke dalam sentimen Positif, Netral, maupun Negatif.

Tentang Model (Deep Learning):
Berbeda dengan versi standar, versi ini sangat mutakhir karena digerakkan oleh algoritma Deep Learning. Kami mengimplementasikan arsitektur LSTM (Long Short-Term Memory) via PyTorch yang memiliki daya ingat tinggi terhadap posisi kata penentu pada kalimat panjang, dibantu filter leksikon (Lexicon) terpadu."""
)

if __name__ == "__main__":
    iface.launch()
