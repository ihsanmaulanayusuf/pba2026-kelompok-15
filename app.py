import gradio as gr
import pandas as pd
from pycaret.classification import load_model, predict_model

# Load model Pipeline PyCaret
# Perlu diperhatikan tanpa '.pkl' di namanya karena pycaret akan menambahkannya secara otomatis.
model = load_model("model_ml/sentiment_ml_model")

def predict_sentiment(text):
    text_processed = text.lower() # Sederhanakan (idealnya sama persis dengan tahap pre-processing di ipynb)
    
    # 1. Aturan Deteksi Cepat Berdasarkan Keyword Baru (Lexicon-based override)
    negative_words = ["nyawit", "mbgnya mana wok", "wowok jelek", "jelek", "buruk", "parah", "kecewa", "hancur", "gagal", "tolol", "bodoh", "gila", "korup"]
    positive_words = ["keren", "hebat", "bagus", "mantap", "terbaik", "luar biasa", "salut", "top", "good", "oke", "mantul"]
    
    for word in negative_words:
        if word in text_processed:
            return "Negatif"
            
    for word in positive_words:
        if word in text_processed:
            return "Positif"
            
    # 2. Lakukan pre-processing dasar agar sesuai dengan input model
    text_length = len(text_processed)
    word_count = len(text_processed.split())
    
    # 3. Buat DataFrame tunggal (Tanpa fitur score untuk mencegah Data Leakage)
    df_input = pd.DataFrame({
        'text_processed': [text_processed],
        'text_length': [text_length],
        'word_count': [word_count]
    })
    
    # 4. Prediksi dengan model
    predictions = predict_model(model, data=df_input)
    
    # Ambil hasil label yang diprediksi
    # Notes: PyCaret 3.x menggunakan kolom 'prediction_label' atau 'Label'
    if 'prediction_label' in predictions.columns:
        result = predictions['prediction_label'].iloc[0]
    else:
        result = predictions['Label'].iloc[0]
        
    return result

# 5. Bangun UI dengan Gradio
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Masukkan komentar di sini..."),
    outputs=gr.Text(label="Hasil Prediksi Sentimen"),
    title="Analisis Sentimen Politik di X (Twitter)",
    description="sentimen tentang politik di indonesia dari kolom komentar di x"
)

if __name__ == "__main__":
    iface.launch()
