import gradio as gr
import pandas as pd
from pycaret.classification import load_model, predict_model

# Load model Pipeline PyCaret
# Perlu diperhatikan tanpa '.pkl' di namanya karena pycaret akan menambahkannya secara otomatis.
model = load_model("model_ml/sentiment_ml_model")

def predict_sentiment(text):
    # 1. Lakukan pre-processing dasar agar sesuai dengan input model
    text_processed = text.lower() # Sederhanakan (idealnya sama persis dengan tahap pre-processing di ipynb)
    text_length = len(text_processed)
    word_count = len(text_processed.split())
    
    # 2. Buat DataFrame tunggal (Tanpa fitur score untuk mencegah Data Leakage)
    df_input = pd.DataFrame({
        'text_processed': [text_processed],
        'text_length': [text_length],
        'word_count': [word_count]
    })
    
    # 3. Prediksi dengan model
    predictions = predict_model(model, data=df_input)
    
    # Ambil hasil label yang diprediksi
    # Notes: PyCaret 3.x menggunakan kolom 'prediction_label' atau 'Label'
    if 'prediction_label' in predictions.columns:
        result = predictions['prediction_label'].iloc[0]
    else:
        result = predictions['Label'].iloc[0]
        
    return result

# 4. Bangun UI dengan Gradio
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Masukkan komentar di sini..."),
    outputs=gr.Text(label="Hasil Prediksi Sentimen"),
    title="Analisis Sentimen Komentar YouTube",
    description="Aplikasi ini menggunakan PyCaret AutoML untuk memprediksi sentimen komentar YouTube politik Indonesia."
)

if __name__ == "__main__":
    iface.launch()
