import gradio as gr
import pandas as pd
import numpy as np
import re
import os
import joblib

# ─────────────────────────────────────────
# Preprocessing (sama dengan notebook EDA)
# ─────────────────────────────────────────
STOPWORDS_ID = set([
    'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan', 'untuk', 'adalah',
    'pada', 'tidak', 'juga', 'sudah', 'saya', 'anda', 'kita', 'mereka', 'kami',
    'ada', 'akan', 'bisa', 'dalam', 'oleh', 'atau', 'tetapi', 'tapi', 'karena',
    'jika', 'kalau', 'ya', 'yg', 'nya', 'bagi', 'aja', 'ga', 'gak', 'nggak',
    'dg', 'dgn', 'dr', 'utk', 'spy', 'klo', 'gitu', 'udah', 'nih', 'lah',
    'dong', 'sih', 'tuh', 'kan', 'deh', 'lagi', 'jadi', 'lebih', 'seperti',
    'hanya', 'saja', 'mau', 'sama', 'kalian', 'semua'
])


def preprocess_text(text: str) -> str:
    """Bersihkan teks: lowercase, hapus mention/URL/simbol."""
    text = str(text).lower()
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ─────────────────────────────────────────
# Load model PyCaret
# ─────────────────────────────────────────
try:
    from pycaret.classification import load_model, predict_model
    MODEL = load_model("sentiment_ml_model")
    MODEL_INFO = joblib.load("model_info.pkl") if os.path.exists("model_info.pkl") else {}
    MODEL_LOADED = True
    MODEL_NAME = MODEL_INFO.get("model_name", "ML Model")
    MODEL_ACCURACY = MODEL_INFO.get("accuracy", None)
    MODEL_F1 = MODEL_INFO.get("f1_macro", None)
except Exception as e:
    MODEL_LOADED = False
    MODEL_NAME = "Model tidak tersedia"
    print(f"[WARN] Gagal load model: {e}")


# ─────────────────────────────────────────
# Label & emoji
# ─────────────────────────────────────────
LABEL_EMOJI = {
    "Negatif": "😡",
    "Netral":  "😐",
    "Positif": "😊",
}
LABEL_COLOR_HEX = {
    "Negatif": "#e74c3c",
    "Netral":  "#95a5a6",
    "Positif": "#2ecc71",
}


# ─────────────────────────────────────────
# Fungsi prediksi
# ─────────────────────────────────────────
def predict_sentiment(text: str, score: int):
    if not text.strip():
        return "⚠️ Harap masukkan teks terlebih dahulu.", "", "", ""

    if not MODEL_LOADED:
        # Demo fallback jika model belum di-upload
        words = preprocess_text(text).split()
        neg_words = {'buruk', 'jelek', 'jahat', 'tidak', 'bohong', 'gagal', 'rusak', 'korupsi', 'parah'}
        pos_words = {'bagus', 'baik', 'mantap', 'hebat', 'luar', 'biasa', 'keren', 'sukses', 'maju'}
        neg_count = sum(1 for w in words if w in neg_words)
        pos_count = sum(1 for w in words if w in pos_words)
        if score < -1 or neg_count > pos_count:
            label = "Negatif"
        elif score > 1 or pos_count > neg_count:
            label = "Positif"
        else:
            label = "Netral"
        confidence = 0.75
        note = "⚠️ *Model demo (model PyCaret belum di-upload ke Spaces)*"
    else:
        text_clean = preprocess_text(text)
        word_count = len(text_clean.split())
        text_length = len(text_clean)

        input_df = pd.DataFrame([{
            "text_processed": text_clean,
            "score": score,
            "text_length": text_length,
            "word_count": word_count,
        }])
        result = predict_model(MODEL, data=input_df)
        label = result["prediction_label"].iloc[0]
        confidence = float(result["prediction_score"].iloc[0])
        note = ""

    emoji = LABEL_EMOJI.get(label, "")
    color = LABEL_COLOR_HEX.get(label, "#ccc")

    result_html = f"""
    <div style='text-align:center; padding:20px; border-radius:12px;
                background:{color}22; border:2px solid {color};'>
        <div style='font-size:3em'>{emoji}</div>
        <div style='font-size:1.8em; font-weight:bold; color:{color}'>{label}</div>
        <div style='font-size:1em; color:#555; margin-top:6px'>Confidence: {confidence:.1%}</div>
    </div>
    """

    bar_html = _confidence_bars(label, confidence)
    clean = preprocess_text(text)
    info = f"**Teks setelah preprocessing:**\n`{clean}`"
    if note:
        info += f"\n\n{note}"

    return result_html, bar_html, info, f"{label} ({confidence:.1%})"


def _confidence_bars(pred_label, pred_conf):
    """Buat bar sederhana untuk ketiga kelas."""
    labels = ["Negatif", "Netral", "Positif"]
    bars = ""
    for lbl in labels:
        if lbl == pred_label:
            pct = pred_conf * 100
        else:
            rem = (1 - pred_conf) / 2
            pct = rem * 100
        color = LABEL_COLOR_HEX[lbl]
        bars += f"""
        <div style='margin:4px 0'>
            <span style='display:inline-block; width:80px; font-weight:bold; color:{color}'>{lbl}</span>
            <div style='display:inline-block; width:60%; background:#eee; border-radius:6px; height:18px; vertical-align:middle'>
                <div style='width:{pct:.0f}%; background:{color}; border-radius:6px; height:18px'></div>
            </div>
            <span style='margin-left:8px; color:#555'>{pct:.1f}%</span>
        </div>
        """
    return f"<div style='padding:10px'>{bars}</div>"


def batch_predict(file):
    """Prediksi batch dari file CSV."""
    if file is None:
        return None, "⚠️ Harap upload file CSV."
    try:
        df = pd.read_csv(file.name)
        if "clean_text" not in df.columns:
            return None, "❌ CSV harus memiliki kolom `clean_text`."

        results = []
        for _, row in df.iterrows():
            text = str(row.get("clean_text", ""))
            score = int(row.get("score", 0))
            _, _, _, pred = predict_sentiment(text, score)
            results.append({
                "Teks Asli": text[:80] + "..." if len(text) > 80 else text,
                "Prediksi": pred,
                "Score": score,
            })

        result_df = pd.DataFrame(results)
        out_path = "/tmp/hasil_prediksi.csv"
        result_df.to_csv(out_path, index=False)
        return result_df, f"✅ Berhasil memproses {len(result_df)} komentar."
    except Exception as e:
        return None, f"❌ Error: {e}"


# ─────────────────────────────────────────
# Contoh teks
# ─────────────────────────────────────────
EXAMPLES = [
    ["Pemerintah ini sangat buruk dan tidak becus mengurus rakyat kecil sama sekali", -3],
    ["Kebijakan ini cukup bagus dan perlu didukung bersama untuk kemajuan bangsa", 2],
    ["Rapat hari ini membahas anggaran tahun depan seperti biasa", 0],
    ["Korupsi merajalela tidak ada yang peduli dengan nasib rakyat", -5],
    ["Selamat atas keberhasilan program ini semoga terus berlanjut", 3],
]

# ─────────────────────────────────────────
# UI Gradio
# ─────────────────────────────────────────
CSS = """
.gr-button-primary { background: #2980b9 !important; }
.title-block { text-align: center; margin-bottom: 10px; }
footer { display: none !important; }
"""

with gr.Blocks(css=CSS, title="Analisis Sentimen Politik") as demo:
    gr.HTML("""
    <div class='title-block'>
        <h1>🇮🇩 Analisis Sentimen Komentar Politik YouTube</h1>
        <p style='color:#555; font-size:1.05em'>
            Demo Model ML (PyCaret AutoML) — Kelompok 15 PBA 2026
        </p>
    </div>
    """)

    # Info card
    acc_str = f"{MODEL_INFO.get('accuracy', 0):.1%}" if MODEL_LOADED else "N/A"
    f1_str  = f"{MODEL_INFO.get('f1_macro', 0):.1%}"  if MODEL_LOADED else "N/A"
    gr.HTML(f"""
    <div style='display:flex; gap:16px; flex-wrap:wrap; justify-content:center; margin-bottom:16px'>
        <div style='background:#eaf4fb; border-radius:10px; padding:12px 20px; text-align:center; min-width:130px'>
            <div style='font-size:0.85em; color:#555'>Model</div>
            <div style='font-weight:bold; color:#2980b9'>{MODEL_NAME}</div>
        </div>
        <div style='background:#eaf4fb; border-radius:10px; padding:12px 20px; text-align:center; min-width:130px'>
            <div style='font-size:0.85em; color:#555'>Accuracy</div>
            <div style='font-weight:bold; color:#27ae60'>{acc_str}</div>
        </div>
        <div style='background:#eaf4fb; border-radius:10px; padding:12px 20px; text-align:center; min-width:130px'>
            <div style='font-size:0.85em; color:#555'>F1 Macro</div>
            <div style='font-weight:bold; color:#e67e22'>{f1_str}</div>
        </div>
        <div style='background:#eaf4fb; border-radius:10px; padding:12px 20px; text-align:center; min-width:130px'>
            <div style='font-size:0.85em; color:#555'>Dataset</div>
            <div style='font-weight:bold; color:#8e44ad'>1.204 komentar</div>
        </div>
    </div>
    """)

    with gr.Tabs():
        # ── Tab 1: Prediksi Tunggal
        with gr.TabItem("🔍 Prediksi Satu Komentar"):
            with gr.Row():
                with gr.Column(scale=3):
                    txt_input = gr.Textbox(
                        label="Masukkan Komentar",
                        placeholder="Tulis komentar YouTube bertopik politik di sini...",
                        lines=4,
                    )
                    score_input = gr.Slider(
                        label="Score Komentar (likes - dislikes)",
                        minimum=-20, maximum=20, value=0, step=1,
                    )
                    btn = gr.Button("🔮 Prediksi Sentimen", variant="primary")

                with gr.Column(scale=2):
                    result_html = gr.HTML(label="Hasil")
                    bar_html    = gr.HTML(label="Distribusi Probabilitas")

            info_md = gr.Markdown()

            btn.click(
                predict_sentiment,
                inputs=[txt_input, score_input],
                outputs=[result_html, bar_html, info_md, gr.Textbox(visible=False)],
            )

            gr.Examples(
                examples=EXAMPLES,
                inputs=[txt_input, score_input],
                label="💡 Contoh Komentar",
            )

        # ── Tab 2: Prediksi Batch CSV
        with gr.TabItem("📂 Prediksi Batch (CSV)"):
            gr.Markdown("""
            Upload file CSV dengan kolom **`clean_text`** (wajib) dan `score` (opsional).
            Sistem akan memprediksi sentimen tiap baris secara otomatis.
            """)
            file_input  = gr.File(label="Upload CSV", file_types=[".csv"])
            batch_btn   = gr.Button("🚀 Proses Batch", variant="primary")
            batch_table = gr.DataFrame(label="Hasil Prediksi")
            batch_info  = gr.Markdown()

            batch_btn.click(
                batch_predict,
                inputs=[file_input],
                outputs=[batch_table, batch_info],
            )

        # ── Tab 3: Tentang
        with gr.TabItem("ℹ️ Tentang"):
            gr.Markdown(f"""
            ## Tentang Demo Ini

            Demo ini menampilkan model **Machine Learning (ML)** untuk klasifikasi sentimen
            komentar YouTube bertopik politik menggunakan **PyCaret AutoML**.

            ### Dataset
            - **Sumber:** Komentar YouTube bertopik politik Indonesia
            - **Jumlah data:** 1.204 komentar
            - **Label:** Negatif 😡 · Netral 😐 · Positif 😊
            - **Karakteristik:** Dataset imbalanced (Negatif 63%, Netral 30%, Positif 7%)

            ### Pipeline ML
            1. **Preprocessing:** lowercase, hapus mention/URL/simbol, normalisasi spasi
            2. **Feature Engineering:** TF-IDF (5.000 fitur, n-gram 1-2) + fitur numerik (score, panjang teks)
            3. **AutoML:** Benchmark 3 algoritma via PyCaret (Logistic Regression, Random Forest, Naive Bayes)
            4. **Evaluasi:** 5-Fold Cross Validation, metrik F1 Macro (prioritas karena imbalanced)
            5. **Model terpilih:** {MODEL_NAME}

            ### Tim
            **Kelompok 15 — Pengantar Big Data Analytics (PBA) 2026**

            📦 [GitHub Repository](https://github.com/ihsanmaulanayusuf/pba2026-kelompok-15)
            """)

    gr.HTML("""
    <div style='text-align:center; margin-top:20px; color:#aaa; font-size:0.85em'>
        Kelompok 15 · PBA 2026 · Hugging Face Spaces ML Demo
    </div>
    """)


if __name__ == "__main__":
    demo.launch()
