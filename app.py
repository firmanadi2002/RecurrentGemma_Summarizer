# app.py
# Jalankan dengan: streamlit run app.py

import streamlit as st
import jax
import flax.serialization as serialization
import sentencepiece as spm
from recurrentgemma import jax as recurrentgemma
import pathlib
import os
import requests

# ========================================
# KONFIGURASI
# ========================================
MODEL_URL = "https://huggingface.co/Fugaki/RecurrentGemma_IndonesiaSummarizerNews/resolve/main/model.msgpack"  # Ganti dengan FILE_ID kamu
MODEL_FILE = "model.msgpack"
TOKENIZER_FILE = "tokenizer.model"
PRESET_VARIANT = "2b"
GENERATION_STEPS = 120
MAX_INPUT_LENGTH = 1024

# ========================================
# UNDUH MODEL JIKA PERLU
# ========================================
def download_model_if_needed():
    if not os.path.exists(MODEL_FILE):
        st.info("📥 Mengunduh model dari Google Drive...")
        gdown.download(MODEL_URL, MODEL_FILE, quiet=False)


# ========================================
# TOKENIZER WRAPPER
# ========================================
class GriffinTokenizer:
    def __init__(self, sp_model):
        self._sp_model = sp_model

    def tokenize(self, text, prefix="", suffix="", add_eos=True):
        tokens = [self._sp_model.bos_id()]
        tokens += self._sp_model.EncodeAsIds(prefix + text + suffix)
        if add_eos:
            tokens.append(self._sp_model.eos_id())
        return jax.numpy.array(tokens, dtype=jax.numpy.int32)

    def to_string(self, tokens):
        return self._sp_model.DecodeIds(tokens.tolist())

# ========================================
# MUAT MODEL & TOKENIZER
# ========================================
@st.cache_resource
def load_model_and_tokenizer():
    artifacts_path = pathlib.Path(__file__).parent.resolve()
    model_path = artifacts_path / MODEL_FILE
    tokenizer_path = artifacts_path / TOKENIZER_FILE

    if not model_path.exists() or not tokenizer_path.exists():
        st.error("Pastikan file model dan tokenizer tersedia.")
        st.stop()

    sp = spm.SentencePieceProcessor()
    sp.Load(str(tokenizer_path))
    tokenizer = GriffinTokenizer(sp)

    preset = recurrentgemma.Preset.RECURRENT_GEMMA_2B_V1
    model_config = recurrentgemma.GriffinConfig.from_preset(preset)
    model = recurrentgemma.Griffin(model_config)

    with open(model_path, "rb") as f:
        model_bytes = f.read()

    dummy_tokens = jax.numpy.zeros((1, 1), dtype=jax.numpy.int32)
    dummy_positions = jax.numpy.zeros((1, 1), dtype=jax.numpy.int32)
    key = jax.random.PRNGKey(0)
    initial_params = model.init({'params': key, 'dropout': key}, dummy_tokens, dummy_positions)['params']
    trained_params = serialization.from_bytes(initial_params, model_bytes)

    sampler = recurrentgemma.Sampler(
        model=model,
        params=trained_params,
        vocab=sp
    )

    return sampler, tokenizer

# ========================================
# STREAMLIT APP
# ========================================
st.set_page_config(page_title="📰 Peringkas Berita", layout="wide")
st.title("📰 Aplikasi Peringkas Berita Otomatis")
st.markdown("Ditenagai oleh `RecurrentGemma-2B` yang telah di-fine-tuning.")

with st.spinner("Memuat model..."):
    sampler, tokenizer = load_model_and_tokenizer()
st.success("✅ Model berhasil dimuat!")

st.subheader("Masukkan Teks Berita")
input_text = st.text_area(
    label="Teks berita",
    placeholder="Contoh: JAKARTA, KOMPAS.com – Pertandingan sengit antara Timnas Indonesia dan Timnas Vietnam berakhir dengan skor imbang...",
    height=300
)

if st.button("✨ Ringkas Sekarang", type="primary", use_container_width=True):
    if not input_text.strip():
        st.warning("Masukkan teks yang valid untuk diringkas.")
    elif len(input_text.strip()) < 50:
        st.warning("Teks terlalu pendek. Harap masukkan teks yang lebih panjang.")
    else:
        with st.spinner("Meringkas..."):
            prompt = (
                "Ringkas paragraf berikut dalam satu paragraf yang komprehensif dan padat:\n\n"
                f"{input_text.strip()}\n"
            )

            input_tokens = tokenizer.tokenize(prompt, add_eos=False)

            if len(input_tokens) > MAX_INPUT_LENGTH:
                st.info("Teks terlalu panjang, akan dipotong.")
                input_tokens = input_tokens[:MAX_INPUT_LENGTH]
                prompt = tokenizer.to_string(input_tokens)

            output = sampler(
                [prompt],
                total_generation_steps=GENERATION_STEPS
            )

            summary = output.text[0].replace("<pad>", "").replace("<eos>", "").strip()

        st.subheader("📄 Ringkasan Hasil")
        st.success(summary)
