import streamlit as st
import pandas as pd
import numpy as np
from rapidfuzz import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import os

st.set_page_config(page_title="Anime Recommender", layout="wide")
st.title("Anime Recommender System")
st.write("Beğendiğiniz 1-5 animeyi girin, size benzeyenleri önerelim.")

# CSV kontrolü
csv_path = "anime_cleaned.csv"
if not os.path.exists(csv_path):
    st.error("'anime_cleaned.csv' dosyası bulunamadı. Lütfen aynı klasöre ekleyin.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(csv_path)
    df = df[df["synopsis"].str.len() >= 30].copy().reset_index(drop=True)
    df['combined_text'] = df['genres'].fillna('') + ' ' + df['synopsis'].fillna('')
    return df

df = load_data()

@st.cache_resource(show_spinner=False)
def get_vectorizer_and_matrix(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = get_vectorizer_and_matrix(df)

def normalize(text):
    return ''.join(e.lower() for e in text if e.isalnum() or e.isspace()).strip()

@st.cache_data
def get_autocomplete_options():
    return df["base_name"].dropna().unique().tolist()

@st.cache_data(show_spinner=False)
def get_anime_info(anime_name):
    try:
        url = f"https://api.jikan.moe/v4/anime?q={anime_name}&limit=1"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data["data"]:
                anime = data["data"][0]
                image_url = anime["images"]["jpg"]["image_url"]
                synopsis = anime.get("synopsis", "Açıklama bulunamadı.")
                return image_url, synopsis
    except:
        return None, None
    return None, None

# =======================
# 🎯 ARAYÜZ
# =======================
st.subheader("Beğendiğiniz Animeleri Girin (En fazla 5 tane)")

autocomplete_list = get_autocomplete_options()

if "clear_anime_inputs" not in st.session_state:
    st.session_state.clear_anime_inputs = False

if st.session_state.clear_anime_inputs:
    anime_inputs = []
    st.session_state.clear_anime_inputs = False
else:
    anime_inputs = st.multiselect(
        "Beğendiğiniz Animeleri Yazın ve Seçin:",
        options=autocomplete_list,
        max_selections=5
    )

for anime in anime_inputs:
    img_url, synopsis = get_anime_info(anime)
    col1, col2 = st.columns([1, 4])
    with col1:
        if img_url:
            st.image(img_url, width=90)
        else:
            st.write(" Yok")

    with col2:
        st.markdown(f"### {anime}")
        if synopsis:
            first_sentence = synopsis.split(".")[0] + "."
            st.markdown(f"{first_sentence}", unsafe_allow_html=True)
            with st.expander("devamını okumak için tıklayınız"):
                st.markdown(synopsis, unsafe_allow_html=True)

# Temizleme butonu
if st.button("Seçilenleri Temizle"):
    st.session_state.clear_anime_inputs = True
    st.rerun()

# =======================
# 🔍 ÖNERİ MOTORU
# =======================
# =======================
# 🔍 ÖNERİ MOTORU
# =======================
if st.button("Önerileri Getir"):
    if not anime_inputs:
        st.warning("Lütfen en az 1 anime girin.")
        st.stop()

    matched_indices = []
    for anime in anime_inputs:
        cleaned_input = normalize(anime)
        match = process.extractOne(cleaned_input, df["base_name"].tolist(), score_cutoff=80)
        if match:
            idx = df[df["base_name"] == match[0]].index
            if not idx.empty:
                matched_indices.append(idx[0])
                st.success(f" '{anime}' → '{match[0]}'")
        else:
            st.warning(f" '{anime}' için eşleşme bulunamadı.")

    if not matched_indices:
        st.error("Hiçbir eşleşme bulunamadı.")
        st.stop()

    input_vecs = tfidf_matrix[matched_indices]
    mean_vec = np.asarray(input_vecs.mean(axis=0))

    cosine_sim = cosine_similarity(mean_vec, tfidf_matrix)[0]

    # ❗ Sadece seçilenlerin dışındaki animeleri puanla
    scored = []
    for i, sim in enumerate(cosine_sim):
        if i not in matched_indices:
            qual = df.iloc[i]['quality_score'] if not pd.isna(df.iloc[i]['quality_score']) else 0
            score = 0.8 * sim + 0.2 * qual
            scored.append((i, score))

    top10 = sorted(scored, key=lambda x: x[1], reverse=True)[:10]
    results = df.iloc[[i for i, _ in top10]].reset_index(drop=True)

    st.subheader("En İyi 10 Öneri:")

    for idx, row in results.iterrows():
        img_url, synopsis = get_anime_info(row["name"])
        first_sentence = synopsis.split(".")[0] + "." if synopsis else ""

        col1, col2 = st.columns([1, 4])
        with col1:
            if img_url:
                st.image(img_url, width=150)  

            else:
                st.write(" Yok")

        with col2:
            st.markdown(f"### {row['name']}")
            st.markdown(
                f" **Skor**: {row['score']:.2f}  \n"
                f" **Tür**: {row['genres']}  \n"
                f" **Kalite**: {row['quality_score']:.2f}"
            )
            if first_sentence:
                st.markdown(f"{first_sentence}", unsafe_allow_html=True)
            if synopsis:
                with st.expander(" devamını okumak için tıklayınız"):
                    st.markdown(synopsis, unsafe_allow_html=True)

