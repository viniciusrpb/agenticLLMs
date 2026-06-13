import os
import json
import unicodedata

import streamlit as st

st.set_page_config(page_title="UnB Agente", layout="wide")
st.title("UnB Agente")
st.caption("Notícias e pesquisadores da Universidade de Brasília")

def normalizar(s):

    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().strip()

@st.cache_data(show_spinner=False)
def carregar_noticias():

    base_dir = os.path.join(os.path.dirname(__file__), "database", "noticias")
    noticias = []

    if not os.path.isdir(base_dir):
        st.warning(f"Pasta '{base_dir}' não encontrada.")
        return noticias

    for filename in os.listdir(base_dir):
        if not filename.endswith(".json"):
            continue
        try:
            with open(os.path.join(base_dir, filename), encoding="utf-8") as f:
                items = json.load(f)
        except Exception:
            continue

        if isinstance(items, dict):
            items = [items]

        for item in items:

            titulo = str(item.get("titulo", item.get("title", ""))).strip()
            texto = str(item.get("texto",  item.get("content", item.get("text", "")))).strip()
            url = str(item.get("url", "")).strip()
            data = str(item.get("data",   item.get("date", ""))).strip()

            if not titulo or not texto:
                continue

            noticias.append({
                "titulo": titulo,
                "texto":  texto,
                "url":    url,
                "data":   data,
            })

    return noticias


@st.cache_data(show_spinner=False)
def carregar_lattes():

    base_dir = os.path.join(os.path.dirname(__file__), "database", "lattes")
    pesquisadores = []

    if not os.path.isdir(base_dir):
        st.warning(f"Pasta '{base_dir}' não encontrada.")
        return pesquisadores

    for filename in os.listdir(base_dir):
        if not filename.endswith(".json"):
            continue
        try:
            with open(os.path.join(base_dir, filename), encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        if isinstance(data, list):
            pesquisadores.extend(data)
        elif isinstance(data, dict):
            pesquisadores.append(data)

    return pesquisadores


if st.button("Limpar conversa"):
    st.session_state["history"] = []
    st.rerun()

if "history" not in st.session_state:
    st.session_state["history"] = []

for msg in st.session_state["history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Pergunte..."):
    st.session_state["history"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando bases de conhecimento..."):
            resposta = responder(prompt, st.session_state["history"][:-1])
        st.markdown(resposta)

    st.session_state["history"].append({"role": "assistant", "content": resposta})
    st.rerun()
