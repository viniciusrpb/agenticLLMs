# =============================================================================
# UnB Research Agent
# Agente conversacional que responde perguntas sobre notícias e pesquisadores
# da Universidade de Brasília usando RAG (Retrieval-Augmented Generation).
# =============================================================================

import os
import json
import unicodedata

import streamlit as st

# --- configuração da página Streamlit ---
st.set_page_config(page_title="UnB Research Agent", layout="wide")
st.title("🎓 UnB Research Agent")
st.caption("Agente com conhecimento sobre notícias e pesquisadores da Universidade de Brasília")


# =============================================================================
# UTILITÁRIOS
# =============================================================================

def normalizar(s):
    # Remove acentos e converte para minúsculas.
    # Usado para comparar nomes sem se preocupar com acentuação.
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().strip()


# =============================================================================
# CARREGAMENTO DE DADOS
# =============================================================================

@st.cache_data(show_spinner=False)
def carregar_noticias():
    # Lê todos os arquivos .json da pasta database/noticias/ e transforma
    # cada item em um dicionário simples com os campos relevantes.
    # @st.cache_data garante que isso roda só uma vez por sessão.
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
            continue  # arquivo corrompido: ignora e segue

        # Aceita tanto um único objeto quanto uma lista de objetos
        if isinstance(items, dict):
            items = [items]

        for item in items:
            # Suporta campos em português (titulo/texto) e em inglês (title/content)
            titulo = str(item.get("titulo", item.get("title", ""))).strip()
            texto  = str(item.get("texto",  item.get("content", item.get("text", "")))).strip()
            url    = str(item.get("url", "")).strip()
            data   = str(item.get("data",   item.get("date", ""))).strip()

            if not titulo or not texto:
                continue  # descarta itens sem conteúdo útil

            noticias.append({
                "titulo": titulo,
                "texto":  texto,
                "url":    url,
                "data":   data,
            })

    return noticias


@st.cache_data(show_spinner=False)
def carregar_lattes():
    # Lê todos os arquivos .json da pasta database/lattes/.
    # O formato esperado é o gerado pelo lattes.py: uma lista de dicts,
    # onde cada dict é o currículo completo de um pesquisador.
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

        # Um arquivo pode conter uma lista de pesquisadores ou um único dict
        if isinstance(data, list):
            pesquisadores.extend(data)
        elif isinstance(data, dict):
            pesquisadores.append(data)

    return pesquisadores


# =============================================================================
# TESTE DE CARREGAMENTO (remover depois)
# =============================================================================

noticias    = carregar_noticias()
lattes_data = carregar_lattes()

st.write(f"Notícias carregadas: **{len(noticias)}**")
st.write(f"Pesquisadores carregados: **{len(lattes_data)}**")
