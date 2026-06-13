import os
import json
import re
import unicodedata

import streamlit as st
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

st.set_page_config(page_title="UnB Research Agent", layout="wide")
st.title("🎓 UnB Research Agent")
st.caption("Agente com conhecimento sobre notícias e pesquisadores da Universidade de Brasília")

client = Groq(api_key=st.secrets["GROQ_API_KEY"])
MODEL = "llama-3.3-70b-versatile"


# ── utilitários ──────────────────────────────────────────────

def normalizar(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().strip()


def limparJson(raw):
    raw = re.sub(r"```[a-z]*", "", raw).strip().strip("`")
    return raw


# ── carregamento de dados ────────────────────────────────────

@st.cache_data(show_spinner=False)
def carregar_noticias():
    base_dir = os.path.join(os.path.dirname(__file__), "database", "noticias")
    documents = []

    if not os.path.isdir(base_dir):
        st.warning(f"Pasta '{base_dir}' não encontrada.")
        return documents

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
            texto  = str(item.get("texto",  item.get("content", item.get("text", "")))).strip()
            url    = str(item.get("url", "")).strip()
            data   = str(item.get("data",   item.get("date", ""))).strip()
            text   = f"Título: {titulo}\nData: {data}\n\n{texto}".strip()

            if len(text) < 80:
                continue

            documents.append(Document(
                page_content=text,
                metadata={"source": "noticia", "titulo": titulo, "url": url, "data": data},
            ))

    return documents


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


# ── extração de campos do Lattes ─────────────────────────────

def getAreas(p):
    areas_raw = p.get("areas_atuacao", p.get("areas_pesquisa", p.get("areas", [])))
    if not areas_raw:
        return ""

    partes = []
    for a in areas_raw:
        if isinstance(a, dict):
            sub  = (a.get("subarea")       or "").strip().rstrip(".")
            area = (a.get("area")          or "").strip().rstrip(".")
            esp  = (a.get("especialidade") or "").strip().rstrip(".")
            label = sub or area
            if esp:
                label = f"{label} ({esp})" if label else esp
            if label:
                partes.append(label)
        elif isinstance(a, str):
            partes.append(a.strip())

    seen = set()
    unique = []
    for x in partes:
        if x and x not in seen:
            seen.add(x)
            unique.append(x)

    return ", ".join(unique)


def getFormacao(p):
    todos = p.get("formacao", []) + p.get("pos_doutorado", [])
    if not todos:
        return ""
    linhas = []
    for f in todos:
        if isinstance(f, dict):
            nivel  = f.get("nivel", "")
            inst   = f.get("instituicao", "")
            period = f.get("periodo", "")
            titulo = f.get("titulo", "")
            linha  = f"{nivel} — {inst} ({period})"
            if titulo:
                linha += f': "{titulo[:80]}"'
            linhas.append(linha)
        elif isinstance(f, str):
            linhas.append(f)
    return "\n".join(linhas)


def get_projetos(p):
    projetos = (
        p.get("projetos_pesquisa", [])
        + p.get("projetos_extensao", [])
        + p.get("projetos_desenvolvimento", [])
    )
    if not projetos:
        return ""
    linhas = []
    for proj in projetos[:15]:
        if isinstance(proj, dict):
            titulo   = (proj.get("titulo")    or "").strip()[:120]
            desc     = (proj.get("descricao") or "").strip()[:200]
            situacao = (proj.get("situacao")  or "").strip()
            periodo  = (proj.get("periodo")   or "").strip()
            linha = f"  • {titulo}"
            if situacao:
                linha += f" [{situacao}]"
            if periodo:
                linha += f" ({periodo})"
            if desc:
                linha += f"\n    {desc}"
            linhas.append(linha)
        elif isinstance(proj, str):
            linhas.append(f"  • {proj}")
    return "\n".join(linhas)


def getPublicacoes(p):
    prod    = p.get("producoes", {})
    artigos = prod.get("artigos_periodicos", [])
    anais   = prod.get("anais_completos", [])

    if not artigos and not anais:
        artigos = p.get("publicacoes", p.get("publications", []))

    linhas = []

    for pub in artigos[:20]:
        if isinstance(pub, dict):
            texto = (pub.get("texto") or "").strip()
            cit_s = pub.get("citacoes_scopus", pub.get("citacoes", ""))
            cit_w = pub.get("citacoes_wos", "")
            linha = texto[:220]
            cits = []
            if cit_w:
                cits.append(f"WoS:{cit_w}")
            if cit_s:
                cits.append(f"Scopus:{cit_s}")
            if cits:
                linha += f" [{', '.join(cits)}]"
            linhas.append(f"  [artigo] {linha}")
        elif isinstance(pub, str):
            linhas.append(f"  [artigo] {pub[:220]}")

    for pub in anais[:15]:
        if isinstance(pub, dict):
            texto = (pub.get("texto") or "").strip()
            linhas.append(f"  [anais]  {texto[:200]}")
        elif isinstance(pub, str):
            linhas.append(f"  [anais]  {pub[:200]}")

    return "\n".join(linhas)


def get_orientacoes(p):
    orient = p.get("orientacoes", {})
    if not orient:
        return ""

    if isinstance(orient, list):
        return f"Total de orientações: {len(orient)}"

    mapa = {
        "dissertacao_de_mestrado":    "Mestrado",
        "tese_de_doutorado":          "Doutorado",
        "trabalho_de_conclusao_de_curso_de_graduacao": "TCC Graduação",
        "iniciacao_cientifica":       "Iniciação Científica",
        "monografia_de_conclusao_de_curso_de_aperfeicoamento/especializacao": "Especialização",
        "supervisao_de_pos-doutorado": "Pós-Doutorado",
    }

    totais = []
    for chave, label in mapa.items():
        itens = orient.get(chave, [])
        if itens:
            totais.append(f"{label}: {len(itens)}")

    return "Orientações — " + ", ".join(totais) if totais else ""


def getPremios(p):
    premios = p.get("premios", [])
    if not premios:
        return ""
    linhas = []
    for pr in premios[:8]:
        if isinstance(pr, dict):
            ano  = pr.get("ano", "")
            desc = (pr.get("descricao") or "").strip()[:150]
            linhas.append(f"  • {desc} ({ano})" if ano else f"  • {desc}")
        elif isinstance(pr, str):
            linhas.append(f"  • {pr[:150]}")
    return "\n".join(linhas)


def get_stats(p):
    prod  = p.get("producoes", {})
    n_art = len(prod.get("artigos_periodicos", p.get("publicacoes", [])))
    n_ana = len(prod.get("anais_completos", []))
    n_pat = len(p.get("patentes", []))
    orient = p.get("orientacoes", {})
    if isinstance(orient, dict):
        n_ori = sum(len(v) for v in orient.values())
    elif isinstance(orient, list):
        n_ori = len(orient)
    else:
        n_ori = 0
    partes = []
    if n_art: partes.append(f"{n_art} artigos em periódicos")
    if n_ana: partes.append(f"{n_ana} trabalhos em anais")
    if n_ori: partes.append(f"{n_ori} orientações")
    if n_pat: partes.append(f"{n_pat} patentes")
    return ", ".join(partes)


# ── conversão Lattes → documentos para RAG ──────────────────

def lattes_para_documentos(pesquisadores):
    docs = []

    for p in pesquisadores:
        nome      = str(p.get("nome",      p.get("name",    "Desconhecido"))).strip()
        resumo    = str(p.get("resumo",    p.get("bio",     p.get("summary", "")))).strip()
        lattes_id = str(p.get("lattes_id", p.get("id_lattes", p.get("url", "")))).strip()
        lattes_url= str(p.get("lattes_url", "")).strip()
        orcid     = str(p.get("orcid", "")).strip()
        ultima_at = str(p.get("ultima_atualizacao", "")).strip()
        depto     = str(p.get("departamento", p.get("department", "CIC/UnB"))).strip()

        areas_txt    = getAreas(p)
        formacao_txt = getFormacao(p)
        projetos_txt = get_projetos(p)
        pubs_txt     = getPublicacoes(p)
        orient_txt   = get_orientacoes(p)
        premios_txt  = getPremios(p)
        stats_txt    = get_stats(p)

        secoes = [f"Pesquisador: {nome}", f"Departamento: {depto}"]
        if areas_txt:    secoes.append(f"Áreas de pesquisa: {areas_txt}")
        if resumo:       secoes.append(f"Resumo: {resumo[:600]}")
        if formacao_txt: secoes.append(f"Formação acadêmica:\n{formacao_txt}")
        if stats_txt:    secoes.append(f"Produção científica: {stats_txt}")
        if projetos_txt: secoes.append(f"Projetos de pesquisa:\n{projetos_txt}")
        if pubs_txt:     secoes.append(f"Publicações selecionadas:\n{pubs_txt}")
        if orient_txt:   secoes.append(orient_txt)
        if premios_txt:  secoes.append(f"Prêmios e reconhecimentos:\n{premios_txt}")
        if lattes_url:   secoes.append(f"Lattes: {lattes_url}")
        if orcid:        secoes.append(f"ORCID: {orcid}")
        if ultima_at:    secoes.append(f"Currículo atualizado em: {ultima_at}")

        text = "\n\n".join(secoes).strip()
        if len(text) < 30:
            continue

        docs.append(Document(
            page_content=text,
            metadata={
                "source":    "lattes",
                "nome":      nome,
                "lattes_id": lattes_id,
                "depto":     depto,
                "areas":     areas_txt,
            },
        ))

    return docs


# ── índice vetorial ──────────────────────────────────────────

@st.cache_resource(show_spinner="Construindo índice vetorial…")
def build_retriever():
    all_docs = carregar_noticias() + lattes_para_documentos(carregar_lattes())

    if not all_docs:
        return None

    chunks = CharacterTextSplitter(
        chunk_size=1200, chunk_overlap=200, separator="\n\n"
    ).split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="alfaneo/bertimbau-base-portuguese-sts",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.from_documents(chunks, embedding=embeddings).as_retriever(
        search_kwargs={"k": 6}
    )


def buscar(pergunta):
    retriever = build_retriever()
    if retriever is None:
        return []
    return retriever.invoke(pergunta)


# ── lookup direto por nome de professor ──────────────────────

def buscarProfessor(pergunta):
    pesquisadores = carregar_lattes()
    pergunta_norm = normalizar(pergunta)
    encontrados = []

    for p in pesquisadores:
        nome_norm = normalizar(p.get("nome", ""))
        partes = [w for w in nome_norm.split() if len(w) > 3]
        if any(w in pergunta_norm for w in partes):
            encontrados.append(p)

    return encontrados


def professorParaContexto(p):
    nome       = p.get("nome", "")
    resumo     = (p.get("resumo") or "")[:600]
    areas_txt  = getAreas(p)
    stats_txt  = get_stats(p)
    proj_txt   = get_projetos(p)
    pubs_txt   = getPublicacoes(p)
    orient_txt = get_orientacoes(p)
    prem_txt   = getPremios(p)
    lattes_url = p.get("lattes_url", "")
    ultima_at  = p.get("ultima_atualizacao", "")

    secoes = [f"Pesquisador: {nome}"]
    if areas_txt:  secoes.append(f"Áreas: {areas_txt}")
    if resumo:     secoes.append(f"Resumo: {resumo}")
    if stats_txt:  secoes.append(f"Produção: {stats_txt}")
    if proj_txt:   secoes.append(f"Projetos:\n{proj_txt}")
    if pubs_txt:   secoes.append(f"Publicações:\n{pubs_txt}")
    if orient_txt: secoes.append(orient_txt)
    if prem_txt:   secoes.append(f"Prêmios:\n{prem_txt}")
    if lattes_url: secoes.append(f"Lattes: {lattes_url}")
    if ultima_at:  secoes.append(f"Atualizado em: {ultima_at}")

    return "\n\n".join(secoes)


# ── detecção de intenção ─────────────────────────────────────

def detectarIntencao(pergunta):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the user's question into ONE of these intents:\n"
                    "  news       → about UnB news, events, announcements, institutional facts\n"
                    "  researcher → about researchers, professors, publications, Lattes, areas of study\n"
                    "  both       → needs both news and researcher knowledge\n"
                    "  general    → general conversation, greetings, unrelated topics\n\n"
                    "Reply ONLY with pure JSON, no markdown.\n"
                    'Format: {"intent": "news"|"researcher"|"both"|"general", "reason": "<one sentence>"}'
                ),
            },
            {"role": "user", "content": pergunta},
        ],
        temperature=0.0,
        max_tokens=80,
    )
    try:
        return json.loads(limparJson(resp.choices[0].message.content))
    except Exception:
        return {"intent": "both", "reason": "fallback"}


# ── pipeline de resposta ─────────────────────────────────────

def responder(pergunta, historico):
    intent = detectarIntencao(pergunta).get("intent", "both")

    context_str = ""

    if intent != "general":
        docs = buscar(pergunta)

        if intent == "news":
            docs = [d for d in docs if d.metadata.get("source") == "noticia"] or docs
        elif intent == "researcher":
            docs = [d for d in docs if d.metadata.get("source") == "lattes"] or docs

        for doc in docs:
            if len(context_str) + len(doc.page_content) > 5000:
                break
            src = "📰 Notícia" if doc.metadata.get("source") == "noticia" else "🎓 Pesquisador"
            context_str += f"[{src}]\n{doc.page_content}\n\n---\n\n"

        if intent in ("researcher", "both"):
            profs_diretos = buscarProfessor(pergunta)
            nomes_rag = {
                normalizar(doc.metadata.get("nome", ""))
                for doc in docs
                if doc.metadata.get("source") == "lattes"
            }
            for prof in profs_diretos:
                if normalizar(prof.get("nome", "")) not in nomes_rag:
                    ctx = professorParaContexto(prof)
                    if len(context_str) + len(ctx) < 7000:
                        context_str += f"[🎓 Pesquisador]\n{ctx}\n\n---\n\n"

    messages = [
        {
            "role": "system",
            "content": (
                "Você é um assistente especializado na Universidade de Brasília (UnB). "
                "Responda em português brasileiro, de forma clara e objetiva.\n\n"
                "Você tem acesso a duas bases de conhecimento:\n"
                "  1. Notícias institucionais da UnB\n"
                "  2. Perfis de pesquisadores com currículos Lattes\n\n"
                "Regras:\n"
                "- Use o contexto fornecido; não invente informações.\n"
                "- Para perguntas sobre pesquisadores: cite nome, áreas, projetos em andamento "
                "e publicações relevantes quando disponíveis.\n"
                "- Para perguntas sobre áreas de pesquisa: liste os professores da área com "
                "uma breve descrição de cada um.\n"
                "- Para perguntas sobre notícias: cite título e data quando disponíveis.\n"
                "- Se nada relevante for encontrado, diga brevemente e ofereça ajuda alternativa.\n"
                "- Nunca diga 'de acordo com o contexto' — responda diretamente."
            ),
        }
    ]

    for turn in historico[-12:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    user_content = pergunta
    if context_str:
        user_content = f"Contexto:\n{context_str}\n\nPergunta: {pergunta}"

    messages.append({"role": "user", "content": user_content})

    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=800,
    )
    return resp.choices[0].message.content.strip()


# ── interface Streamlit ──────────────────────────────────────

if st.button("🧹 Limpar conversa"):
    st.session_state["history"] = []
    st.rerun()

if "history" not in st.session_state:
    st.session_state["history"] = []

for msg in st.session_state["history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Pergunte sobre notícias ou pesquisadores da UnB…"):
    st.session_state["history"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando bases de conhecimento…"):
            resposta = responder(prompt, st.session_state["history"][:-1])
        st.markdown(resposta)

    st.session_state["history"].append({"role": "assistant", "content": resposta})
    st.rerun()
