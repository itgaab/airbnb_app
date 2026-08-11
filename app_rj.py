"""
Análise Espacial e Preditiva do Airbnb — Rio de Janeiro

App Streamlit interativo que cruza dados geográficos, temporais e de avaliação
dos anúncios para apoiar decisões de hóspedes e investidores. Organizado em
seis abas:
  1. Mapa Dinâmico — luxo, rentabilidade, ocupação e turismo por bairro
  2. Evolução Temporal dos Preços — sazonalidade e histórico entre coletas
  3. Simulador de Investimento — previsão de preço, ocupação e rentabilidade
  4. Análise de Avaliações — pontos fortes e fracos por bairro e anúncio
  5. Recomendação por Turismo — melhores anúncios perto de pontos de interesse
  6. Assistente IA — perguntas em linguagem natural sobre os dados
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium import FeatureGroup, LayerControl
from branca.colormap import linear
from streamlit_folium import st_folium
from scipy.spatial import cKDTree

# --- Bibliotecas de análise fatorial e modelagem (opcionais, com fallback) -------
try:
    from factor_analyzer import FactorAnalyzer
    from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
    FACTOR_ANALYZER_DISPONIVEL = True
except ImportError:
    FACTOR_ANALYZER_DISPONIVEL = False

try:
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import RidgeCV, LassoCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.cluster import KMeans
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    MODELAGEM_DISPONIVEL = True
except ImportError:
    MODELAGEM_DISPONIVEL = False

try:
    from scipy import stats as scipy_stats
    SCIPY_STATS_DISPONIVEL = True
except ImportError:
    SCIPY_STATS_DISPONIVEL = False

try:
    import joblib
    JOBLIB_DISPONIVEL = True
except ImportError:
    JOBLIB_DISPONIVEL = False

try:
    import anthropic
    ANTHROPIC_DISPONIVEL = True
except ImportError:
    ANTHROPIC_DISPONIVEL = False

st.set_page_config(
    page_title="Airbnb Rio de Janeiro — Análise Espacial",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Estilo — layout mais elegante (tipografia, espaçamento, abas e cartões)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2.5rem;
            max-width: 1300px;
        }

        /* Cabeçalho — título + botão de informação alinhados */
        .cabecalho-app h1 {
            font-size: 2.05rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin-bottom: 0.15rem;
            color: var(--text-color);
        }
        .cabecalho-app p {
            color: var(--text-color);
            opacity: 0.65;
            font-size: 0.95rem;
            margin-top: 0;
        }

        /* Botão de informação compacto — empurrado pra baixo do menu nativo do
           Streamlit (Share/⋮) e com cores que se adaptam ao tema claro/escuro */
        div[data-testid="stPopover"] {
            margin-top: 0.35rem;
        }
        div[data-testid="stPopover"] button {
            border-radius: 999px;
            padding: 0.25rem 0.7rem;
            font-size: 0.8rem;
            color: var(--text-color);
            border: 1px solid rgba(128, 128, 128, 0.35);
            background-color: var(--secondary-background-color);
        }
        div[data-testid="stPopover"] button:hover {
            border-color: var(--primary-color);
            color: var(--primary-color);
        }

        /* Abas — mais espaçadas e com destaque suave na aba ativa */
        button[data-baseweb="tab"] {
            font-size: 0.95rem;
            font-weight: 600;
            padding: 0.5rem 1rem;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: var(--primary-color);
        }
        div[data-baseweb="tab-highlight"] {
            background-color: var(--primary-color);
        }
        div[data-baseweb="tab-border"] {
            background-color: rgba(128, 128, 128, 0.25);
        }

        /* Métricas e cartões com leve sombra — adaptados ao tema */
        div[data-testid="stMetric"] {
            background-color: var(--secondary-background-color);
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 10px;
            padding: 0.8rem 1rem;
        }

        hr {
            margin: 1.6rem 0;
            border-color: rgba(128, 128, 128, 0.25);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

PASTA_DADOS = os.path.join(os.path.dirname(__file__), "dados")
NOMES_MES = {1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun",
             7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"}


def capitalizar_primeira(texto):
    """Deixa a primeira letra maiúscula sem mexer no resto do texto (usado como
    format_func em campos cujas opções vêm cruas do dataset, ex.: tipo_quarto)."""
    if isinstance(texto, str) and texto:
        return texto[0].upper() + texto[1:]
    return texto


# ---------------------------------------------------------------------------
# Carregamento e preparação dos dados (cacheado — só roda 1x por sessão)
# ---------------------------------------------------------------------------
@st.cache_data
def carregar_dados():
    df = pd.read_parquet(os.path.join(PASTA_DADOS, "dataset_features.parquet"))
    gdf = gpd.read_file(os.path.join(PASTA_DADOS, "bairros_estatistica_espacial.geojson"))
    # o geojson já traz um preco_medio (calculado na análise LISA); descartamos aqui
    # para que o preco_medio recalculado em agregar_por_bairro() prevaleça sem sufixo _x/_y
    gdf = gdf.drop(columns=["preco_medio"], errors="ignore")
    try:
        calendario = pd.read_parquet(os.path.join(PASTA_DADOS, "calendar_agregado.parquet"))
    except FileNotFoundError:
        calendario = None
    try:
        temporal = pd.read_parquet(os.path.join(PASTA_DADOS, "listings_temporal.parquet"))
    except FileNotFoundError:
        temporal = None

    # Traz nome e link do anúncio a partir do listings.csv/parquet bruto (Inside Airbnb), se o
    # arquivo estiver na pasta de dados. dataset_features.parquet não carrega essas colunas.
    # Preferimos o parquet (listings_nome_link.parquet), bem mais leve pro repositório;
    # se não existir, caímos de volta pro CSV completo.
    bruto = None
    caminho_parquet_nome_link = os.path.join(PASTA_DADOS, "listings_nome_link.parquet")
    caminho_csv_nome_link = os.path.join(PASTA_DADOS, "listings.csv")
    try:
        if os.path.exists(caminho_parquet_nome_link):
            bruto = pd.read_parquet(caminho_parquet_nome_link, columns=["id", "name", "listing_url"])
        else:
            bruto = pd.read_csv(caminho_csv_nome_link, usecols=["id", "name", "listing_url"])
    except (FileNotFoundError, ValueError, KeyError):
        bruto = None

    if bruto is not None:
        bruto = bruto.rename(columns={"id": "id_anuncio"})
        bruto["id_anuncio"] = bruto["id_anuncio"].astype(str)
        df["id_anuncio"] = df["id_anuncio"].astype(str)
        df = df.merge(bruto, on="id_anuncio", how="left")

    return df, gdf, calendario, temporal


@st.cache_data
def agregar_por_bairro(df):
    agg = df.groupby("bairro_padronizado").agg(
        qtd_anuncios=("id_anuncio", "count"),
        score_luxo_medio=("score_luxo", "mean"),
        faixa_preco_moda=("faixa_preco", lambda s: s.mode().iat[0] if not s.mode().empty else None),
        preco_medio=("preco", "mean"),
        rentabilidade_media=("rentabilidade_diaria", "mean"),
        taxa_ocupacao_media=("taxa_ocupacao_estimada", "mean"),
        demanda_media=("demanda_bairro", "mean"),
        regiao_popular=("regiao_popular", lambda s: s.mode().iat[0] if not s.mode().empty else 0),
        perfil_anfitriao_predominante=(
            "perfil_anfitriao", lambda s: s.mode().iat[0] if not s.mode().empty else None
        ),
        nota_composta_media=("nota_composta", "mean"),
        lat=("latitude", "mean"),
        lon=("longitude", "mean"),
    ).reset_index()
    return agg


@st.cache_data
def montar_contexto_dados(df, agg):
    """Resume o dataset carregado em texto, para dar contexto ao assistente de IA
    sem precisar mandar o dataframe inteiro (que estouraria o limite de tokens)."""
    top_caros = agg.nlargest(5, "preco_medio")[["bairro_padronizado", "preco_medio"]]
    top_baratos = agg.nsmallest(5, "preco_medio")[["bairro_padronizado", "preco_medio"]]
    top_rentaveis = agg.nlargest(5, "rentabilidade_media")[["bairro_padronizado", "rentabilidade_media"]]

    partes = [
        f"O dataset tem {len(df)} anúncios do Airbnb no Rio de Janeiro, distribuídos em "
        f"{agg.shape[0]} bairros.",
        f"Preço médio geral: R$ {df['preco'].mean():.2f} (mediana R$ {df['preco'].median():.2f}).",
        "5 bairros mais caros (preço médio): " + "; ".join(
            f"{r.bairro_padronizado} (R$ {r.preco_medio:.0f})" for r in top_caros.itertuples()
        ),
        "5 bairros mais baratos (preço médio): " + "; ".join(
            f"{r.bairro_padronizado} (R$ {r.preco_medio:.0f})" for r in top_baratos.itertuples()
        ),
        "5 bairros com maior rentabilidade diária média: " + "; ".join(
            f"{r.bairro_padronizado} (R$ {r.rentabilidade_media:.2f})" for r in top_rentaveis.itertuples()
        ),
    ]
    if "nota_composta" in df.columns:
        partes.append(f"Nota composta média das hospedagens: {df['nota_composta'].mean():.2f}.")
    if "faixa_preco" in df.columns:
        dist = df["faixa_preco"].value_counts(normalize=True).mul(100).round(1)
        partes.append("Distribuição por faixa de preço (%): " + "; ".join(f"{k}: {v}%" for k, v in dist.items()))
    return "\n".join(partes)


@st.cache_data(show_spinner=False)
def buscar_pontos_turisticos():
    """Lista curada com os principais pontos turísticos do Rio de Janeiro
    (coordenadas aproximadas), distribuídos entre Zona Sul, Centro, Zona Norte
    e Zona Oeste/Barra. Fixa, para manter o mapa legível — antes a consulta ao
    OpenStreetMap trazia centenas de pontos concentrados sobretudo na Zona Sul."""
    pontos = pd.DataFrame([
        # --- Zona Sul ---
        {"nome": "Cristo Redentor",                      "lat": -22.9519, "lon": -43.2105},
        {"nome": "Pão de Açúcar",                         "lat": -22.9492, "lon": -43.1545},
        {"nome": "Praia Vermelha",                        "lat": -22.9497, "lon": -43.1631},
        {"nome": "Praia de Copacabana",                   "lat": -22.9711, "lon": -43.1822},
        {"nome": "Forte de Copacabana",                   "lat": -22.9878, "lon": -43.1778},
        {"nome": "Praia do Arpoador",                     "lat": -22.9878, "lon": -43.1936},
        {"nome": "Praia de Ipanema",                      "lat": -22.9838, "lon": -43.2096},
        {"nome": "Praia do Leblon",                       "lat": -22.9847, "lon": -43.2247},
        {"nome": "Jardim Botânico",                       "lat": -22.9675, "lon": -43.2247},
        {"nome": "Parque Lage",                           "lat": -22.9581, "lon": -43.2145},
        {"nome": "Lagoa Rodrigo de Freitas",              "lat": -22.9722, "lon": -43.2044},
        {"nome": "Mirante Dona Marta",                    "lat": -22.9553, "lon": -43.1975},
        {"nome": "Vista Chinesa",                         "lat": -22.9686, "lon": -43.2394},
        {"nome": "Praia de São Conrado",                  "lat": -22.9997, "lon": -43.2564},
        {"nome": "Pedra da Gávea",                        "lat": -22.9908, "lon": -43.2789},
        # --- Centro / Santa Teresa / Lapa ---
        {"nome": "Santa Teresa (Bondinho)",               "lat": -22.9192, "lon": -43.1867},
        {"nome": "Escadaria Selarón",                     "lat": -22.9147, "lon": -43.1806},
        {"nome": "Arcos da Lapa",                         "lat": -22.9139, "lon": -43.1794},
        {"nome": "Theatro Municipal",                     "lat": -22.9092, "lon": -43.1761},
        {"nome": "Confeitaria Colombo",                   "lat": -22.9053, "lon": -43.1789},
        {"nome": "Museu Nacional de Belas Artes",         "lat": -22.9086, "lon": -43.1747},
        {"nome": "Biblioteca Nacional",                   "lat": -22.9097, "lon": -43.1758},
        {"nome": "Catedral Metropolitana",                "lat": -22.9114, "lon": -43.1802},
        {"nome": "Igreja de São Francisco da Penitência", "lat": -22.9057, "lon": -43.1806},
        {"nome": "Museu do Amanhã",                       "lat": -22.8947, "lon": -43.1806},
        {"nome": "Museu de Arte Moderna (MAM)",           "lat": -22.9147, "lon": -43.1719},
        {"nome": "Museu de Arte do Rio (MAR)",            "lat": -22.8958, "lon": -43.1817},
        {"nome": "AquaRio / Praça Mauá",                  "lat": -22.8944, "lon": -43.1808},
        {"nome": "Cais do Valongo",                       "lat": -22.8944, "lon": -43.1867},
        {"nome": "Ilha Fiscal",                           "lat": -22.8956, "lon": -43.1697},
        # --- Zona Norte ---
        {"nome": "Maracanã",                              "lat": -22.9122, "lon": -43.2302},
        {"nome": "Quinta da Boa Vista",                   "lat": -22.9058, "lon": -43.2244},
        {"nome": "Museu Nacional (Quinta da Boa Vista)",  "lat": -22.9067, "lon": -43.2242},
        {"nome": "Feira de São Cristóvão",                "lat": -22.8975, "lon": -43.2225},
        {"nome": "Sambódromo",                            "lat": -22.9111, "lon": -43.1964},
        {"nome": "Estádio Nilton Santos (Engenhão)",      "lat": -22.8931, "lon": -43.2831},
        {"nome": "Floresta da Tijuca",                    "lat": -22.9556, "lon": -43.2803},
        # --- Zona Oeste / Barra ---
        {"nome": "Praia da Barra da Tijuca",              "lat": -23.0086, "lon": -43.3651},
        {"nome": "Praia do Recreio dos Bandeirantes",     "lat": -23.0197, "lon": -43.4536},
        {"nome": "Parque Olímpico",                       "lat": -22.9769, "lon": -43.3953},
        {"nome": "Cidade das Artes",                      "lat": -22.9942, "lon": -43.3653},
        {"nome": "Pedra Bonita",                          "lat": -22.9875, "lon": -43.2764},
    ])
    return pontos[["nome", "lat", "lon"]].reset_index(drop=True)


def obter_coluna_nome_anuncio(dataframe):
    """Retorna uma Series com o nome/título de cada anúncio para exibição.
    Tenta várias colunas comuns (o nome exato depende de como o
    dataset_features.parquet foi montado); se nenhuma existir, usa o ID."""
    COLUNAS_NOME_ANUNCIO = ["name", "nome_anuncio", "titulo_anuncio", "titulo", "listing_name"]
    coluna_nome_anuncio = next((c for c in COLUNAS_NOME_ANUNCIO if c in dataframe.columns), None)
    if coluna_nome_anuncio:
        return dataframe[coluna_nome_anuncio].fillna("Anúncio sem título")
    elif "id_anuncio" in dataframe.columns:
        return "Anúncio #" + dataframe["id_anuncio"].astype(str)
    return pd.Series("Anúncio", index=dataframe.index)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


@st.cache_data
def adicionar_distancia_turistica(agg, pontos):
    agg = agg.copy()
    if pontos.empty:
        agg["distancia_ponto_turistico_km"] = np.nan
        agg["ponto_turistico_proximo"] = None
        return agg
    arvore = cKDTree(pontos[["lat", "lon"]].values)
    _, idx = arvore.query(agg[["lat", "lon"]].values, k=1)
    agg["ponto_turistico_proximo"] = pontos.loc[idx, "nome"].values
    agg["distancia_ponto_turistico_km"] = [
        haversine_km(r.lat, r.lon, pontos.loc[i, "lat"], pontos.loc[i, "lon"])
        for r, i in zip(agg.itertuples(), idx)
    ]
    return agg


# ---------------------------------------------------------------------------
# Perfil do anfitrião — variável composta e documentada
# Regra (aplicada nesta ordem, a primeira que bater define a categoria):
#   1) Superanfitrião         → e_superanfitriao == "t"
#   2) Profissional (4+)      → total_anuncios_anfitriao >= 4
#   3) Recorrente (2-3)       → total_anuncios_anfitriao in [2, 3]
#   4) Iniciante (1 imóvel)   → total_anuncios_anfitriao == 1
# A regra prioriza o selo de superanfitrião porque ele já é um critério oficial
# do Airbnb (nota alta + baixa taxa de cancelamento + resposta rápida), então um
# superanfitrião com poucos imóveis ainda é classificado como "Superanfitrião".
# ---------------------------------------------------------------------------
def _classificar_perfil_anfitriao(row):
    if row.get("e_superanfitriao") == "t":
        return "Superanfitrião"
    total = row.get("total_anuncios_anfitriao")
    if pd.isna(total):
        return "Não classificado"
    if total >= 4:
        return "Profissional (4+ imóveis)"
    if total >= 2:
        return "Recorrente (2-3 imóveis)"
    return "Iniciante (1 imóvel)"


@st.cache_data
def criar_perfil_anfitriao(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["perfil_anfitriao"] = df.apply(_classificar_perfil_anfitriao, axis=1)
    return df


# ---------------------------------------------------------------------------
# Variáveis derivadas de notas e avaliações
# - nivel_atividade_avaliacoes: classifica o anúncio pela frequência de avaliações
#   recebidas (avaliacoes_por_mes), em tercis calculados sobre os anúncios que
#   já têm ao menos 1 avaliação (evita distorcer os tercis com zeros).
# - qualidade_percebida: faixas de nota_composta inspiradas no próprio critério
#   de superanfitrião do Airbnb (nota mínima 4.8) — não são tercis arbitrários.
# - indice_engajamento: nota_composta × avaliacoes_por_mes, resume num só número
#   "quão bem avaliado e quão frequentemente avaliado" — útil pra rankear
#   anúncios que são bons E populares, não só um ou outro.
# ---------------------------------------------------------------------------
@st.cache_data
def criar_variaveis_avaliacoes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "avaliacoes_por_mes" in df.columns:
        com_avaliacao = df.loc[df["avaliacoes_por_mes"] > 0, "avaliacoes_por_mes"]
        if not com_avaliacao.empty:
            t1, t2 = com_avaliacao.quantile([0.33, 0.66]).values
        else:
            t1 = t2 = 0

        def _nivel_atividade(v):
            if pd.isna(v) or v == 0:
                return "Sem avaliações recentes"
            if v <= t1:
                return "Baixa atividade"
            if v <= t2:
                return "Atividade moderada"
            return "Alta atividade"

        df["nivel_atividade_avaliacoes"] = df["avaliacoes_por_mes"].apply(_nivel_atividade)

    if "nota_composta" in df.columns:
        def _qualidade(v):
            if pd.isna(v):
                return "Sem nota"
            if v < 4.0:
                return "Atenção (< 4.0)"
            if v < 4.8:
                return "Boa (4.0 – 4.7)"
            return "Excelente (≥ 4.8)"

        df["qualidade_percebida"] = df["nota_composta"].apply(_qualidade)

    if {"nota_composta", "avaliacoes_por_mes"}.issubset(df.columns):
        df["indice_engajamento"] = (df["nota_composta"] * df["avaliacoes_por_mes"]).round(3)

    return df


# ---------------------------------------------------------------------------
# Faixa de preço com controle de outliers
# Em vez de deixar diárias extremas (ex.: > R$ 500 mil, provavelmente anúncios
# "trollados" ou com erro de cadastro) esticarem a escala de cor do mapa e os
# filtros, aqui:
#   1) calculamos um teto no percentil 99 (preco_teto_outlier) — tudo acima
#      disso vira uma faixa própria "Outlier (fora da curva)";
#   2) as demais faixas são cortadas em quartis do preço "normal" (≤ p99).
# Isso não remove os outliers do dataset (eles continuam nos gráficos que já
# lidam bem com isso, como boxplot/histograma), só evita que eles dominem
# escalas de cor e filtros de slider.
# ---------------------------------------------------------------------------
@st.cache_data
def criar_faixa_preco_controlada(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "preco" not in df.columns:
        return df

    teto_outlier = df["preco"].quantile(0.99)
    normal = df.loc[df["preco"] <= teto_outlier, "preco"]
    q1, q2, q3 = normal.quantile([0.25, 0.5, 0.75]).values

    def _faixa(v):
        if pd.isna(v):
            return None
        if v > teto_outlier:
            return "Outlier (fora da curva)"
        if v <= q1:
            return "Econômico"
        if v <= q2:
            return "Médio"
        if v <= q3:
            return "Alto"
        return "Premium"

    df["faixa_preco_controlada"] = df["preco"].apply(_faixa)
    df["preco_teto_outlier"] = teto_outlier
    return df


# ---------------------------------------------------------------------------
# Análise de sensibilidade do teto de outlier
# Recalcula a faixa de preço trocando o teto (p95 / p99 / p99.5) para checar se
# a distribuição de anúncios por faixa muda muito conforme o corte escolhido.
# ---------------------------------------------------------------------------
@st.cache_data
def teste_sensibilidade_faixa_preco(df: pd.DataFrame, tetos=(0.95, 0.99, 0.995)) -> pd.DataFrame:
    if "preco" not in df.columns:
        return pd.DataFrame()

    linhas = []
    for p in tetos:
        teto = df["preco"].quantile(p)
        normal = df.loc[df["preco"] <= teto, "preco"]
        q1, q2, q3 = normal.quantile([0.25, 0.5, 0.75]).values

        def _faixa_local(v, teto=teto, q1=q1, q2=q2, q3=q3):
            if pd.isna(v):
                return None
            if v > teto:
                return "Outlier"
            if v <= q1:
                return "Econômico"
            if v <= q2:
                return "Médio"
            if v <= q3:
                return "Alto"
            return "Premium"

        contagem = df["preco"].apply(_faixa_local).value_counts()
        linha = {"teto_percentil": f"p{p * 100:g}", "valor_teto_reais": round(float(teto), 2)}
        for faixa in ["Econômico", "Médio", "Alto", "Premium", "Outlier"]:
            linha[faixa] = int(contagem.get(faixa, 0))
        linhas.append(linha)
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# Teste de associação: qualidade_percebida x preço / ocupação
# ANOVA (variável numérica contínua) mostra se a média de preço/ocupação difere
# entre as faixas de qualidade percebida — usado para defender que a
# categorização captura sinal real, não só reduz a variável.
# ---------------------------------------------------------------------------
@st.cache_data
def teste_associacao_qualidade(df: pd.DataFrame) -> dict:
    resultado = {}
    if not SCIPY_STATS_DISPONIVEL or "qualidade_percebida" not in df.columns:
        return resultado
    for alvo in ["preco", "taxa_ocupacao_estimada"]:
        if alvo not in df.columns:
            continue
        grupos = [
            g[alvo].dropna().values
            for _, g in df.dropna(subset=[alvo, "qualidade_percebida"]).groupby("qualidade_percebida")
        ]
        grupos = [g for g in grupos if len(g) > 1]
        if len(grupos) < 2:
            continue
        f_stat, p_valor = scipy_stats.f_oneway(*grupos)
        resultado[alvo] = {"f_stat": float(f_stat), "p_valor": float(p_valor)}
    return resultado


# ---------------------------------------------------------------------------
# Dispersão das notas por bairro
# Complementa a média: dois bairros com nota média igual podem ter consistência
# bem diferente (desvio-padrão baixo = experiência previsível; alto = "loteria").
# ---------------------------------------------------------------------------
@st.cache_data
def dispersao_notas_bairro(df: pd.DataFrame, n_min: int = 5) -> pd.DataFrame:
    if not {"bairro_padronizado", "nota_composta", "id_anuncio"}.issubset(df.columns):
        return pd.DataFrame()
    agg = df.groupby("bairro_padronizado").agg(
        nota_media=("nota_composta", "mean"),
        nota_desvio_padrao=("nota_composta", "std"),
        qtd_anuncios=("id_anuncio", "count"),
    ).dropna()
    return agg[agg["qtd_anuncios"] >= n_min].sort_values("nota_desvio_padrao")


# ---------------------------------------------------------------------------
# Validação quantitativa do "perfil do anfitrião" manual via KMeans
# Roda um KMeans (mesmo nº de clusters que categorias manuais, ignorando
# "Não classificado") sobre variáveis do anfitrião e compara com a regra
# hierárquica manual via tabela cruzada — se os clusters "batem" bastante com
# as categorias manuais, é evidência de que a regra captura uma estrutura real
# nos dados, não é arbitrária.
# ---------------------------------------------------------------------------
@st.cache_data
def comparar_perfil_kmeans(df: pd.DataFrame, n_clusters: int = 4) -> dict:
    if not MODELAGEM_DISPONIVEL or "perfil_anfitriao" not in df.columns:
        return {}

    variaveis = [c for c in [
        "total_anuncios_anfitriao", "nota_composta", "numero_avaliacoes",
        "avaliacoes_por_mes", "e_superanfitriao",
    ] if c in df.columns]
    if not variaveis:
        return {}

    base = df.dropna(subset=variaveis).copy()
    if base.empty:
        return {}
    if "e_superanfitriao" in base.columns:
        base["e_superanfitriao"] = (base["e_superanfitriao"] == "t").astype(int)

    X = StandardScaler().fit_transform(base[variaveis])
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    base["cluster_kmeans"] = km.fit_predict(X)

    pct_nao_classificado = float((df["perfil_anfitriao"] == "Não classificado").mean())

    tabela_cruzada = pd.crosstab(base["perfil_anfitriao"], base["cluster_kmeans"])
    # concordância: para cada categoria manual, % de anúncios no cluster majoritário dela
    concordancia = (tabela_cruzada.max(axis=1) / tabela_cruzada.sum(axis=1)).mean()

    return {
        "tabela_cruzada": tabela_cruzada,
        "concordancia_media": float(concordancia),
        "pct_nao_classificado": pct_nao_classificado,
        "variaveis_usadas": variaveis,
    }


# ---------------------------------------------------------------------------
# Modelos preditivos (Simulador de Investimento)
# Gerados pelo notebook 05-1_modelagem_preditiva.ipynb (pasta outputs/modelos/)
# ---------------------------------------------------------------------------
PASTA_MODELOS = os.path.join(os.path.dirname(__file__), "modelos")


@st.cache_resource
def carregar_modelos():
    if not JOBLIB_DISPONIVEL:
        return None
    caminho_preco = os.path.join(PASTA_MODELOS, "modelo_preco.joblib")
    caminho_ocup = os.path.join(PASTA_MODELOS, "modelo_ocupacao.joblib")
    caminho_config = os.path.join(PASTA_MODELOS, "config_preditores.joblib")
    if not (os.path.exists(caminho_preco) and os.path.exists(caminho_ocup) and os.path.exists(caminho_config)):
        return None
    config = joblib.load(caminho_config)
    return {
        "modelo_preco": joblib.load(caminho_preco),
        "modelo_ocupacao": joblib.load(caminho_ocup),
        **config,  # preditores_num_preco, preditores_cat_preco, preditores_num_ocup,
                   # preditores_cat_ocup, multiplicador_sazonal
    }


def prever_investimento(modelos: dict, dados_imovel: dict, mes_referencia: str = None) -> dict:
    """Réplica da função definida em 05-1_modelagem_preditiva.ipynb (célula 'Composição
    da rentabilidade estimada'). Recebe as características do imóvel e devolve preço
    de diária previsto, taxa de ocupação prevista e rentabilidade anual estimada, além
    da margem de erro dos modelos (se o `config_preditores.joblib` já tiver essas métricas —
    versões antigas do arquivo não têm, e o app cai de volta para "sem margem" nesse caso).
    """
    preditores_preco = modelos["preditores_num_preco"] + modelos["preditores_cat_preco"]
    preditores_ocup = modelos["preditores_num_ocup"] + modelos["preditores_cat_ocup"]

    linha_preco = pd.DataFrame([{k: dados_imovel.get(k) for k in preditores_preco}])
    linha_ocup = pd.DataFrame([{k: dados_imovel.get(k) for k in preditores_ocup}])

    preco_previsto = float(np.expm1(modelos["modelo_preco"].predict(linha_preco)[0]))
    ocupacao_prevista = float(modelos["modelo_ocupacao"].predict(linha_ocup)[0])
    ocupacao_prevista = min(max(ocupacao_prevista, 0.0), 1.0)  # limita entre 0% e 100%

    rentabilidade_anual = preco_previsto * ocupacao_prevista * 365
    fator_sazonal = 1.0
    multiplicador_sazonal = modelos.get("multiplicador_sazonal", {})
    if mes_referencia and mes_referencia in multiplicador_sazonal:
        fator_sazonal = multiplicador_sazonal[mes_referencia]
        rentabilidade_anual *= fator_sazonal

    resultado = {
        "preco_diaria_previsto": round(preco_previsto, 2),
        "taxa_ocupacao_prevista": round(ocupacao_prevista, 4),
        "noites_ocupadas_ano_estimadas": round(ocupacao_prevista * 365, 1),
        "rentabilidade_anual_estimada": round(rentabilidade_anual, 2),
        "fator_sazonal_aplicado": round(fator_sazonal, 3),
        "tem_margem_erro": False,
    }

    erro_preco = modelos.get("erro_preco_reais")
    erro_ocup = modelos.get("erro_ocupacao")
    if erro_preco is not None and erro_ocup is not None:
        preco_min = max(preco_previsto - erro_preco, 0.0)
        preco_max = preco_previsto + erro_preco
        ocup_min = min(max(ocupacao_prevista - erro_ocup, 0.0), 1.0)
        ocup_max = min(max(ocupacao_prevista + erro_ocup, 0.0), 1.0)
        rent_min = preco_min * ocup_min * 365 * fator_sazonal
        rent_max = preco_max * ocup_max * 365 * fator_sazonal
        resultado.update({
            "tem_margem_erro": True,
            "preco_erro_abs": round(erro_preco, 2),
            "ocupacao_erro_abs": round(erro_ocup, 4),
            "preco_intervalo": (round(preco_min, 2), round(preco_max, 2)),
            "ocupacao_intervalo": (round(ocup_min, 4), round(ocup_max, 4)),
            "rentabilidade_intervalo": (round(rent_min, 2), round(rent_max, 2)),
            "r2_preco": modelos.get("r2_preco"),
            "r2_ocupacao": modelos.get("r2_ocupacao"),
        })
    return resultado





# ---------------------------------------------------------------------------
# Perfis predefinidos do simulador (Econômico / Padrão / Luxo)
# Calculados a partir dos dados reais (tercis de `score_luxo`), não são valores
# "chutados" — são a mediana/moda dos imóveis que caem em cada faixa de luxo.
# ---------------------------------------------------------------------------
CAMPOS_NUM_PERFIL = [
    "capacidade_hospedes", "banheiros", "quartos", "camas", "qtd_comodidades",
    "nota_composta", "noites_minimas", "noites_maximas", "total_anuncios_anfitriao",
    "numero_avaliacoes", "avaliacoes_por_mes",
]
CAMPOS_CAT_PERFIL = [
    "tipo_quarto", "tipo_propriedade", "tipo_hospedagem", "e_superanfitriao", "reserva_instantanea",
]
CAMPOS_BINARIOS_PERFIL = ["tem_banheiro_privativo", "flexibilidade_estadia"]
# Campos cujo widget no formulário é int (min_value/max_value inteiros) — a mediana
# precisa ser arredondada e convertida, senão o Streamlit reclama de tipo (int vs float).
# `quartos`/`camas` entraram aqui porque não faz sentido ter "2,5 quartos" — só
# `banheiros` fica de fora, pois meio-banheiro (lavabo, sem chuveiro) é uma categoria
# real nos dados do Airbnb.
CAMPOS_INT_PERFIL = {
    "capacidade_hospedes", "quartos", "camas", "qtd_comodidades", "noites_minimas",
    "noites_maximas", "total_anuncios_anfitriao", "numero_avaliacoes",
}


@st.cache_data
def calcular_perfis_predefinidos(df: pd.DataFrame) -> dict:
    if "score_luxo" not in df.columns:
        return {}

    tercil_1, tercil_2 = df["score_luxo"].quantile([0.33, 0.66]).values
    faixas = {
        "Econômico": df["score_luxo"] <= tercil_1,
        "Padrão": (df["score_luxo"] > tercil_1) & (df["score_luxo"] <= tercil_2),
        "Luxo": df["score_luxo"] > tercil_2,
    }

    perfis = {}
    for nome_perfil, mascara in faixas.items():
        sub = df[mascara]
        if sub.empty:
            continue
        perfil = {}
        for c in CAMPOS_NUM_PERFIL:
            if c in sub.columns:
                mediana = sub[c].median()
                perfil[c] = int(round(mediana)) if c in CAMPOS_INT_PERFIL else float(mediana)
        for c in CAMPOS_CAT_PERFIL:
            if c in sub.columns and not sub[c].mode().empty:
                perfil[c] = sub[c].mode().iat[0]
        for c in CAMPOS_BINARIOS_PERFIL:
            if c in sub.columns:
                perfil[c] = int(round(sub[c].median()))
        perfis[nome_perfil] = perfil
    return perfis


# ---------------------------------------------------------------------------
# Multicolinearidade (VIF) das variáveis numéricas do modelo de preço
# score_luxo, nota_composta e várias amenities tendem a ser correlacionadas —
# VIF alto (regra prática: > 5 ou > 10) indica que a variável carrega
# informação redundante com as demais, o que infla o erro-padrão dos
# coeficientes (mais relevante para modelos lineares/Ridge/Lasso que para RF).
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def calcular_vif_preco(df: pd.DataFrame, _modelos: dict) -> pd.DataFrame:
    if not MODELAGEM_DISPONIVEL:
        return pd.DataFrame()
    preditores_num = _modelos.get("preditores_num_preco", [])
    preditores_num = [c for c in preditores_num if c in df.columns]
    if len(preditores_num) < 2:
        return pd.DataFrame()

    base = df[preditores_num].dropna()
    if base.shape[0] < len(preditores_num) + 1:
        return pd.DataFrame()

    X = sm.add_constant(base)
    linhas = []
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        try:
            vif = variance_inflation_factor(X.values, i)
        except Exception:
            vif = np.nan
        linhas.append({"variavel": col, "vif": round(float(vif), 2)})
    return pd.DataFrame(linhas).sort_values("vif", ascending=False)


# ---------------------------------------------------------------------------
# Baseline simples (média de preço por bairro) para comparar com o modelo real
# Mostra o ganho de usar Ridge/Lasso/RF em vez de simplesmente prever a média
# histórica do bairro — argumento direto de "o modelo agrega valor" pra banca.
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def comparar_modelo_com_baseline(df: pd.DataFrame, _modelos: dict) -> dict:
    if not MODELAGEM_DISPONIVEL or "bairro_padronizado" not in df.columns or "preco" not in df.columns:
        return {}
    preditores_preco = _modelos.get("preditores_num_preco", []) + _modelos.get("preditores_cat_preco", [])
    colunas = [c for c in set(preditores_preco + ["bairro_padronizado", "preco"]) if c in df.columns]
    base = df.dropna(subset=colunas).copy()
    if base.empty:
        return {}

    preco_real = base["preco"].values
    preco_modelo = np.expm1(_modelos["modelo_preco"].predict(base[preditores_preco]))

    media_bairro = base.groupby("bairro_padronizado")["preco"].transform("mean")
    preco_baseline = media_bairro.values

    return {
        "rmse_modelo": float(np.sqrt(mean_squared_error(preco_real, preco_modelo))),
        "mae_modelo": float(mean_absolute_error(preco_real, preco_modelo)),
        "r2_modelo": float(r2_score(preco_real, preco_modelo)),
        "rmse_baseline": float(np.sqrt(mean_squared_error(preco_real, preco_baseline))),
        "mae_baseline": float(mean_absolute_error(preco_real, preco_baseline)),
        "r2_baseline": float(r2_score(preco_real, preco_baseline)),
    }


@st.cache_data(show_spinner=False)
def calcular_previsoes_dataset(df: pd.DataFrame, _modelos: dict) -> pd.DataFrame:
    """Roda os modelos de preço e ocupação em LOTE sobre todos os imóveis reais do
    dataset (não linha a linha), para alimentar a busca do 'imóvel ideal'. O underscore
    em `_modelos` evita que o Streamlit tente fazer hash dos objetos do modelo (não
    hasheáveis) — só o `df` é usado como chave do cache."""
    preditores_preco = _modelos["preditores_num_preco"] + _modelos["preditores_cat_preco"]
    preditores_ocup = _modelos["preditores_num_ocup"] + _modelos["preditores_cat_ocup"]
    colunas_necessarias = [c for c in set(preditores_preco + preditores_ocup) if c in df.columns]

    base = df.dropna(subset=colunas_necessarias).copy()

    preco_previsto = np.expm1(_modelos["modelo_preco"].predict(base[preditores_preco]))
    ocupacao_prevista = _modelos["modelo_ocupacao"].predict(base[preditores_ocup])
    ocupacao_prevista = np.clip(ocupacao_prevista, 0.0, 1.0)

    base["preco_previsto"] = preco_previsto
    base["ocupacao_prevista"] = ocupacao_prevista
    base["rentabilidade_anual_prevista"] = preco_previsto * ocupacao_prevista * 365
    return base


def aplicar_perfil(perfil: dict):
    """Callback: copia os valores do perfil escolhido para o session_state,
    fazendo os widgets do formulário assumirem esses valores no próximo rerun."""
    mapa_chaves = {
        "capacidade_hospedes": "sim_capacidade_hospedes", "banheiros": "sim_banheiros",
        "quartos": "sim_quartos", "camas": "sim_camas", "qtd_comodidades": "sim_qtd_comodidades",
        "nota_composta": "sim_nota_composta", "noites_minimas": "sim_noites_minimas",
        "noites_maximas": "sim_noites_maximas", "total_anuncios_anfitriao": "sim_total_anuncios",
        "numero_avaliacoes": "sim_numero_avaliacoes", "avaliacoes_por_mes": "sim_avaliacoes_por_mes",
        "tipo_quarto": "sim_tipo_quarto", "tipo_propriedade": "sim_tipo_propriedade",
        "tipo_hospedagem": "sim_tipo_hospedagem", "e_superanfitriao": "sim_e_superanfitriao",
        "reserva_instantanea": "sim_reserva_instantanea",
        "tem_banheiro_privativo": "sim_banheiro_privativo", "flexibilidade_estadia": "sim_flexibilidade",
    }
    for campo, chave_widget in mapa_chaves.items():
        if campo in perfil:
            valor = perfil[campo]
            if campo in ("tem_banheiro_privativo", "flexibilidade_estadia"):
                valor = "Sim" if valor else "Não"
            st.session_state[chave_widget] = valor


# ---------------------------------------------------------------------------
# Carrega tudo
# ---------------------------------------------------------------------------
df, gdf, calendario, temporal = carregar_dados()
df = criar_perfil_anfitriao(df)
df = criar_variaveis_avaliacoes(df)
df = criar_faixa_preco_controlada(df)
agg = agregar_por_bairro(df)
pontos_turisticos = buscar_pontos_turisticos()
agg = adicionar_distancia_turistica(agg, pontos_turisticos)
mapa_gdf = gdf.merge(agg, left_on="neighbourhood", right_on="bairro_padronizado", how="left")

PASTA_ASSETS = os.path.join(os.path.dirname(__file__), "assets")
CAMINHO_LOGO = os.path.join(PASTA_ASSETS, "logo_vander22.png")

col_titulo, col_info = st.columns([8, 1], vertical_alignment="bottom")
with col_titulo:
    if os.path.exists(CAMINHO_LOGO):
        st.image(CAMINHO_LOGO, width=260)
        st.markdown(
            '<div class="cabecalho-app">'
            "<p>Análise Espacial — Airbnb Rio de Janeiro · Preço, ocupação, luxo, "
            "turismo e sazonalidade dos anúncios por bairro.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="cabecalho-app">'
            "<h1>Análise Espacial — Airbnb Rio de Janeiro</h1>"
            "<p>Preço, ocupação, luxo, turismo e sazonalidade dos anúncios por bairro.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
with col_info:
    with st.popover("ℹ️ Info", use_container_width=True):
        st.markdown(
            "**Perfil do anfitrião** — 1) `Superanfitrião` se tiver o selo oficial do Airbnb; "
            "2) senão, `Profissional (4+ imóveis)` se administra 4 ou mais anúncios; "
            "3) `Recorrente (2-3 imóveis)`; 4) `Iniciante (1 imóvel)`. O selo tem prioridade "
            "porque já é um critério oficial (nota alta + baixo cancelamento + resposta rápida).\n\n"
            f"**Faixa de preço controlada** — para não deixar diárias muito fora da curva "
            f"(≥ R$ {df['preco_teto_outlier'].iloc[0]:,.0f}, percentil 99) distorcerem a escala "
            "de cor do mapa e os filtros, essas diárias caem numa faixa própria "
            "`Outlier (fora da curva)`. As demais são cortadas em quartis do preço 'normal' "
            "(Econômico / Médio / Alto / Premium).\n\n"
            "**Variáveis de avaliação** — `qualidade_percebida` usa as faixas de nota do próprio "
            "critério de superanfitrião (< 4,0 / 4,0–4,7 / ≥ 4,8); `nivel_atividade_avaliacoes` "
            "usa tercis de avaliações/mês (só entre quem já recebeu avaliação); "
            "`indice_engajamento` = nota composta × avaliações por mês, pra rankear anúncios "
            "bons **e** populares ao mesmo tempo."
        )

aba_inicio, aba_mapa, aba_sazonalidade, aba_simulador, aba_avaliacoes, aba_recomendacao, aba_assistente = st.tabs(
    ["🏠 Início", "🗺️ Mapa Dinâmico", "📅 Evolução Temporal dos Preços", "🧮 Simulador de Investimento",
     "📝 Análise de Avaliações", "🎯 Recomendação por Turismo", "🤖 Assistente IA"]
)

# ---------------------------------------------------------------------------
# ABA 0 — INÍCIO
# Tela de boas-vindas: visão geral do dataset (KPIs) + guia das demais abas.
# Fica sempre visível ao abrir o app, antes de qualquer filtro ser aplicado.
# ---------------------------------------------------------------------------
with aba_inicio:
    st.markdown(
        """
        <div style="padding: 1.8rem 2rem; border-radius: 14px;
                    background: linear-gradient(135deg, rgba(255,110,64,0.12), rgba(30,61,89,0.08));
                    border: 1px solid rgba(128,128,128,0.2); margin-bottom: 1.6rem;">
            <h2 style="margin-bottom: 0.4rem;">👋 Bem-vindo(a)!</h2>
            <p style="font-size: 1.05rem; opacity: 0.85; margin-bottom: 0;">
                Este painel cruza dados geográficos, temporais e de avaliação dos anúncios de
                Airbnb no Rio de Janeiro para apoiar decisões de <b>hóspedes</b> e
                <b>investidores</b>. Use as abas acima para explorar cada análise —
                um resumo de cada uma está logo abaixo.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### 📊 Panorama geral do dataset")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Anúncios analisados", f"{len(df):,}".replace(",", "."))
    kpi2.metric("Bairros cobertos", f"{agg.shape[0]}")
    kpi3.metric("Preço médio/diária", f"R$ {df['preco'].mean():,.0f}".replace(",", "."))
    if "nota_composta" in df.columns:
        kpi4.metric("Nota composta média", f"{df['nota_composta'].mean():.2f}")
    else:
        kpi4.metric("Rentabilidade média/dia", f"R$ {df['rentabilidade_diaria'].mean():,.0f}".replace(",", "."))

    st.divider()
    st.markdown("##### 🧭 O que você encontra em cada aba")

    guia_abas = [
        ("🗺️", "Mapa Dinâmico",
         "Explore o Rio por bairro: score de luxo, rentabilidade, ocupação e proximidade "
         "de pontos turísticos, tudo num mapa interativo com filtros de preço e região."),
        ("📅", "Evolução Temporal dos Preços",
         "Acompanhe a sazonalidade e o histórico de preços entre as coletas de dados, "
         "identificando altas e baixas ao longo do ano."),
        ("🧮", "Simulador de Investimento",
         "Monte o perfil de um imóvel (quartos, comodidades, tipo de hospedagem etc.) e "
         "receba previsões de preço, ocupação e rentabilidade anual."),
        ("📝", "Análise de Avaliações",
         "Veja os pontos fortes e fracos por bairro e por anúncio, com base nas notas e "
         "na frequência de avaliações recebidas."),
        ("🎯", "Recomendação por Turismo",
         "Encontre os melhores anúncios perto de pontos turísticos escolhidos, combinando "
         "distância, preço e nota num único score."),
        ("🤖", "Assistente IA",
         "Converse livremente ou pergunte sobre os dados deste app — a assistente tem "
         "acesso a um resumo do dataset carregado."),
    ]

    for linha_guia in (guia_abas[i:i + 3] for i in range(0, len(guia_abas), 3)):
        colunas_guia = st.columns(3)
        for coluna, (emoji, titulo, descricao) in zip(colunas_guia, linha_guia):
            with coluna:
                with st.container(border=True):
                    st.markdown(f"###### {emoji} {titulo}")
                    st.caption(descricao)

# ---------------------------------------------------------------------------
# ABA 1 — MAPA DINÂMICO
# ---------------------------------------------------------------------------
with aba_mapa:
    st.subheader("Mapa por bairro: luxo, rentabilidade, ocupação e turismo")

    col_filtros, col_mapa = st.columns([1, 3])

    with col_filtros:
        preco_min, preco_max = float(agg["preco_medio"].min()), float(agg["preco_medio"].max())
        faixa_preco = st.slider(
            "Faixa de preço médio do bairro (R$)",
            min_value=float(np.floor(preco_min)),
            max_value=float(np.ceil(preco_max)),
            value=(float(np.floor(preco_min)), float(np.ceil(preco_max))),
        )
        mostrar_populares = st.checkbox("Mostrar bairros de região popular", value=True)
        mostrar_nao_populares = st.checkbox("Mostrar bairros de região não popular", value=True)
        mostrar_turismo = st.checkbox("Mostrar pontos turísticos", value=True)

        st.metric("Bairros no filtro atual",
                   int(agg[(agg["preco_medio"] >= faixa_preco[0]) & (agg["preco_medio"] <= faixa_preco[1])].shape[0]))

    agg_filtrado = agg[(agg["preco_medio"] >= faixa_preco[0]) & (agg["preco_medio"] <= faixa_preco[1])]
    if not mostrar_populares:
        agg_filtrado = agg_filtrado[agg_filtrado["regiao_popular"] != 1]
    if not mostrar_nao_populares:
        agg_filtrado = agg_filtrado[agg_filtrado["regiao_popular"] != 0]

    mapa_gdf_filtrado = mapa_gdf[mapa_gdf["bairro_padronizado"].isin(agg_filtrado["bairro_padronizado"])]

    with col_mapa:
        m = folium.Map(location=[-22.925, -43.30], zoom_start=11, tiles="CartoDB positron")

        if len(mapa_gdf_filtrado) > 0:
            colormap_luxo = linear.YlOrRd_09.scale(
                agg["score_luxo_medio"].min(), agg["score_luxo_medio"].max()
            )
            colormap_luxo.caption = "Score de Luxo médio por bairro"

            def estilo_bairro(feature):
                valor = feature["properties"].get("score_luxo_medio")
                return {
                    "fillColor": colormap_luxo(valor) if valor is not None else "#cccccc",
                    "color": "#555555",
                    "weight": 0.8,
                    "fillOpacity": 0.65,
                }

            tooltip_fields = ["neighbourhood", "qtd_anuncios", "score_luxo_medio",
                               "faixa_preco_moda", "preco_medio", "rentabilidade_media",
                               "taxa_ocupacao_media", "distancia_ponto_turistico_km"]
            tooltip_aliases = ["Bairro:", "Nº anúncios:", "Score luxo:",
                                "Faixa de preço:", "Preço médio (R$):", "Rentabilidade média:",
                                "Taxa ocupação média:", "Dist. ponto turístico (km):"]

            folium.GeoJson(
                mapa_gdf_filtrado,
                name="Score de Luxo por bairro",
                style_function=estilo_bairro,
                tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_aliases,
                                                localize=True, sticky=True),
            ).add_to(m)
            colormap_luxo.add_to(m)

            colormap_ocup = linear.PuBuGn_09.scale(
                agg["taxa_ocupacao_media"].min(), agg["taxa_ocupacao_media"].max()
            )
            grupo_popular = FeatureGroup(name="Bairros de região popular", show=True)
            grupo_nao_popular = FeatureGroup(name="Bairros de região não popular", show=True)
            max_rent = agg["rentabilidade_media"].max()

            for _, row in agg_filtrado.dropna(subset=["lat", "lon"]).iterrows():
                raio = 4 + 20 * (row["rentabilidade_media"] / max_rent if max_rent else 0)
                popup_html = (
                    f"<b>{row['bairro_padronizado']}</b><br>"
                    f"Anúncios: {int(row['qtd_anuncios'])}<br>"
                    f"Score luxo: {row['score_luxo_medio']:.2f}<br>"
                    f"Faixa preço: {row['faixa_preco_moda']}<br>"
                    f"Preço médio: R$ {row['preco_medio']:.0f}<br>"
                    f"Rentabilidade média: {row['rentabilidade_media']:.2f}<br>"
                    f"Taxa ocupação média: {row['taxa_ocupacao_media']:.2%}"
                )
                marker = folium.CircleMarker(
                    location=[row["lat"], row["lon"]],
                    radius=raio,
                    color=colormap_ocup(row["taxa_ocupacao_media"]),
                    fill=True,
                    fill_color=colormap_ocup(row["taxa_ocupacao_media"]),
                    fill_opacity=0.75,
                    weight=1,
                    popup=folium.Popup(popup_html, max_width=250),
                )
                destino = grupo_popular if row["regiao_popular"] == 1 else grupo_nao_popular
                marker.add_to(destino)

            grupo_popular.add_to(m)
            grupo_nao_popular.add_to(m)

            if mostrar_turismo and not pontos_turisticos.empty:
                grupo_turismo = FeatureGroup(name="Pontos turísticos", show=True)
                for _, pt in pontos_turisticos.iterrows():
                    folium.Marker(
                        location=[pt["lat"], pt["lon"]],
                        popup=pt["nome"],
                        icon=folium.Icon(color="cadetblue", icon="star", prefix="fa"),
                    ).add_to(grupo_turismo)
                grupo_turismo.add_to(m)

            LayerControl(collapsed=False).add_to(m)
        else:
            st.warning("Nenhum bairro dentro do filtro selecionado.")

        st_folium(m, width=None, height=650, returned_objects=[])

# ---------------------------------------------------------------------------
# ABA 2 — SAZONALIDADE
# ---------------------------------------------------------------------------
with aba_sazonalidade:
    if calendario is None or temporal is None:
        st.warning(
            "Esta aba precisa dos arquivos `calendar_agregado.parquet` e `listings_temporal.parquet` "
            "(calendário de disponibilidade mês a mês e múltiplas coletas), que não foram encontrados "
            "na pasta de dados. Coloque esses arquivos em `dados/` para habilitar esta análise."
        )
    else:
        st.subheader("Ocupação e preço ao longo do ano")

        mensal = calendario[calendario["granularidade"] == "mensal"].copy()
        media_mes = mensal.groupby("mes")["taxa_ocupacao"].mean().reset_index()
        media_mes["mes_nome"] = media_mes["mes"].map(NOMES_MES)
        media_mes = media_mes.sort_values("mes")

        col1, col2 = st.columns(2)

        with col1:
            fig_ocup = px.line(media_mes, x="mes_nome", y="taxa_ocupacao", markers=True,
                                title="Taxa de ocupação média por mês (todas as coletas)")
            fig_ocup.update_layout(yaxis_tickformat=".0%", xaxis_title="Mês", yaxis_title="Taxa de ocupação")
            mes_pico = media_mes.loc[media_mes["taxa_ocupacao"].idxmax()]
            mes_baixa = media_mes.loc[media_mes["taxa_ocupacao"].idxmin()]
            fig_ocup.add_annotation(x=mes_pico["mes_nome"], y=mes_pico["taxa_ocupacao"],
                                     text="Alta temporada", showarrow=True, arrowhead=2, yshift=15)
            fig_ocup.add_annotation(x=mes_baixa["mes_nome"], y=mes_baixa["taxa_ocupacao"],
                                     text="Baixa temporada", showarrow=True, arrowhead=2, yshift=-15)
            st.plotly_chart(fig_ocup, width='stretch')

        with col2:
            preco_mes = temporal.groupby("mes_coleta")["price"].mean().reset_index()
            preco_mes["mes_nome"] = preco_mes["mes_coleta"].map(NOMES_MES)
            preco_mes = preco_mes.sort_values("mes_coleta")

            fig_preco = px.bar(preco_mes, x="mes_nome", y="price",
                                title="Preço médio por mês (coletas disponíveis)")
            fig_preco.update_layout(xaxis_title="Mês", yaxis_title="Preço médio (R$)")
            st.plotly_chart(fig_preco, width='stretch')

        st.info(
            f"📈 **Melhor mês pra alugar/investir (maior ocupação): {mes_pico['mes_nome']}** "
            f"({mes_pico['taxa_ocupacao']:.1%} de ocupação média)\n\n"
            f"📉 **Melhor mês pra economizar viajando (menor ocupação): {mes_baixa['mes_nome']}** "
            f"({mes_baixa['taxa_ocupacao']:.1%} de ocupação média)"
        )

        st.caption(
            "Nota: o preço por mês só está disponível para os meses efetivamente coletados "
            f"({', '.join(preco_mes['mes_nome'])}). "
            "A ocupação mensal cobre o ano cheio, pois vem do calendário futuro de disponibilidade."
        )

    st.divider()
    st.subheader("📈 Evolução histórica de preços entre coletas")

    if temporal is None:
        st.warning(
            "Esta seção precisa do arquivo `listings_temporal.parquet` (múltiplas coletas do "
            "Airbnb ao longo do tempo), que não foi encontrado na pasta de dados. "
            "Coloque esse arquivo em `dados/` para habilitar esta análise."
        )
    else:
        st.caption(
            "Cada coleta é uma foto do mercado num momento diferente — não é o preço de um "
            "mesmo anúncio ao longo do tempo, mas sim como o conjunto de anúncios ativos se "
            "comportava em cada data de coleta."
        )

        coluna_tempo = "mes_coleta" if "mes_coleta" in temporal.columns else None
        coluna_preco_temp = "price" if "price" in temporal.columns else (
            "preco" if "preco" in temporal.columns else None
        )

        if coluna_tempo is None or coluna_preco_temp is None:
            st.warning("Não encontrei as colunas necessárias em `listings_temporal.parquet` para montar esta análise.")
        else:
            resumo_temporal = (
                temporal.groupby(coluna_tempo)[coluna_preco_temp]
                .agg(preco_medio="mean", preco_mediano="median", n_anuncios="count")
                .reset_index()
            )
            resumo_temporal["mes_nome"] = resumo_temporal[coluna_tempo].map(NOMES_MES).fillna(
                resumo_temporal[coluna_tempo].astype(str)
            )
            resumo_temporal = resumo_temporal.sort_values(coluna_tempo)
            resumo_temporal["variacao_pct_nominal"] = (
                resumo_temporal["preco_medio"] / resumo_temporal["preco_medio"].iloc[0] - 1
            ) * 100

            col_evo1, col_evo2 = st.columns(2)

            with col_evo1:
                fig_evo = make_subplots(specs=[[{"secondary_y": True}]])
                fig_evo.add_trace(
                    go.Scatter(x=resumo_temporal["mes_nome"], y=resumo_temporal["preco_medio"],
                               mode="lines+markers", name="Preço médio", line=dict(color="#1e3d59", width=3)),
                    secondary_y=False,
                )
                fig_evo.add_trace(
                    go.Scatter(x=resumo_temporal["mes_nome"], y=resumo_temporal["preco_mediano"],
                               mode="lines+markers", name="Preço mediano",
                               line=dict(color="#ff6e40", width=3, dash="dash")),
                    secondary_y=False,
                )
                fig_evo.add_trace(
                    go.Bar(x=resumo_temporal["mes_nome"], y=resumo_temporal["n_anuncios"],
                           name="Nº de anúncios", marker_color="#17b978", opacity=0.35),
                    secondary_y=True,
                )
                fig_evo.update_layout(title="Preço médio/mediano e volume de anúncios por coleta",
                                       xaxis_title="Coleta", legend=dict(orientation="h", y=-0.2))
                fig_evo.update_yaxes(title_text="Preço (R$)", secondary_y=False)
                fig_evo.update_yaxes(title_text="Nº de anúncios", secondary_y=True)
                st.plotly_chart(fig_evo, width="stretch")

            with col_evo2:
                fig_var = px.line(
                    resumo_temporal, x="mes_nome", y="variacao_pct_nominal", markers=True,
                    title="Variação percentual acumulada do preço médio (nominal)",
                )
                fig_var.add_hline(y=0, line_dash="dot", line_color="gray")
                fig_var.update_layout(xaxis_title="Coleta", yaxis_title="Variação acumulada (%)")
                fig_var.update_traces(line_color="#9B59B6")
                st.plotly_chart(fig_var, width="stretch")
                st.caption("Valores nominais, sem correção pela inflação (IPCA).")

            st.dataframe(
                resumo_temporal[["mes_nome", "preco_medio", "preco_mediano", "n_anuncios", "variacao_pct_nominal"]]
                .rename(columns={
                    "mes_nome": "Coleta", "preco_medio": "Preço médio (R$)",
                    "preco_mediano": "Preço mediano (R$)", "n_anuncios": "Nº anúncios",
                    "variacao_pct_nominal": "Variação acumulada (%)",
                }).round(2),
                width="stretch", hide_index=True,
            )

            st.divider()
            st.markdown("##### Evolução do preço médio por bairro")

            coluna_bairro_temp = next(
                (c for c in ["bairro_padronizado", "neighbourhood_cleansed", "neighbourhood"] if c in temporal.columns),
                None,
            )
            if coluna_bairro_temp is None:
                st.info("A base `listings_temporal.parquet` não possui coluna de bairro — não é possível detalhar a evolução por bairro.")
            else:
                ultima_coleta = temporal[coluna_tempo].max()
                top_bairros_default = (
                    temporal[temporal[coluna_tempo] == ultima_coleta]
                    .groupby(coluna_bairro_temp).size().sort_values(ascending=False).head(10).index.tolist()
                )
                todos_bairros = sorted(temporal[coluna_bairro_temp].dropna().unique().tolist())
                bairros_selecionados = st.multiselect(
                    "Bairros para comparar (padrão: top 10 com mais anúncios na coleta mais recente)",
                    options=todos_bairros, default=top_bairros_default,
                )

                if bairros_selecionados:
                    evolucao_bairro = (
                        temporal[temporal[coluna_bairro_temp].isin(bairros_selecionados)]
                        .groupby([coluna_tempo, coluna_bairro_temp])[coluna_preco_temp]
                        .mean().reset_index()
                    )
                    evolucao_bairro["mes_nome"] = evolucao_bairro[coluna_tempo].map(NOMES_MES).fillna(
                        evolucao_bairro[coluna_tempo].astype(str)
                    )
                    evolucao_bairro = evolucao_bairro.sort_values(coluna_tempo)

                    fig_bairro = px.line(
                        evolucao_bairro, x="mes_nome", y=coluna_preco_temp, color=coluna_bairro_temp,
                        markers=True, title="Evolução do preço médio por bairro selecionado",
                    )
                    fig_bairro.update_layout(xaxis_title="Coleta", yaxis_title="Preço médio (R$)",
                                              legend_title="Bairro")
                    st.plotly_chart(fig_bairro, width="stretch")
                else:
                    st.info("Selecione pelo menos um bairro para ver o gráfico.")

# ---------------------------------------------------------------------------
# ABA 3 — SIMULADOR DE INVESTIMENTO
# ---------------------------------------------------------------------------
with aba_simulador:
    st.subheader("Simulador de Preço, Ocupação e Rentabilidade")
    st.caption(
        "Informe as características de um imóvel e receba uma estimativa de preço de diária, "
        "taxa de ocupação e rentabilidade anual."
    )

    modelos = carregar_modelos()

    if not JOBLIB_DISPONIVEL:
        st.warning("A biblioteca `joblib` não está instalada. Rode `pip install joblib` para habilitar esta aba.")
    elif modelos is None:
        st.warning(
            "Não encontrei os arquivos do modelo. Rode o notebook `05-1_modelagem_preditiva.ipynb` "
            "(ele já gera tudo sozinho) e copie os 3 arquivos que ele salva em `outputs/modelos/`:\n\n"
            "- `modelo_preco.joblib`\n"
            "- `modelo_ocupacao.joblib`\n"
            "- `config_preditores.joblib`\n\n"
            f"para a pasta `{PASTA_MODELOS}` (crie a pasta `modelos` ao lado de `app2.py` se ela não existir)."
        )
    else:
        perfis_predefinidos = calcular_perfis_predefinidos(df)

        st.markdown("##### 🏷️ Perfil rápido")
        st.caption(
            "Escolha um perfil de Airbnb."
        )
        opcoes_perfil = ["Personalizado"] + list(perfis_predefinidos.keys())
        st.radio(
            "Perfil do imóvel", opcoes_perfil, horizontal=True, key="sim_perfil_escolhido",
            on_change=lambda: (aplicar_perfil(perfis_predefinidos[st.session_state["sim_perfil_escolhido"]])
                                if st.session_state["sim_perfil_escolhido"] != "Personalizado" else None),
        )

        st.markdown("##### 📍 Localização")
        opcoes_bairro = ["bairro"] + sorted(
            agg["bairro_padronizado"].dropna().unique().tolist()
        )
        bairro_escolhido = st.selectbox(
            "Bairro de referência (preenche automaticamente a latitude/longitude média do bairro)",
            opcoes_bairro,
        )
        if bairro_escolhido != "bairro":
            linha_bairro = agg[agg["bairro_padronizado"] == bairro_escolhido].iloc[0]
            lat_padrao, lon_padrao = float(linha_bairro["lat"]), float(linha_bairro["lon"])
            bairro_referencia = {
                "nome": bairro_escolhido,
                "preco_medio": float(linha_bairro["preco_medio"]),
                "taxa_ocupacao_media": float(linha_bairro["taxa_ocupacao_media"]),
                "rentabilidade_anual_media": float(linha_bairro["rentabilidade_media"]) * 365,
            }
        else:
            lat_padrao, lon_padrao = float(df["latitude"].mean()), float(df["longitude"].mean())
            bairro_referencia = None

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            max_hospedes = int(df["capacidade_hospedes"].max())
            capacidade_hospedes = st.number_input(f"Capacidade de hóspedes (máx.: {max_hospedes})",
                                                     min_value=1, max_value=max_hospedes,
                                                     value=4, key="sim_capacidade_hospedes")
            max_banheiros = float(df["banheiros"].max())
            banheiros = st.number_input(f"Banheiros (máx.: {max_banheiros:.1f})",
                                          min_value=0.0, max_value=max_banheiros,
                                          value=1.0, step=0.5, key="sim_banheiros",
                                          help="Meio-banheiro (ex.: 1.5) representa um lavabo/banheiro social sem chuveiro.")
            max_quartos = int(df["quartos"].max())
            quartos = st.number_input(f"Quartos (máx.: {max_quartos})",
                                        min_value=0, max_value=max_quartos,
                                        value=2, step=1, key="sim_quartos")
            max_camas = int(df["camas"].max())
            camas = st.number_input(f"Camas (máx.: {max_camas})",
                                      min_value=0, max_value=max_camas,
                                      value=2, step=1, key="sim_camas")
        with col_b:
            max_comodidades = int(df["qtd_comodidades"].max())
            qtd_comodidades = st.number_input(f"Nº de comodidades (máx.: {max_comodidades})",
                                                min_value=0, max_value=max_comodidades,
                                                value=15, key="sim_qtd_comodidades")
            nota_composta = st.slider("Nota composta (0 a 5)", 0.0, 5.0, 4.8, 0.1, key="sim_nota_composta")
            max_noites_min = int(df["noites_minimas"].max())
            noites_minimas = st.number_input(f"Noites mínimas (máx.: {max_noites_min})",
                                               min_value=1, max_value=max_noites_min,
                                               value=2, key="sim_noites_minimas")
            max_noites_max = int(df["noites_maximas"].max())
            noites_maximas = st.number_input(f"Noites máximas (máx.: {max_noites_max})",
                                               min_value=1, max_value=max_noites_max,
                                               value=30, key="sim_noites_maximas")
        with col_c:
            max_anuncios = int(df["total_anuncios_anfitriao"].max())
            total_anuncios_anfitriao = st.number_input(f"Total de anúncios do anfitrião (máx.: {max_anuncios})",
                                                          min_value=1, max_value=max_anuncios,
                                                          value=1, key="sim_total_anuncios")
            tem_banheiro_privativo = st.selectbox("Banheiro privativo?", ["Sim", "Não"],
                                                    key="sim_banheiro_privativo") == "Sim"
            flexibilidade_estadia = st.selectbox("Cancelamento/estadia flexível?", ["Sim", "Não"],
                                                   key="sim_flexibilidade") == "Sim"
            latitude = st.number_input("Latitude", value=lat_padrao, format="%.5f")
            longitude = st.number_input("Longitude", value=lon_padrao, format="%.5f")

        st.markdown("##### 🏷️ Características categóricas")
        col_d, col_e, col_f = st.columns(3)
        with col_d:
            tipo_quarto = st.selectbox("Tipo de quarto", sorted(df["tipo_quarto"].dropna().unique().tolist()),
                                         format_func=capitalizar_primeira, key="sim_tipo_quarto")
            tipo_propriedade = st.selectbox("Tipo de propriedade",
                                              sorted(df["tipo_propriedade"].dropna().unique().tolist()),
                                              format_func=capitalizar_primeira, key="sim_tipo_propriedade")
        with col_e:
            tipo_hospedagem = st.selectbox("Tipo de hospedagem",
                                             sorted(df["tipo_hospedagem"].dropna().unique().tolist()),
                                             format_func=capitalizar_primeira, key="sim_tipo_hospedagem")
            e_superanfitriao = st.selectbox("É superanfitrião?", ["f", "t"],
                                              format_func=lambda x: "Sim" if x == "t" else "Não",
                                              key="sim_e_superanfitriao")
        with col_f:
            reserva_instantanea = st.selectbox("Reserva instantânea?", ["f", "t"],
                                                 format_func=lambda x: "Sim" if x == "t" else "Não",
                                                 key="sim_reserva_instantanea")
            meses_disponiveis = list(modelos.get("multiplicador_sazonal", {}).keys())
            mes_referencia = st.selectbox("Mês de referência (ajuste sazonal, opcional)",
                                            ["(sem ajuste sazonal)"] + meses_disponiveis)

        st.markdown("##### ⭐ Histórico de avaliações (usado apenas no modelo de preço)")
        col_g, col_h = st.columns(2)
        with col_g:
            max_avaliacoes = int(df["numero_avaliacoes"].max())
            numero_avaliacoes = st.number_input(f"Número de avaliações (máx.: {max_avaliacoes})",
                                                  min_value=0, max_value=max_avaliacoes,
                                                  value=20, key="sim_numero_avaliacoes")
        with col_h:
            max_avaliacoes_mes = float(df["avaliacoes_por_mes"].max())
            avaliacoes_por_mes = st.number_input(f"Avaliações por mês (máx.: {max_avaliacoes_mes:.1f})",
                                                   min_value=0.0, max_value=max_avaliacoes_mes,
                                                   value=1.5, step=0.1, key="sim_avaliacoes_por_mes")

        if st.button("🔍 Calcular estimativa", type="primary"):
            dados_imovel = {
                "capacidade_hospedes": capacidade_hospedes, "banheiros": banheiros, "quartos": quartos,
                "camas": camas, "qtd_comodidades": qtd_comodidades, "nota_composta": nota_composta,
                "latitude": latitude, "longitude": longitude, "noites_minimas": noites_minimas,
                "noites_maximas": noites_maximas, "total_anuncios_anfitriao": total_anuncios_anfitriao,
                "tem_banheiro_privativo": int(tem_banheiro_privativo),
                "flexibilidade_estadia": int(flexibilidade_estadia),
                "tipo_quarto": tipo_quarto, "tipo_propriedade": tipo_propriedade,
                "tipo_hospedagem": tipo_hospedagem, "e_superanfitriao": e_superanfitriao,
                "reserva_instantanea": reserva_instantanea,
                "numero_avaliacoes": numero_avaliacoes, "avaliacoes_por_mes": avaliacoes_por_mes,
            }
            mes_para_calculo = None if mes_referencia == "(sem ajuste sazonal)" else mes_referencia

            try:
                resultado = prever_investimento(modelos, dados_imovel, mes_referencia=mes_para_calculo)
            except Exception as e:
                st.error(f"Não foi possível calcular a estimativa: {e}")
            else:
                st.divider()
                st.subheader("📊 Resultado da simulação")

                if resultado["tem_margem_erro"]:
                    p_min, p_max = resultado["preco_intervalo"]
                    o_min, o_max = resultado["ocupacao_intervalo"]
                    r_min, r_max = resultado["rentabilidade_intervalo"]
                    valor_preco = f"R$ {resultado['preco_diaria_previsto']:.2f} (± R$ {resultado['preco_erro_abs']:.2f})"
                    valor_ocup = f"{resultado['taxa_ocupacao_prevista']:.1%} (± {resultado['ocupacao_erro_abs']:.1%})"
                    valor_rent = f"R$ {resultado['rentabilidade_anual_estimada']:,.2f}"
                else:
                    valor_preco = f"R$ {resultado['preco_diaria_previsto']:.2f}"
                    valor_ocup = f"{resultado['taxa_ocupacao_prevista']:.1%}"
                    valor_rent = f"R$ {resultado['rentabilidade_anual_estimada']:,.2f}"

                if bairro_referencia:
                    delta_preco = resultado["preco_diaria_previsto"] - bairro_referencia["preco_medio"]
                    delta_ocup = resultado["taxa_ocupacao_prevista"] - bairro_referencia["taxa_ocupacao_media"]
                    delta_rent = (resultado["rentabilidade_anual_estimada"]
                                  - bairro_referencia["rentabilidade_anual_media"])
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Preço sugerido / diária", valor_preco,
                              delta=f"R$ {delta_preco:+.2f} vs. média do bairro")
                    m2.metric("Taxa de ocupação estimada", valor_ocup,
                              delta=f"{delta_ocup:+.1%} vs. média do bairro")
                    m3.metric("Noites ocupadas/ano (estim.)", f"{resultado['noites_ocupadas_ano_estimadas']:.0f}")
                    m4.metric("Rentabilidade anual estimada", valor_rent,
                              delta=f"R$ {delta_rent:+,.2f} vs. média do bairro")
                    st.caption(
                        f"Comparação com a média observada em **{bairro_referencia['nome']}**: "
                        f"preço R$ {bairro_referencia['preco_medio']:.2f}, ocupação "
                        f"{bairro_referencia['taxa_ocupacao_media']:.1%}, rentabilidade anual "
                        f"R$ {bairro_referencia['rentabilidade_anual_media']:,.2f} (estimativa própria do app, "
                        "não vem do modelo)."
                    )
                else:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Preço sugerido / diária", valor_preco)
                    m2.metric("Taxa de ocupação estimada", valor_ocup)
                    m3.metric("Noites ocupadas/ano (estim.)", f"{resultado['noites_ocupadas_ano_estimadas']:.0f}")
                    m4.metric("Rentabilidade anual estimada", valor_rent)
                    st.caption("💡 Escolha um bairro de referência acima para comparar com a média da região.")

                if resultado["tem_margem_erro"]:
                    st.caption(
                        f"📐 Faixa realista considerando o erro do modelo: rentabilidade anual entre "
                        f"R$ {r_min:,.2f} e R$ {r_max:,.2f}. "
                        f"Confiabilidade dos modelos: preço R² = {resultado['r2_preco']:.2f} · "
                        f"ocupação R² = {resultado['r2_ocupacao']:.2f} "
                        "(quanto mais próximo de 1, melhor o modelo explica os dados)."
                    )


                if mes_para_calculo:
                    st.caption(
                        f"Ajuste sazonal aplicado para {mes_para_calculo}: "
                        f"fator {resultado['fator_sazonal_aplicado']:.3f} "
                        "(1.0 = ocupação média das coletas; acima de 1.0 indica alta temporada)."
                    )
                st.caption(
                    "⚠️ Estimativa gerada por modelos de Machine Learning treinados com dados históricos "
                    "do Airbnb Rio de Janeiro. Não constitui recomendação de investimento."
                )

        # -----------------------------------------------------------------
        # Buscar o imóvel ideal (otimização inversa, entre imóveis reais)
        # -----------------------------------------------------------------
        st.divider()
        st.markdown("##### 🎯 Buscar o imóvel ideal")
        st.caption(
            "Aqui você informa o **objetivo** e o app busca, entre os imóveis "
            "**reais** do dataset, o top 5 de configurações que o modelo prevê como mais próximas dele. "
        )

        objetivo = st.selectbox(
            "Objetivo",
            ["Maior rentabilidade possível", "Maior preço de diária possível", "Ocupação-alvo (mín. dias ocupados)"],
            key="busca_objetivo",
        )
        ocupacao_alvo_pct = None
        if objetivo == "Ocupação-alvo (mín. dias ocupados)":
            ocupacao_alvo_pct = st.selectbox(
                "Taxa de ocupação-alvo", [50, 75, 90, 100],
                format_func=lambda x: f"{x}%", key="busca_ocupacao_alvo",
            )

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            filtro_tipo_propriedade = st.multiselect(
                "Filtrar por tipo de propriedade (opcional)",
                sorted(df["tipo_propriedade"].dropna().unique().tolist()),
                format_func=capitalizar_primeira,
                key="busca_filtro_tipo_prop",
            )
        with col_f2:
            filtro_bairro_busca = st.multiselect(
                "Filtrar por bairro (opcional)",
                sorted(agg["bairro_padronizado"].dropna().unique().tolist()),
                key="busca_filtro_bairro",
            )

        if st.button("🔎 Buscar top 5 imóveis ideais", type="primary", key="busca_btn"):
            with st.spinner("Calculando previsões do modelo para os imóveis reais do dataset..."):
                base_pred = calcular_previsoes_dataset(df, modelos)

            filtrado = base_pred
            if filtro_tipo_propriedade:
                filtrado = filtrado[filtrado["tipo_propriedade"].isin(filtro_tipo_propriedade)]
            if filtro_bairro_busca:
                filtrado = filtrado[filtrado["bairro_padronizado"].isin(filtro_bairro_busca)]

            if filtrado.empty:
                st.warning("Nenhum imóvel real corresponde a esses filtros. Tente remover algum filtro.")
            else:
                if objetivo == "Maior rentabilidade possível":
                    top5 = filtrado.sort_values("rentabilidade_anual_prevista", ascending=False).head(5)
                elif objetivo == "Maior preço de diária possível":
                    top5 = filtrado.sort_values("preco_previsto", ascending=False).head(5)
                else:
                    alvo_frac = ocupacao_alvo_pct / 100
                    filtrado = filtrado.copy()
                    filtrado["dist_ocupacao_alvo"] = (filtrado["ocupacao_prevista"] - alvo_frac).abs()
                    top5 = filtrado.sort_values(
                        ["dist_ocupacao_alvo", "rentabilidade_anual_prevista"], ascending=[True, False]
                    ).head(5)

                erro_preco_busca = modelos.get("erro_preco_reais")
                erro_ocup_busca = modelos.get("erro_ocupacao")
                tem_margem_busca = erro_preco_busca is not None and erro_ocup_busca is not None

                st.success(f"Top {len(top5)} imóveis reais mais próximos do objetivo escolhido:")
                for posicao, (_, linha) in enumerate(top5.iterrows(), start=1):
                    with st.container(border=True):
                        st.markdown(
                            f"**#{posicao} — {linha.get('bairro_padronizado', 'Bairro desconhecido')}** · "
                            f"{capitalizar_primeira(linha.get('tipo_propriedade', '-'))} · "
                            f"{capitalizar_primeira(linha.get('tipo_quarto', '-'))}"
                        )
                        cc1, cc2, cc3, cc4 = st.columns(4)
                        if tem_margem_busca:
                            cc1.metric("Preço/diária previsto",
                                       f"R$ {linha['preco_previsto']:.2f} (± R$ {erro_preco_busca:.2f})")
                            cc2.metric("Ocupação prevista",
                                       f"{linha['ocupacao_prevista']:.1%} (± {erro_ocup_busca:.1%})")
                        else:
                            cc1.metric("Preço/diária previsto", f"R$ {linha['preco_previsto']:.2f}")
                            cc2.metric("Ocupação prevista", f"{linha['ocupacao_prevista']:.1%}")
                        cc3.metric("Rentabilidade anual prevista", f"R$ {linha['rentabilidade_anual_prevista']:,.2f}")
                        cc4.metric(
                            "Capacidade / quartos",
                            f"{int(linha.get('capacidade_hospedes', 0))} hóspedes · "
                            f"{int(linha.get('quartos', 0))} qts"
                        )
                if not tem_margem_busca:
                    st.caption(
                        "ℹ️ Margem de erro não disponível neste `config_preditores.joblib` "
                        "(gere de novo com a versão atualizada do notebook `05-1` para vê-la aqui)."
                    )


# ---------------------------------------------------------------------------
# ABA 4 — ANÁLISE DE AVALIAÇÕES (pontos fortes e fracos)
# Esta aba já era declarada em st.tabs() mas não tinha bloco `with aba_avaliacoes:`
# — ficava vazia. Se o dataset tiver sub-notas por aspecto (limpeza, comunicação,
# localização etc.), usamos elas; senão, caímos de volta para nota_composta e os
# indicadores de engajamento já calculados em criar_variaveis_avaliacoes().
# ---------------------------------------------------------------------------
with aba_avaliacoes:
    st.subheader("📝 Pontos fortes e fracos das hospedagens")
    st.caption(
        "Usa as sub-notas de avaliação (limpeza, comunicação, localização etc.), quando "
        "disponíveis no dataset, para identificar em que aspecto cada bairro se destaca "
        "ou precisa melhorar. Quando essas sub-notas não estão no arquivo de features, a "
        "análise cai de volta para a nota composta e os indicadores de engajamento."
    )

    # --- Ranking de anúncios individuais (não só agregado por bairro) ---
    if {"nota_composta", "id_anuncio"}.issubset(df.columns):
        st.markdown("##### 🏆 Anúncios mais e menos bem avaliados")
        n_min_avaliacoes_rank = st.slider(
            "Mínimo de avaliações por mês para entrar no ranking (evita anúncios com poucos dados)",
            0.0, 5.0, 0.5, 0.1, key="rank_min_avaliacoes",
        )
        base_rank = df.copy()
        base_rank["nome_exibicao"] = obter_coluna_nome_anuncio(base_rank)
        if "avaliacoes_por_mes" in base_rank.columns:
            base_rank = base_rank[base_rank["avaliacoes_por_mes"] >= n_min_avaliacoes_rank]

        colunas_card = [c for c in ["nome_exibicao", "bairro_padronizado", "nota_composta",
                                     "preco", "avaliacoes_por_mes"] if c in base_rank.columns]

        col_top1, col_top2 = st.columns(2)
        with col_top1:
            st.success("**Top 10 melhores avaliados**")
            melhores = base_rank.sort_values("nota_composta", ascending=False).head(10)[colunas_card]
            st.dataframe(
                melhores.rename(columns={
                    "nome_exibicao": "Anúncio", "bairro_padronizado": "Bairro",
                    "nota_composta": "Nota", "preco": "Preço (R$)",
                    "avaliacoes_por_mes": "Aval./mês",
                }).style.format({"Nota": "{:.2f}", "Preço (R$)": "{:.2f}", "Aval./mês": "{:.2f}"}),
                hide_index=True, use_container_width=True,
            )
        with col_top2:
            st.warning("**Top 10 piores avaliados**")
            piores = base_rank.sort_values("nota_composta", ascending=True).head(10)[colunas_card]
            st.dataframe(
                piores.rename(columns={
                    "nome_exibicao": "Anúncio", "bairro_padronizado": "Bairro",
                    "nota_composta": "Nota", "preco": "Preço (R$)",
                    "avaliacoes_por_mes": "Aval./mês",
                }).style.format({"Nota": "{:.2f}", "Preço (R$)": "{:.2f}", "Aval./mês": "{:.2f}"}),
                hide_index=True, use_container_width=True,
            )
        st.caption(
            "💡 Anúncios com poucas avaliações têm notas mais instáveis (uma única "
            "avaliação de 5 estrelas já garante nota máxima) — por isso o filtro de "
            "avaliações mínimas por mês acima."
        )
        st.divider()

    CANDIDATOS_SUBNOTAS = {
        "review_scores_accuracy": "Precisão do anúncio",
        "review_scores_cleanliness": "Limpeza",
        "review_scores_checkin": "Check-in",
        "review_scores_communication": "Comunicação",
        "review_scores_location": "Localização",
        "review_scores_value": "Custo-benefício",
        "nota_precisao": "Precisão do anúncio",
        "nota_limpeza": "Limpeza",
        "nota_checkin": "Check-in",
        "nota_comunicacao": "Comunicação",
        "nota_localizacao": "Localização",
        "nota_custo_beneficio": "Custo-benefício",
    }
    subnotas_disponiveis = {c: r for c, r in CANDIDATOS_SUBNOTAS.items() if c in df.columns}

    if subnotas_disponiveis:
        # --- visão geral: qual aspecto puxa a média pra baixo/cima em toda a cidade
        medias_gerais = (
            df[list(subnotas_disponiveis)].mean().rename(index=subnotas_disponiveis).sort_values()
        )
        fig_geral = px.bar(
            medias_gerais, orientation="h",
            labels={"value": "Nota média", "index": ""},
            title="Nota média por aspecto — cidade toda",
            color=medias_gerais.values, color_continuous_scale="RdYlGn",
        )
        fig_geral.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_geral, use_container_width=True)
        st.caption(
            f"🔴 Ponto fraco geral: **{medias_gerais.index[0]}** (menor média) · "
            f"🟢 Ponto forte geral: **{medias_gerais.index[-1]}** (maior média)."
        )

        st.divider()
        st.markdown("##### Comparar um bairro com a média da cidade")
        n_min_anuncios_sub = st.slider(
            "Mínimo de anúncios por bairro (evita bairros com poucos dados)",
            1, 30, 5, key="sub_min_anuncios",
        )
        contagem_bairro = df.groupby("bairro_padronizado")["id_anuncio"].count()
        bairros_disponiveis = sorted(
            contagem_bairro[contagem_bairro >= n_min_anuncios_sub].index.tolist()
        )
        if not bairros_disponiveis:
            st.warning("Nenhum bairro atinge esse mínimo de anúncios — reduza o filtro.")
        else:
            bairro_escolhido = st.selectbox("Bairro", bairros_disponiveis)

            media_cidade = df[list(subnotas_disponiveis)].mean()
            media_bairro = df.loc[
                df["bairro_padronizado"] == bairro_escolhido, list(subnotas_disponiveis)
            ].mean()
            delta = (media_bairro - media_cidade).rename(index=subnotas_disponiveis).sort_values()

            col_rad1, col_rad2 = st.columns(2)
            with col_rad1:
                categorias_radar = list(subnotas_disponiveis.values())
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=media_cidade.rename(index=subnotas_disponiveis)[categorias_radar].values,
                    theta=categorias_radar, fill="toself", name="Cidade (média)",
                    line_color="#9aa0a6",
                ))
                fig_radar.add_trace(go.Scatterpolar(
                    r=media_bairro.rename(index=subnotas_disponiveis)[categorias_radar].values,
                    theta=categorias_radar, fill="toself", name=bairro_escolhido,
                    line_color="#ff6e40",
                ))
                fig_radar.update_layout(
                    title=f"{bairro_escolhido} vs. cidade — por aspecto",
                    polar=dict(radialaxis=dict(visible=True)),
                    showlegend=True, legend=dict(orientation="h", y=-0.1),
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            with col_rad2:
                fig_delta = px.bar(
                    delta, orientation="h",
                    labels={"value": "Diferença em relação à média da cidade", "index": ""},
                    title=f"{bairro_escolhido}: acima/abaixo da média (diferença absoluta)",
                    color=delta.values, color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                )
                fig_delta.update_layout(showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig_delta, use_container_width=True)

            pontos_fracos = delta[delta < -0.05].index.tolist()
            pontos_fortes = delta[delta > 0.05].index.tolist()
            col_pf1, col_pf2 = st.columns(2)
            with col_pf1:
                st.success(
                    f"**Pontos fortes:** {', '.join(pontos_fortes) if pontos_fortes else 'nenhum destaque relevante'}"
                )
            with col_pf2:
                st.warning(
                    f"**Pontos de atenção:** {', '.join(pontos_fracos) if pontos_fracos else 'nenhum problema relevante'}"
                )

        st.divider()
        st.markdown("##### O que mais pesa no preço e na ocupação?")
        colunas_corr = list(subnotas_disponiveis)
        alvo_disponivel = [c for c in ["preco", "taxa_ocupacao_estimada"] if c in df.columns]
        if colunas_corr and alvo_disponivel:
            corr = df[colunas_corr + alvo_disponivel].corr().loc[colunas_corr, alvo_disponivel]
            corr.index = [subnotas_disponiveis[c] for c in colunas_corr]
            st.dataframe(
                corr.style.format("{:.2f}").background_gradient(cmap="RdYlGn", axis=None),
                use_container_width=True,
            )
            st.caption(
                "Correlação entre cada aspecto avaliado e preço/ocupação. Valores mais "
                "próximos de 1 (verde) indicam que melhorar aquele aspecto tende a andar "
                "junto com preços mais altos ou mais ocupação; próximos de -1 (vermelho), "
                "o oposto — não implica causalidade."
            )
    else:
        st.info(
            "O dataset de features não traz sub-notas por aspecto (limpeza, comunicação, "
            "localização etc.) — só a nota composta. A análise abaixo usa o que está "
            "disponível: nota composta, volume/frequência de avaliações e o índice de "
            "engajamento já calculados no carregamento dos dados."
        )

        st.markdown("##### Distribuição da qualidade percebida")
        if "qualidade_percebida" in df.columns:
            contagem_qualidade = df["qualidade_percebida"].value_counts().reset_index()
            contagem_qualidade.columns = ["qualidade_percebida", "quantidade"]
            fig_qual = px.bar(
                contagem_qualidade, x="qualidade_percebida", y="quantidade",
                title="Quantidade de anúncios por faixa de qualidade percebida",
                color="qualidade_percebida",
            )
            st.plotly_chart(fig_qual, use_container_width=True)

        st.divider()
        st.markdown("##### Bairros: destaques e pontos de atenção")
        if {"bairro_padronizado", "nota_composta", "id_anuncio"}.issubset(df.columns):
            agregacoes = {
                "nota_media": ("nota_composta", "mean"),
                "qtd_anuncios": ("id_anuncio", "count"),
            }
            if "indice_engajamento" in df.columns:
                agregacoes["engajamento_medio"] = ("indice_engajamento", "mean")

            ranking_bairro = (
                df.groupby("bairro_padronizado").agg(**agregacoes).dropna().sort_values("nota_media")
            )
            n_min_anuncios = st.slider(
                "Mínimo de anúncios por bairro (evita bairros com poucos dados)", 1, 30, 5
            )
            ranking_filtrado = ranking_bairro[ranking_bairro["qtd_anuncios"] >= n_min_anuncios]
            formato_cols = {"nota_media": "{:.2f}"}
            if "engajamento_medio" in ranking_filtrado.columns:
                formato_cols["engajamento_medio"] = "{:.2f}"

            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.warning("**Bairros de atenção (nota média mais baixa):**")
                st.dataframe(ranking_filtrado.head(5).style.format(formato_cols))
            with col_r2:
                st.success("**Bairros destaque (nota média mais alta):**")
                st.dataframe(
                    ranking_filtrado.tail(5).sort_values("nota_media", ascending=False)
                    .style.format(formato_cols)
                )

        st.divider()
        st.markdown("##### Bem avaliados x pouco avaliados")
        if {"nota_composta", "avaliacoes_por_mes"}.issubset(df.columns):
            fig_disp = px.scatter(
                df, x="avaliacoes_por_mes", y="nota_composta",
                color="qualidade_percebida" if "qualidade_percebida" in df.columns else None,
                hover_data=["bairro_padronizado"] if "bairro_padronizado" in df.columns else None,
                title="Nota composta x frequência de avaliações",
                labels={"avaliacoes_por_mes": "Avaliações por mês", "nota_composta": "Nota composta"},
            )
            st.plotly_chart(fig_disp, use_container_width=True)
            st.caption(
                "Cantos úteis pra ler: **canto superior direito** = bem avaliados e "
                "populares (referência do bairro); **canto superior esquerdo** = bem "
                "avaliados mas pouco visitados (oportunidade de divulgação); **canto "
                "inferior direito** = populares mas mal avaliados (risco — vale investigar)."
            )


# ---------------------------------------------------------------------------
# ABA 5 — RECOMENDAÇÃO POR PONTOS TURÍSTICOS
# ---------------------------------------------------------------------------
with aba_recomendacao:
    st.subheader("🎯 Encontre hospedagens perto dos lugares que você quer visitar")
    st.caption(
        "Escolha um ou mais pontos turísticos e o app calcula, para cada anúncio real do "
        "dataset, a distância até o ponto mais próximo dentre os selecionados — e sugere "
        "as melhores opções considerando também preço e nota."
    )

    if pontos_turisticos.empty:
        st.warning(
            "Nenhum ponto turístico cadastrado no momento. Esta aba fica desabilitada."
        )
    else:
        nomes_pontos = sorted(pontos_turisticos["nome"].dropna().unique().tolist())
        pontos_escolhidos = st.multiselect(
            "Pontos turísticos de interesse",
            nomes_pontos,
            default=nomes_pontos[:1] if nomes_pontos else [],
            help="Se escolher mais de um ponto, a distância considerada para cada anúncio "
                 "é a menor entre ele e qualquer um dos pontos selecionados.",
        )

        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            raio_max_km = st.slider("Distância máxima aceitável (km)", 0.5, 20.0, 3.0, 0.5)
        with col_r2:
            preco_max_rec = st.number_input(
                "Preço máximo por diária (R$)", min_value=0.0,
                value=float(df["preco"].quantile(0.9)), step=50.0,
            )
        with col_r3:
            peso_distancia = st.slider(
                "Peso da proximidade no ranking (0 = só preço/nota, 1 = só distância)",
                0.0, 1.0, 0.6, 0.1,
            )

        if not pontos_escolhidos:
            st.info("Selecione ao menos um ponto turístico para ver as recomendações.")
        else:
            subset_pontos = pontos_turisticos[pontos_turisticos["nome"].isin(pontos_escolhidos)].reset_index(drop=True)

            colunas_necessarias_rec = [c for c in ["latitude", "longitude", "preco"] if c in df.columns]
            base = df.dropna(subset=colunas_necessarias_rec).copy()
            base["nome_exibicao"] = obter_coluna_nome_anuncio(base)

            # distância mínima de cada anúncio a qualquer um dos pontos selecionados
            distancias = np.column_stack([
                haversine_km(base["latitude"].values, base["longitude"].values, pt.lat, pt.lon)
                for pt in subset_pontos.itertuples()
            ])
            base["distancia_km"] = distancias.min(axis=1)
            base["ponto_mais_proximo"] = subset_pontos["nome"].values[distancias.argmin(axis=1)]

            filtrado = base[
                (base["distancia_km"] <= raio_max_km) & (base["preco"] <= preco_max_rec)
            ].copy()

            if filtrado.empty:
                st.warning(
                    "Nenhum anúncio atende aos filtros de distância/preço. "
                    "Tente aumentar o raio ou o preço máximo."
                )
            else:
                def normalizar(serie, inverter=False):
                    minimo, maximo = serie.min(), serie.max()
                    if maximo == minimo:
                        return pd.Series(0.5, index=serie.index)
                    norm = (serie - minimo) / (maximo - minimo)
                    return 1 - norm if inverter else norm

                dist_norm = normalizar(filtrado["distancia_km"], inverter=True)  # mais perto = melhor
                preco_norm = normalizar(filtrado["preco"], inverter=True)        # mais barato = melhor
                nota_norm = (
                    normalizar(filtrado["nota_composta"])
                    if "nota_composta" in filtrado.columns
                    else pd.Series(0.5, index=filtrado.index)
                )

                peso_preco_nota = (1 - peso_distancia) / 2
                filtrado["score_recomendacao"] = (
                    peso_distancia * dist_norm
                    + peso_preco_nota * preco_norm
                    + peso_preco_nota * nota_norm
                )

                top_n = filtrado.sort_values("score_recomendacao", ascending=False).head(10)

                st.success(
                    f"{len(filtrado)} anúncios dentro do raio/preço escolhidos — "
                    "mostrando os 10 melhores pelo score combinado (distância + preço + nota)."
                )

                for posicao, (_, linha) in enumerate(top_n.iterrows(), start=1):
                    with st.container(border=True):
                        link_anuncio_card = linha.get("listing_url")
                        link_card_md = (
                            f"[Ver anúncio no Airbnb]({link_anuncio_card})"
                            if pd.notna(link_anuncio_card) else "Link indisponível"
                        )
                        st.markdown(
                            f"**#{posicao} — {linha['nome_exibicao']}**\n\n"
                            f"📍 {linha.get('bairro_padronizado', 'Bairro desconhecido')} · "
                            f"{capitalizar_primeira(linha.get('tipo_propriedade', '-'))} · "
                            f"{capitalizar_primeira(linha.get('tipo_quarto', '-'))} · "
                            f"perto de *{linha['ponto_mais_proximo']}* · {link_card_md}"
                        )
                        rc1, rc2, rc3, rc4 = st.columns(4)
                        rc1.metric("Preço/diária", f"R$ {linha['preco']:.2f}")
                        rc2.metric("Distância", f"{linha['distancia_km']:.2f} km")
                        if "nota_composta" in top_n.columns:
                            rc3.metric("Nota", f"{linha['nota_composta']:.2f}")
                        rc4.metric("Score", f"{linha['score_recomendacao']:.2f}")

                st.divider()
                st.markdown("##### 🗺️ Mapa das recomendações")
                m_rec = folium.Map(
                    location=[subset_pontos["lat"].mean(), subset_pontos["lon"].mean()],
                    zoom_start=13, tiles="CartoDB positron",
                )
                for pt in subset_pontos.itertuples():
                    folium.Marker(
                        location=[pt.lat, pt.lon], popup=pt.nome,
                        icon=folium.Icon(color="cadetblue", icon="star", prefix="fa"),
                    ).add_to(m_rec)
                for posicao, (_, linha) in enumerate(top_n.iterrows(), start=1):
                    link_anuncio = linha.get("listing_url")
                    link_html = (
                        f'<a href="{link_anuncio}" target="_blank">Ver anúncio no Airbnb</a>'
                        if pd.notna(link_anuncio) else "Link indisponível"
                    )
                    popup_html_rec = (
                        f"<b>#{posicao}</b><br>"
                        f"{linha['nome_exibicao']}<br>"
                        f"R$ {linha['preco']:.2f} / diária<br>"
                        f"{link_html}"
                    )
                    folium.CircleMarker(
                        location=[linha["latitude"], linha["longitude"]],
                        radius=8, color="#1e3d59", fill=True, fill_color="#ff6e40",
                        fill_opacity=0.85,
                        popup=folium.Popup(popup_html_rec, max_width=250),
                    ).add_to(m_rec)
                st_folium(m_rec, width=None, height=500, returned_objects=[])

                st.caption(
                    "💡 O score combinado normaliza distância, preço e nota entre 0 e 1 "
                    "dentro do conjunto filtrado, e faz uma média ponderada pelos pesos "
                    "escolhidos acima. É um ranking relativo aos anúncios que passaram "
                    "pelos filtros de distância máxima e preço máximo — não um valor absoluto."
                )


# ---------------------------------------------------------------------------
# ABA 6 — ASSISTENTE IA
# Chat livre (Claude, via API da Anthropic), mas com o resumo do dataset
# injetado no system prompt para que a assistente também consiga responder
# perguntas sobre os dados carregados no app.
# ---------------------------------------------------------------------------
with aba_assistente:
    st.subheader("🤖 Assistente IA")
    st.caption(
        "Converse livremente ou pergunte sobre os dados deste app (bairros, preços, "
        "rentabilidade, avaliações etc.). A assistente tem acesso a um resumo do dataset "
        "carregado, mas não aos filtros que você aplicou nas outras abas."
    )

    if not ANTHROPIC_DISPONIVEL:
        st.error(
            "A biblioteca `anthropic` não está instalada. Adicione `anthropic` ao "
            "requirements.txt e reinstale as dependências."
        )
    elif "ANTHROPIC_API_KEY" not in st.secrets:
        st.warning(
            "Nenhuma chave de API encontrada. Crie o arquivo `.streamlit/secrets.toml` com:"
        )
        st.code('ANTHROPIC_API_KEY = "sk-ant-..."', language="toml")
        st.caption(
            "No Streamlit Community Cloud, configure em **App settings → Secrets**. "
            "A chave é gerada em https://console.anthropic.com."
        )
    else:
        cliente_ia = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
        contexto_dados = montar_contexto_dados(df, agg)

        system_prompt = (
            "Você é a assistente de IA embutida em um app Streamlit de análise do mercado "
            "de Airbnb no Rio de Janeiro. Responda em português do Brasil, de forma direta "
            "e objetiva. Use o resumo de dados abaixo quando a pergunta for sobre o "
            "dataset; para perguntas gerais, responda normalmente sem forçar o contexto.\n\n"
            f"### Resumo do dataset carregado\n{contexto_dados}"
        )

        if "mensagens_assistente" not in st.session_state:
            st.session_state.mensagens_assistente = []

        col_limpar, _ = st.columns([1, 4])
        with col_limpar:
            if st.button("🗑️ Limpar conversa"):
                st.session_state.mensagens_assistente = []
                st.rerun()

        for msg in st.session_state.mensagens_assistente:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        pergunta = st.chat_input("Pergunte algo sobre os dados ou converse livremente...")
        if pergunta:
            st.session_state.mensagens_assistente.append({"role": "user", "content": pergunta})
            with st.chat_message("user"):
                st.markdown(pergunta)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                resposta_completa = ""
                try:
                    with cliente_ia.messages.stream(
                        model="claude-sonnet-4-6",
                        max_tokens=1024,
                        system=system_prompt,
                        messages=st.session_state.mensagens_assistente,
                    ) as stream:
                        for texto in stream.text_stream:
                            resposta_completa += texto
                            placeholder.markdown(resposta_completa + "▌")
                    placeholder.markdown(resposta_completa)
                except Exception as e:
                    resposta_completa = f"Erro ao consultar a API da Anthropic: {e}"
                    placeholder.error(resposta_completa)

            st.session_state.mensagens_assistente.append(
                {"role": "assistant", "content": resposta_completa}
            )