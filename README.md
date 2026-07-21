# Airbnb Rio de Janeiro — Análise Espacial (App Streamlit)

App com 6 abas:

- 🗺️ **Mapa Dinâmico** — bairros coloridos por score de luxo, bolhas de rentabilidade/ocupação, pontos turísticos, com filtro por faixa de preço e por região popular/não popular.
- 📅 **Sazonalidade** — ocupação média por mês (ano cheio) e preço médio por mês (meses coletados), destacando alta e baixa temporada.
- 📈 **Evolução Histórica de Preços** — variação do preço da diária entre coletas (valores nominais, sem correção pela inflação/IPCA) e evolução do preço médio por bairro.
- 🧮 **Simulador de Investimento** — simulação de preço, ocupação e rentabilidade esperada, comparando com a média de um bairro de referência escolhido.
- 🎯 **Recomendação por Turismo** — sugestão de hospedagens perto dos pontos turísticos que o usuário quer visitar.
- 📝 **Análise de Avaliações** — pontos fortes e fracos por bairro a partir das sub-notas de avaliação (limpeza, comunicação, localização etc.), com fallback para nota composta e indicadores de engajamento quando essas sub-notas não estão disponíveis no dataset.

## Estrutura de pastas

```
.
├── app.py
├── requirements.txt
└── dados/
    ├── dataset_features_slim.parquet
    ├── neighbourhoods.geojson
    ├── calendar_agregado.parquet
    └── listings_temporal.parquet
```

Suba essa estrutura inteira (pasta `dados/` incluída) pro repositório do GitHub — o `app.py` espera achar os arquivos dentro de `dados/`, no mesmo nível dele.

## Como colocar no ar (Streamlit Community Cloud — gratuito)

1. Crie um repositório novo no GitHub e suba todos esses arquivos (pode ser público ou privado).
2. Acesse [streamlit.io/cloud](https://streamlit.io/cloud) e faça login com sua conta do GitHub.
3. Clique em **"New app"**, escolha o repositório, a branch (`main`) e o arquivo principal (`app.py`).
4. Clique em **Deploy**. Em 1-2 minutos o app sobe e gera um link tipo `seunome-airbnb-rio.streamlit.app`.

## Rodando localmente (pra testar antes de subir)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Abre automaticamente em `http://localhost:8501`.

## Observações importantes

- **Pontos turísticos (osmnx):** a primeira vez que o app carrega, ele busca os pontos turísticos do Rio no OpenStreetMap (precisa de internet). O resultado fica em cache (`@st.cache_data`) enquanto o app estiver no ar — não busca de novo a cada clique do usuário, só quando o servidor reinicia.
- **Se o deploy no Streamlit Cloud falhar por causa do `geopandas`/`osmnx`** (erro relacionado a GDAL): crie um arquivo `packages.txt` na raiz do repositório com o conteúdo abaixo — ele instala as dependências de sistema que faltam:

```
libgdal-dev
gdal-bin
```

- **Análise Fatorial e Modelagem Preditiva:** partes do código dependem de `factor_analyzer`, `scikit-learn`, `statsmodels` e `joblib`. Se algum desses pacotes não estiver instalado, o app usa fallback automático (`FACTOR_ANALYZER_DISPONIVEL` / `MODELAGEM_DISPONIVEL`) e simplesmente esconde os recursos que dependem deles, sem quebrar.
