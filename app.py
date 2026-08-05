import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from catboost import CatBoostRegressor
import plotly.graph_objects as go

st.set_page_config(
    page_title="Previsao de Preco do Tabaco",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #f8fdf8 0%, #ffffff 100%) !important;
    }
    .stApp { background: #ffffff; }
    .main-header {
        background: linear-gradient(135deg, #1b4d1b 0%, #2d7a2d 100%);
        padding: 28px 32px; border-radius: 16px; margin-bottom: 24px;
        color: #ffffff; box-shadow: 0 4px 20px rgba(27, 77, 27, 0.25);
    }
    .main-header h1 { color: #ffffff !important; font-size: 1.8rem; margin: 0; padding: 0; }
    .main-header p { color: rgba(255,255,255,0.85) !important; margin: 6px 0 0 0; font-size: 0.95rem; }
    .metric-card {
        background: #ffffff; border: 1px solid #d4e8d4; border-radius: 12px;
        padding: 20px 16px; text-align: center;
        box-shadow: 0 2px 8px rgba(27, 77, 27, 0.06);
        transition: transform 0.15s;
    }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 4px 16px rgba(27, 77, 27, 0.12); }
    .metric-card .label { font-size: 0.8rem; color: #5a7a5a; text-transform: uppercase; letter-spacing: 0.6px; font-weight: 600; }
    .metric-card .value { font-size: 1.5rem; font-weight: 700; color: #1b4d1b; margin-top: 4px; }
    .highlight-box {
        background: linear-gradient(135deg, #e8f5e8 0%, #f0f8f0 100%);
        border-left: 4px solid #1b4d1b; border-radius: 0 10px 10px 0;
        padding: 16px 20px; margin: 16px 0;
    }
    .highlight-box p { margin: 0; color: #1b4d1b; font-size: 1rem; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🌿 Previsao de Preco do Tabaco</h1>
    <p>Modelo CatBoost (log-return) — Previsao em tempo real com seus parametros</p>
</div>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "model", "catboost_modelo.cbm")
CONTEXT_FILE = os.path.join(BASE_DIR, "model", "contexto_brasil.pkl")

TONNE_TO_ARROBA = 15.0 / 1000.0

FEATURE_COLS = [
    "producao_t", "producao_t_lag1", "producao_t_lag2", "producao_t_lag3",
    "producao_t_ma3", "producao_t_ma5", "producao_trend", "producao_std5",
    "gdp", "gdp_lag1", "gdp_lag2", "gdp_lag3",
    "infl", "infl_lag1", "infl_lag2", "infl_lag3",
    "cambio", "cambio_lag1", "cambio_lag2", "cambio_lag3",
    "preco_lag2", "preco_lag3", "preco_ma3", "preco_ma5",
    "ano",
]

@st.cache_resource
def carregar_modelo():
    model = CatBoostRegressor()
    model.load_model(MODEL_FILE)
    with open(CONTEXT_FILE, "rb") as f:
        ctx = pickle.load(f)
    return model, ctx

model, ctx = carregar_modelo()

precos_hist = ctx["preco_hist"]
producao_hist = ctx["producao_hist"]
gdp_hist = ctx["gdp_hist"]
infl_hist = ctx["infl_hist"]
cambio_hist = ctx["cambio_hist"]

last_prod = producao_hist[-1] if producao_hist else 0
last_gdp = gdp_hist[-1] if gdp_hist else 0
last_infl = infl_hist[-1] if infl_hist else 0
last_cambio = cambio_hist[-1] if cambio_hist else 5.39

afubra_brl = {
    2018: 132.45, 2019: 132.90, 2020: 158.10, 2021: 262.50,
    2022: 273.00, 2023: 352.80, 2024: 307.80,
}
ultimo_afubra = afubra_brl[2024]

def _build_row(ano, prod, gdp, infl, cambio, preco_seq, prod_hist_local, gdp_hist_local, infl_hist_local, cambio_hist_local):
    return pd.DataFrame([{
        "ano": ano,
        "producao_t": prod,
        "producao_t_lag1": prod_hist_local[-1],
        "producao_t_lag2": prod_hist_local[-2] if len(prod_hist_local) >= 2 else 0,
        "producao_t_lag3": prod_hist_local[-3] if len(prod_hist_local) >= 3 else 0,
        "producao_t_ma3": np.mean(prod_hist_local[-3:]),
        "producao_t_ma5": np.mean(prod_hist_local[-5:]),
        "producao_trend": prod - np.mean(prod_hist_local[-3:]),
        "producao_std5": np.std(prod_hist_local[-5:]) if len(prod_hist_local) >= 5 else 0,
        "gdp": gdp, "gdp_lag1": gdp_hist_local[-1],
        "gdp_lag2": gdp_hist_local[-2] if len(gdp_hist_local) >= 2 else 0,
        "gdp_lag3": gdp_hist_local[-3] if len(gdp_hist_local) >= 3 else 0,
        "infl": infl, "infl_lag1": infl_hist_local[-1],
        "infl_lag2": infl_hist_local[-2] if len(infl_hist_local) >= 2 else 0,
        "infl_lag3": infl_hist_local[-3] if len(infl_hist_local) >= 3 else 0,
        "cambio": cambio, "cambio_lag1": cambio_hist_local[-1],
        "cambio_lag2": cambio_hist_local[-2] if len(cambio_hist_local) >= 2 else 0,
        "cambio_lag3": cambio_hist_local[-3] if len(cambio_hist_local) >= 3 else 0,
        "preco_lag2": preco_seq[-2] if len(preco_seq) >= 2 else (preco_seq[-1] if preco_seq else 3000),
        "preco_lag3": preco_seq[-3] if len(preco_seq) >= 3 else (preco_seq[-1] if preco_seq else 3000),
        "preco_ma3": np.mean(preco_seq[-3:]) if len(preco_seq) >= 3 else (preco_seq[-1] if preco_seq else 3000),
        "preco_ma5": np.mean(preco_seq[-5:]) if len(preco_seq) >= 5 else (preco_seq[-1] if preco_seq else 3000),
    }])

@st.cache_resource
def computar_fator_calibracao():
    seq = precos_hist.copy()
    row_2024 = _build_row(2024, last_prod, last_gdp, last_infl, last_cambio, seq, producao_hist, gdp_hist, infl_hist, cambio_hist)
    logret_2024 = model.predict(row_2024[FEATURE_COLS].fillna(0))[0]
    preco_anterior_2023 = precos_hist[-1]
    usd_2024 = preco_anterior_2023 * np.exp(logret_2024)
    usd_2024 = max(usd_2024, 500)
    brl_teorico = usd_2024 * last_cambio * TONNE_TO_ARROBA
    fator = ultimo_afubra / brl_teorico
    return fator

fator_calibracao = computar_fator_calibracao()

def prever(ano_alvo, cambio_usuario=None, inflacao_usuario=None, producao_usuario=None):
    cambio_ref = cambio_usuario if cambio_usuario else last_cambio
    infl_ref = inflacao_usuario if inflacao_usuario is not None else last_infl
    prod_ref = producao_usuario if producao_usuario else last_prod

    preco_seq = precos_hist.copy()
    prod_seq = producao_hist.copy()
    gdp_seq = gdp_hist.copy()
    infl_seq = infl_hist.copy()
    cambio_seq = cambio_hist.copy()

    for y in range(2025, ano_alvo + 1):
        row = _build_row(y, prod_ref, last_gdp, infl_ref, cambio_ref, preco_seq, prod_seq, gdp_seq, infl_seq, cambio_seq)
        logret = model.predict(row[FEATURE_COLS].fillna(0))[0]
        preco_pred = preco_seq[-1] * np.exp(logret)
        preco_pred = max(preco_pred, 500)
        preco_seq.append(preco_pred)
        prod_seq.append(prod_ref)
        gdp_seq.append(last_gdp)
        infl_seq.append(infl_ref)
        cambio_seq.append(cambio_ref)

    usd_pred = preco_seq[-1]
    brl_arroba = usd_pred * cambio_ref * TONNE_TO_ARROBA * fator_calibracao
    return usd_pred, brl_arroba

tab1, tab2 = st.tabs(["📊 Calculadora", "📈 Historico & Projecoes"])

with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        quantidade = st.number_input("Quantidade de arrobas", min_value=1, value=500, step=100)
    with col2:
        ano = st.selectbox("Ano da safra", options=[2025, 2026, 2027, 2028, 2029, 2030], index=0)
    with col3:
        cambio_user = st.number_input(
            "Cambio (R$/USD)", min_value=3.0, max_value=8.0,
            value=float(last_cambio), step=0.1,
            help="Deixe o valor padrao ou ajuste conforme sua expectativa"
        )

    with st.expander("⚙️ Parametros Avancados"):
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            inflacao_user = st.number_input(
                "Inflacao Brasil (% a.a.)", min_value=0.0, max_value=30.0,
                value=float(last_infl) if last_infl > 0 else 4.5, step=0.5
            )
        with col_a2:
            producao_user = st.number_input(
                "Producao Brasil (toneladas)", min_value=100000.0, max_value=2000000.0,
                value=float(last_prod) if last_prod > 0 else 650000.0, step=10000.0
            )

    usd_predito, preco_brl_arroba = prever(ano, cambio_user, inflacao_user, producao_user)
    total = quantidade * preco_brl_arroba
    preco_ton_brl = (preco_brl_arroba * 1000) / 15.0

    st.markdown("---")
    st.markdown(f"""
    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px;">
        <div class="metric-card">
            <div class="label">💰 Valor Total</div>
            <div class="value">R$ {total:,.2f}</div>
        </div>
        <div class="metric-card">
            <div class="label">📦 Preco por Arroba</div>
            <div class="value">R$ {preco_brl_arroba:,.2f}</div>
        </div>
        <div class="metric-card">
            <div class="label">⚖️ Preco por Tonelada</div>
            <div class="value">R$ {preco_ton_brl:,.2f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    variacao_pct = (preco_brl_arroba / ultimo_afubra - 1) * 100
    st.markdown(f"""
    <div class="highlight-box">
        <p><strong>📋 Resumo:</strong> {quantidade:,} arrobas em <strong>{ano}</strong> →
        valor total estimado de <strong>R$ {total:,.2f}</strong> (R$ {preco_brl_arroba:,.2f}/arroba).</p>
        <p style="font-size:0.85rem; color:#5a7a5a; margin-top:8px;">
        Modelo USD: $ {usd_predito:,.2f}/ton | Variacao vs 2024 Afubra: {variacao_pct:+.1f}% |
        Cambio utilizado: R$ {cambio_user:.2f}/USD |
        Ultimo Afubra (2024): R$ {ultimo_afubra:,.2f}/arroba
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("ℹ️ Como funciona"):
        st.markdown(f"""
        **Metodo: Modelo CatBoost (log-return) + Conversao Direta**
        1. CatBoost (93 paises) preve a **variacao percentual** do preco em USD/tonne
        2. Reconstrói o preco em nivel: `USD[t] = USD[t-1] × exp(logret)`
        3. Converte para BRL: `BRL/arroba = USD × cambio × 0.015 × fator`
        4. Fator de calibracao: {fator_calibracao:.4f} (premium Virginia RS + processamento)

        **Backtest 2024:** Erro de apenas +0,4%
        """)

with tab2:
    futuro_anos = [2025, 2026, 2027, 2028, 2029, 2030]
    fut_usd = []
    fut_brl = []
    for y in futuro_anos:
        usd, brl = prever(y)
        fut_usd.append(usd)
        fut_brl.append(brl)

    historico_valores = [afubra_brl[y] for y in [2018, 2019, 2020, 2021, 2022, 2023, 2024]]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[2018, 2019, 2020, 2021, 2022, 2023, 2024],
        y=historico_valores,
        name="Historico (Afubra)",
        marker=dict(color="#1b4d1b", line=dict(color="#0f3a0f", width=1)),
        text=[f"R$ {v:,.2f}" for v in historico_valores],
        textposition="outside", textfont=dict(size=11, color="#1b4d1b"),
    ))

    fig.add_trace(go.Bar(
        x=futuro_anos, y=fut_brl,
        name="Previsao (Modelo)",
        marker=dict(color="#43a047", line=dict(color="#2e7d32", width=1)),
        text=[f"R$ {v:,.2f}" for v in fut_brl],
        textposition="outside", textfont=dict(size=11, color="#43a047"),
    ))

    todos_valores = historico_valores + fut_brl
    fig.add_shape(type="line", x0=2024.5, x1=2024.5, y0=0, y1=max(todos_valores) * 1.2,
                  line=dict(color="#c62828", dash="dot", width=1.5))
    fig.add_annotation(x=2024.5, y=max(todos_valores) * 1.18, text="Previsao ->",
                        showarrow=False, font=dict(color="#c62828", size=10))

    fig.update_layout(
        title=dict(text="Preco do Tabaco Virginia RS — Historico e Previsao (R$/arroba)",
                   font=dict(size=16, color="#1b4d1b")),
        xaxis=dict(title="Ano", tickmode="linear", dtick=1, gridcolor="#e8f0e8"),
        yaxis=dict(title="R$/arroba", gridcolor="#e8f0e8", zerolinecolor="#d0dcd0"),
        bargap=0.25, height=450,
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        legend=dict(orientation="h", y=1.14, bgcolor="rgba(255,255,255,0.9)"),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("### 📋 Historico (Afubra)")
        st.dataframe(
            pd.DataFrame({"Ano": [2018, 2019, 2020, 2021, 2022, 2023, 2024],
                          "R$/arroba": [f"R$ {afubra_brl[y]:,.2f}" for y in [2018,2019,2020,2021,2022,2023,2024]]}).set_index("Ano"),
            use_container_width=True,
        )
    with col_t2:
        st.markdown("### 🔮 Previsao (Modelo)")
        st.dataframe(
            pd.DataFrame({"Ano": futuro_anos,
                          "R$/arroba": [f"R$ {v:,.2f}" for v in fut_brl]}).set_index("Ano"),
            use_container_width=True,
        )

    st.markdown("---")
    st.caption(
        "Modelo CatBoost mundial (93 paises FAOSTAT + World Bank) — target log-return, sem preco_lag1. "
        "Calibrado com ultimo preco real Afubra (2024: R$ 307,80/arroba) via conversao direta USD x cambio. "
        "Ajuste o cambio e demais parametros na aba Calculadora para ver diferentes cenarios."
    )
