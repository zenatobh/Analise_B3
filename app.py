"""
app.py — painel de ativos para celular (Streamlit + Plotly).

Local:   streamlit run app.py
Deploy:  Streamlit Community Cloud apontando para este arquivo.
"""
import pandas as pd
import streamlit as st

from analise_ativos import (DIAS_UTEIS_ANO, JANELA_PLOT, NAO_ACOES, TICKERS,
                            baixar_precos, moeda, serie, tabela_painel,
                            tabela_resumo)
from graficos_plotly import CONFIG, fig_ativo, fig_hist

st.set_page_config(page_title="Ativos B3", page_icon="📈", layout="centered")

# Períodos exibidos no gráfico principal (em pregões)
PERIODOS = {"1M": 21, "3M": 63, "6M": 126, "1A": 248, "2A": 504, "Tudo": None}
PERIODO_PADRAO = "1A"


@st.cache_data(ttl=60 * 60, show_spinner="Baixando cotações…")
def carregar():
    """Baixa uma vez por hora; todos os acessos reutilizam o cache."""
    return baixar_precos(TICKERS)


def mostrar_fig(fig):
    st.plotly_chart(fig, width="stretch", config=CONFIG)


def pct(x, casas=2):
    """Percentual no formato brasileiro, com sinal: +1,23%."""
    return f"{x:+.{casas}%}".replace(".", ",")


def sinal_moeda(x):
    return ("+" if x >= 0 else "") + moeda(x)


def cor_variacao(v):
    """Estilo de texto para variações: verde se positiva, vermelho se negativa."""
    if pd.isna(v) or v == 0:
        return ""
    return "color: #1a9850" if v > 0 else "color: #d73027"


# ----------------------------------------------------------------------------
# Dados
# ----------------------------------------------------------------------------
try:
    precos = carregar()
except Exception as e:  # rede, rate limit do Yahoo, ticker inválido
    st.error(f"Não foi possível baixar as cotações: {e}")
    if st.button("Tentar novamente"):
        carregar.clear()
        st.rerun()
    st.stop()

topo_esq, topo_dir = st.columns([3, 1])
topo_esq.caption(f"Dados até {precos.index[-1]:%d/%m/%Y}")
if topo_dir.button("↻", help="Atualizar dados agora"):
    carregar.clear()
    st.rerun()

aba_ativo, aba_painel, aba_resumo = st.tabs(["Ativo", "Painel", "Resumo"])

# ----------------------------------------------------------------------------
# Aba 1 — análise de um ativo
# ----------------------------------------------------------------------------
with aba_ativo:
    opcoes = list(precos.columns)
    ticker = st.selectbox("Ativo", opcoes,
                          index=opcoes.index("IBOV") if "IBOV" in opcoes else 0)

    s = serie(precos, ticker)
    j = s.tail(JANELA_PLOT)
    n_ano = min(len(s), DIAS_UTEIS_ANO)
    c1, c2, c3 = st.columns(3)
    c1.metric("Último", moeda(s.iloc[-1]), delta=pct(s.iloc[-1] / s.iloc[-2] - 1),
              help="Variação em relação ao pregão anterior")
    c2.metric("12 meses", pct(s.iloc[-1] / s.iloc[-n_ano] - 1),
              delta=sinal_moeda(s.iloc[-1] - s.iloc[-n_ano]),
              help="Variação em 12 meses; abaixo, a diferença em pontos/R$")
    c3.metric("vs. média 1A", f"{(s.iloc[-1] - j.mean()) / j.std():+.1f}σ".replace(".", ","),
              help="Distância do preço atual à média dos últimos 248 pregões, "
                   "em desvios-padrão (mesma referência das bandas do gráfico)")

    escolha = st.segmented_control("Período", list(PERIODOS), default=PERIODO_PADRAO,
                                   label_visibility="collapsed")
    periodo = PERIODOS[escolha or PERIODO_PADRAO] or len(s)

    with st.expander("Opções"):
        percentual = st.toggle("Difusor em %", value=False)

    mostrar_fig(fig_ativo(precos, ticker, period=periodo, percentual=percentual))
    st.caption("Arraste sobre o gráfico para dar zoom · toque duplo para voltar.")
    mostrar_fig(fig_hist(precos, ticker))

# ----------------------------------------------------------------------------
# Aba 2 — visão geral de todos os ativos
# ----------------------------------------------------------------------------
with aba_painel:
    painel = tabela_painel(precos, dias=JANELA_PLOT)
    st.caption(f"Faixa, média e σ dos últimos {JANELA_PLOT} pregões de cada ativo. "
               "Toque no nome de uma coluna para ordenar.")
    st.dataframe(
        painel.style.map(cor_variacao, subset=["Dia", "12 meses"]),
        width="stretch",
        height=35 * (len(painel) + 1) + 3,      # mostra todas as linhas sem rolagem
        column_config={
            "Último":   st.column_config.NumberColumn(format="localized"),
            "Dia":      st.column_config.NumberColumn(format="percent"),
            "12 meses": st.column_config.NumberColumn(format="percent"),
            "Faixa":    st.column_config.ProgressColumn(
                "Faixa mín–máx", min_value=0, max_value=1, format="percent",
                help="Posição do preço entre a mínima (0%) e a máxima (100%) do período"),
            "σ média":  st.column_config.NumberColumn(
                "σ da média", format="%+.1f",
                help="Distância da média em desvios-padrão: "
                     "acima de +2 ou abaixo de −2 = preço esticado"),
            "Mín":      st.column_config.NumberColumn(format="localized"),
            "Máx":      st.column_config.NumberColumn(format="localized"),
        },
    )

# ----------------------------------------------------------------------------
# Aba 3 — estatísticas descritivas das ações
# ----------------------------------------------------------------------------
with aba_resumo:
    dias = st.number_input("Últimos N pregões", 20, 750, 496, step=20)
    incluir_etf = st.toggle("Incluir BOVA11", value=True)
    excluir = NAO_ACOES if incluir_etf else NAO_ACOES + ["BOVA11"]

    resumo = tabela_resumo(precos, dias=int(dias), excluir=excluir)
    # Estatísticas nas linhas, uma coluna por ação; números no formato brasileiro
    estilo = (resumo.style
              .format(precision=2, decimal=",", thousands=".")
              .format(precision=0, decimal=",", thousands=".",
                      subset=pd.IndexSlice[["count"], :]))
    st.dataframe(estilo, width="stretch")
