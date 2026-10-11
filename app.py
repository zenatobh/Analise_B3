"""
app.py — painel de ativos para celular (Streamlit + Plotly).

Local:   streamlit run app.py
Deploy:  Streamlit Community Cloud apontando para este arquivo.
"""
import streamlit as st

from analise_ativos import (NAO_ACOES, TICKERS, baixar_precos, moeda, serie,
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

aba_ativo, aba_resumo = st.tabs(["Ativo", "Resumo"])

# ----------------------------------------------------------------------------
# Aba 1 — análise de um ativo
# ----------------------------------------------------------------------------
with aba_ativo:
    opcoes = list(precos.columns)
    ticker = st.selectbox("Ativo", opcoes, index=opcoes.index("IBOV"))

    s = serie(precos, ticker)
    var_dia = s.iloc[-1] / s.iloc[-2] - 1
    var_ano = s.iloc[-1] / s.iloc[-min(len(s), 252)] - 1
    c1, c2, c3 = st.columns(3)
    c1.metric("Último", moeda(s.iloc[-1]))
    c2.metric("Dia", f"{var_dia:+.2%}".replace(".", ","))
    c3.metric("12 meses", f"{var_ano:+.2%}".replace(".", ","))

    escolha = st.segmented_control("Período", list(PERIODOS), default=PERIODO_PADRAO,
                                   label_visibility="collapsed")
    periodo = PERIODOS[escolha or PERIODO_PADRAO] or len(s)

    with st.expander("Opções"):
        percentual = st.toggle("Difusor em %", value=False)

    mostrar_fig(fig_ativo(precos, ticker, period=periodo, percentual=percentual))
    st.caption("Arraste sobre o gráfico para dar zoom · toque duplo para voltar.")
    mostrar_fig(fig_hist(precos, ticker))

# ----------------------------------------------------------------------------
# Aba 2 — tabela comparativa das ações
# ----------------------------------------------------------------------------
with aba_resumo:
    dias = st.number_input("Últimos N pregões", 20, 750, 496, step=20)
    incluir_etf = st.toggle("Incluir BOVA11", value=True)
    excluir = NAO_ACOES if incluir_etf else NAO_ACOES + ["BOVA11"]

    resumo = tabela_resumo(precos, dias=int(dias), excluir=excluir)
    # Transposta: uma linha por ação cabe melhor na largura do celular
    st.dataframe(resumo.T, width="stretch")
