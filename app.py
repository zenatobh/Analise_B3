"""
app.py — painel de ativos para celular (Streamlit).

Local:   streamlit run app.py
Deploy:  Streamlit Community Cloud apontando para este arquivo.
"""
import matplotlib
matplotlib.use("Agg")  # backend sem janela (servidor)

import matplotlib.pyplot as plt
import streamlit as st

from analise_ativos import (JANELA_PLOT, NAO_ACOES, TICKERS, baixar_precos,
                            plot_ativo, plot_hist, serie, tabela_resumo)

st.set_page_config(page_title="Ativos", page_icon="📈", layout="centered")

# Figuras mais "altas" que no notebook: leem melhor na tela estreita do celular
FIG_ATIVO = (10, 8)
FIG_HIST = (10, 6)


@st.cache_data(ttl=60 * 60, show_spinner="Baixando cotações…")
def carregar():
    """Baixa uma vez por hora; todos os acessos reutilizam o cache."""
    return baixar_precos(TICKERS)


def mostrar_fig(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)  # libera memória no servidor


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
    c1.metric("Último", f"{s.iloc[-1]:,.2f}".translate(str.maketrans(",.", ".,")))
    c2.metric("Dia", f"{var_dia:+.2%}")
    c3.metric("12 meses", f"{var_ano:+.2%}")

    with st.expander("Opções"):
        periodo = st.slider("Pregões exibidos", 60, 500, JANELA_PLOT, step=4)
        percentual = st.toggle("Difusor em %", value=False)

    mostrar_fig(plot_ativo(precos, ticker, period=periodo, percentual=percentual,
                           figsize=FIG_ATIVO, mostrar=False))
    mostrar_fig(plot_hist(precos, ticker, figsize=FIG_HIST, mostrar=False))

# ----------------------------------------------------------------------------
# Aba 2 — tabela comparativa das ações
# ----------------------------------------------------------------------------
with aba_resumo:
    dias = st.number_input("Últimos N pregões", 20, 750, JANELA_PLOT * 2, step=20)
    incluir_etf = st.toggle("Incluir BOVA11", value=True)
    excluir = NAO_ACOES if incluir_etf else NAO_ACOES + ["BOVA11"]

    resumo = tabela_resumo(precos, dias=int(dias), excluir=excluir)
    # Transposta: uma linha por ação cabe melhor na largura do celular
    st.dataframe(resumo.T, use_container_width=True)
