"""
app.py — painel de ativos para celular (Streamlit + Plotly).

Local:   streamlit run app.py
Deploy:  Streamlit Community Cloud apontando para este arquivo.
"""
import pandas as pd
import streamlit as st

from analise_ativos import (HORA_CONSOLIDADO, JANELA_PLOT, K_SIGMA, K_STOP_GAIN,
                            ACOMPANHAMENTO, MMA_LENTA, MMA_RAPIDA, NAO_ACOES, SO_MODO_B,
                            TICKERS, aporte_do_mes, sinal_compra,
                            bandas_fixas, base_12m, baixar_precos, falhas, moeda,
                            provisorio, serie, sinal_hoje, tabela_painel, tabela_resumo)
from graficos_plotly import aplicar_zoom, config_zoom, fig_ativo, fig_hist

st.set_page_config(page_title="Ativos B3", page_icon="📈", layout="centered")

# Períodos exibidos no gráfico principal (em pregões)
PERIODOS = {"1M": 21, "3M": 63, "6M": 126, "1A": 248, "2A": 504, "Tudo": None}
PERIODO_PADRAO = "1A"


@st.cache_data(ttl=60 * 60, show_spinner="Baixando cotações…")
def carregar(tickers):
    """Baixa uma vez por hora; todos os acessos reutilizam o cache. A lista de ativos faz
    parte da chave do cache: mudou TICKERS, baixa de novo sem precisar do ↻."""
    return baixar_precos(list(tickers))


def mostrar_fig(fig, zoom=False):
    st.plotly_chart(aplicar_zoom(fig, zoom), width="stretch", config=config_zoom(zoom))


def pct(x, casas=2):
    """Percentual no formato brasileiro, com sinal: +1,23%."""
    return f"{x:+.{casas}%}".replace(".", ",")


def sinal_moeda(x):
    return ("+" if x >= 0 else "") + moeda(x)


def sigma(k):
    """Nível em σ no formato brasileiro, com sinal: +2σ, −1,5σ."""
    return f"{k:+g}σ".replace(".", ",").replace("-", "−")


def em_sigma(v):
    """Distância em σ no formato brasileiro: +1,2σ, −0,8σ."""
    v = round(v, 1) + 0.0        # evita "−0,0σ"
    return f"{v:+.1f}σ".replace(".", ",").replace("-", "−").replace("+0,0", "0,0")


def dia_txt(n):
    return "1º dia" if n == 1 else f"{n}º dia seguido"


def cor_variacao(v):
    """Estilo de texto para variações: verde se positiva, vermelho se negativa."""
    if pd.isna(v) or v == 0:
        return ""
    return "color: #1a9850" if v > 0 else "color: #d73027"


# ----------------------------------------------------------------------------
# Dados
# ----------------------------------------------------------------------------
try:
    precos = carregar(tuple(TICKERS))
except Exception as e:  # rede, rate limit do Yahoo, ticker inválido
    st.error(f"Não foi possível baixar as cotações: {e}")
    if st.button("Tentar novamente"):
        carregar.clear()
        st.rerun()
    st.stop()

if falhas(precos):
    st.warning(f"Sem dados do Yahoo para: {', '.join(falhas(precos))}. "
               "Os demais ativos estão atualizados; toque em ↻ para tentar de novo.",
               icon="⚠️")

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
    mu_1a, sd_1a = bandas_fixas(s)
    base = base_12m(s)
    c1, c2, c3 = st.columns(3)
    c1.metric("Último", moeda(s.iloc[-1]), delta=pct(s.iloc[-1] / s.iloc[-2] - 1),
              help="Variação em relação ao pregão anterior")
    c2.metric("12 meses", pct(s.iloc[-1] / base - 1),
              delta=sinal_moeda(s.iloc[-1] - base),
              help="Variação em 12 meses; abaixo, a diferença em pontos/R$")
    c3.metric("vs. média 1A", f"{(s.iloc[-1] - mu_1a) / sd_1a:+.1f}σ".replace(".", ","),
              help="Distância do preço atual à média dos últimos 248 pregões, "
                   "em desvios-padrão (mesma referência das bandas do gráfico)")

    escolha = st.segmented_control("Período", list(PERIODOS), default=PERIODO_PADRAO,
                                   label_visibility="collapsed")
    periodo = PERIODOS[escolha or PERIODO_PADRAO] or len(s)

    with st.expander("Opções"):
        k_gain = st.select_slider("Stop gain (acima da média)", [1.5, 1.8, 2.0, 2.5, 3.0],
                                  value=float(K_STOP_GAIN), format_func=sigma,
                                  help="Sai com lucro quando o fechamento passa desta banda")
        zoom = st.toggle("Zoom no gráfico", value=False,
                         help="Ligado: arraste para ampliar uma região e toque duas vezes "
                              "para voltar. Desligado: o gráfico não muda ao tocar.")

    # Sinal do dia: STOP GAIN prevalece sobre COMPRAR
    sh = sinal_hoje(precos, ticker, k_gain)
    if sh["acompanhamento"] and ticker in SO_MODO_B:
        cond = bool(sinal_compra(precos, ticker)["ativo"].iloc[-1])
        st.info("**Só no aporte do mês (modo B)** — sem sinais do modo A.  \n"
                "Condições de COMPRAR hoje: "
                + ("**sim** (tem prioridade no aporte)" if cond else "não"), icon="👁")
    elif sh["acompanhamento"]:
        st.info("**Só acompanhamento** — sem sinais de compra (modos A e B).", icon="👁")
    elif sh["sinal"] == "comprar":
        st.success(f"**COMPRAR** · {dia_txt(sh['dia_n'])} · preço {moeda(sh['preco'])}  \n"
                   f"Modo A: entrada; stop gain ({sigma(k_gain)}) em "
                   f"{moeda(sh['stop_gain'])} ({pct(sh['ate_gain'])})  \n"
                   "Modo B: prioridade no aporte do mês", icon="🟢")
    elif sh["sinal"] == "gain":
        op = sh["op"]
        st.success(f"**STOP GAIN · modo A** a {moeda(op['preco_saida'])} · compra de "
                   f"{op['entrada']:%d/%m/%y} a {moeda(op['preco_entrada'])} · "
                   f"resultado **{pct(op['resultado'])}** em {op['dias']} pregões  \n"
                   "Vale só para compras do modo A — no modo B não se vende.", icon="✅")
    else:
        st.info(f"**Sem sinal hoje** — falta: {' e '.join(sh['faltam'])}", icon="⚪")
    if provisorio(precos, ticker):
        st.caption(f"⏳ Pregão em andamento: preço e sinal provisórios até o fechamento "
                   f"(consolidados após {HORA_CONSOLIDADO:%Hh%M}).")

    mostrar_fig(fig_ativo(precos, ticker, period=periodo, k_gain=k_gain), zoom)
    st.caption("Arraste sobre o gráfico para dar zoom · toque duplo para voltar."
               if zoom else
               "Toque para ver os valores · use dois dedos para ampliar a tela.")
    mostrar_fig(fig_hist(precos, ticker), zoom)

# ----------------------------------------------------------------------------
# Aba 2 — visão geral de todos os ativos
# ----------------------------------------------------------------------------
with aba_painel:
    # Modo B — aporte do mês: COMPRAR hoje (dividindo) ou o mais abaixo da média
    ap = aporte_do_mes(precos)
    if ap["ativos"]:
        rk = ap["ranking"]
        if ap["motivo"] == "comprar":
            onde = " e ".join(ap["ativos"])
            dias = ", ".join(f"{a} {dia_txt(int(rk.at[a, 'dia_n']))}" for a in ap["ativos"])
            motivo = (f"COMPRAR hoje ({dias})"
                      + (" — dividir o aporte igualmente" if len(ap["ativos"]) > 1 else ""))
        else:
            onde = ap["ativos"][0]
            motivo = f"a mais abaixo da média ({em_sigma(rk.at[onde, 'sigma'])})"
        seguintes = [a for a in rk.index if a not in ap["ativos"]][:3]
        fila = " · ".join(f"{a} ({em_sigma(rk.at[a, 'sigma'])})" for a in seguintes)
        st.info(f"**Modo B — aporte de hoje: {onde}**  \n{motivo}  \n"
                f"Seguintes: {fila}", icon="💰")
        st.caption("Modo B: compre no dia do aporte e não venda — o STOP GAIN é só do "
                   "modo A. Prioridade para COMPRAR hoje; sem sinal, a mais abaixo da "
                   f"média dos últimos {JANELA_PLOT} pregões (coluna σ da média)."
                   + (" ⏳ Pregão em andamento: indicação provisória."
                      if any(provisorio(precos, a) for a in ap["ativos"]) else ""))

    painel = tabela_painel(precos, dias=JANELA_PLOT, k_gain=k_gain)
    st.caption(f"**Modo A** — entrada no COMPRAR (difusor short > long, "
               f"MMA{MMA_RAPIDA} > MMA{MMA_LENTA} e preço abaixo de {sigma(K_SIGMA)}); "
               f"saída no stop gain {sigma(k_gain)} (sem stop loss); fora das operações, "
               f"CDI. 👁 = sem sinais do modo A ({', '.join(ACOMPANHAMENTO)}); "
               f"{', '.join(SO_MODO_B)} entra só no aporte do mês (modo B). "
               f"Faixa, média e σ dos últimos {JANELA_PLOT} pregões. "
               "Toque no nome de uma coluna para ordenar.")
    prov = [a for a in painel.index if provisorio(precos, a)]
    if prov:
        st.caption(f"⏳ Provisórios (pregão em andamento): {', '.join(prov)}.")
    st.dataframe(
        painel.style.map(cor_variacao, subset=["Dia", "12 meses"]),
        width="stretch",
        height=35 * (len(painel) + 1) + 3,      # mostra todas as linhas sem rolagem
        column_config={
            "Sinal":    st.column_config.TextColumn(
                "Sinal hoje", help="COMPRAR (modos A e B) ou STOP GAIN (só modo A) "
                                   "no último pregão; 👁 = só acompanhamento"),
            "Dia nº":   st.column_config.NumberColumn(
                "Dia nº", format="%d",
                help="COMPRAR: há quantos pregões seguidos as condições valem (1 = hoje)"),
            "Até gain": st.column_config.NumberColumn(
                "Até gain", format="percent",
                help="COMPRAR: distância do preço de hoje até o stop gain"),
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
    etfs = [a for a in ("BOVA11", *SO_MODO_B) if a in precos.columns]
    incluir_etf = st.toggle(f"Incluir ETFs ({', '.join(etfs)})", value=True)
    excluir = NAO_ACOES if incluir_etf else NAO_ACOES + etfs

    resumo = tabela_resumo(precos, dias=int(dias), excluir=excluir)
    # Estatísticas nas linhas, uma coluna por ação; números no formato brasileiro
    estilo = (resumo.style
              .format(precision=2, decimal=",", thousands=".")
              .format(precision=0, decimal=",", thousands=".",
                      subset=pd.IndexSlice[["count"], :]))
    st.dataframe(estilo, width="stretch")
