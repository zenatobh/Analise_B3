"""
graficos_plotly.py — versões interativas (Plotly) dos gráficos de analise_ativos.py,
pensadas para o app Streamlit no celular: zoom por toque, valores ao tocar na linha
e layout vertical.

As versões matplotlib (plot_ativo / plot_hist) continuam em analise_ativos.py
para uso no notebook.
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analise_ativos import (BANDAS, EMA_FAST, EMA_LONG, EMA_SLOW,
                            JANELA_PLOT, MMAS, difusor, mma, moeda, serie)

# Cores equivalentes às do matplotlib ("seaborn-v0_8-colorblind" e nomes curtos)
COR_PRECO = "#0173B2"
CORES_MPL = {"k": "black", "r": "red", "b": "blue", "g": "green", "y": "goldenrod"}
CORES_ANOS = ["#0173B2", "#029E73", "#DE8F05"]   # ano −3, −2, −1

# Espessura das médias móveis no app (padrão 1.2 para as não listadas)
ESPESSURA_MMA = {200: 2.0, 84: 1.8, 63: 1.5}   # laranja e rosa

# Toque/arrasto dá zoom; duplo toque volta ao enquadramento original
CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
    "locale": "pt-BR",
}

_LAYOUT_BASE = dict(
    template="plotly_white",
    separators=",.",                      # decimal com vírgula, milhar com ponto
    hovermode="x unified",
    dragmode="zoom",
    margin=dict(l=8, r=8, t=70, b=8),
    legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                font=dict(size=10)),
    font=dict(size=11),
)


def _cor(c):
    return CORES_MPL.get(c, c)


def fig_ativo(precos, ticker, period=JANELA_PLOT, percentual=False, altura=620):
    """Preço + MMAs + bandas média ± k·desvio (painel 1) e difusor (painel 2)."""
    s_full = serie(precos, ticker)
    s = s_full.tail(period)
    media, std = s.mean(), s.std()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28], vertical_spacing=0.04)

    # Painel 1 — preço e médias móveis (calculadas na série cheia)
    fig.add_trace(go.Scatter(x=s.index, y=s, name="Fechamento",
                             line=dict(color=COR_PRECO, width=1.8)), row=1, col=1)
    for janela, cor in MMAS:
        m = mma(s_full, janela).tail(period)
        fig.add_trace(go.Scatter(x=m.index, y=m, name=f"MMA {janela}",
                                 line=dict(color=cor, width=ESPESSURA_MMA.get(janela, 1.2))),
                      row=1, col=1)

    # Bandas estáticas da janela
    fig.add_hline(y=media, line=dict(color="blue", dash="dashdot", width=1.4), row=1, col=1)
    for k, cor, lw in BANDAS:
        fig.add_hline(y=media + k * std, line=dict(color=_cor(cor or "darkgreen"),
                      dash="dashdot", width=lw), row=1, col=1)
        fig.add_hline(y=media - k * std, line=dict(color=_cor(cor or "r"),
                      dash="dashdot", width=lw), row=1, col=1)

    # Painel 2 — difusor
    dif = difusor(precos, ticker, period, percentual)
    for col, nome, cor, lw in [
        ("long",   f"Long ({EMA_LONG})",               "rgba(128,128,128,0.94)", 3),
        ("short",  f"Short ({EMA_FAST}-{EMA_SLOW})",   "red",                    2),
    ]:
        fig.add_trace(go.Scatter(x=dif.index, y=dif[col], name=nome,
                                 line=dict(color=cor, width=lw)), row=2, col=1)
    fig.add_hline(y=0, line=dict(color="black", width=0.6), row=2, col=1)

    fig.update_layout(
        **_LAYOUT_BASE,
        height=altura,
        title=dict(
            text=(f"<b>{ticker}</b> — {moeda(s.iloc[-1])} ({s.index[-1]:%d/%m/%y})<br>"
                  f"<sup>Mín {moeda(s.min())} · Máx {moeda(s.max())} · "
                  f"Méd {moeda(media)} · Std {moeda(std)}</sup>"),
            x=0, font=dict(size=15)),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False, tickformat=",.2f", row=1, col=1)
    fig.update_yaxes(showgrid=False, tickformat=",.2f",
                     ticksuffix="%" if percentual else "", row=2, col=1)
    fig.update_traces(hovertemplate="%{y:,.2f}")
    # Título em cima, legenda logo abaixo dele (cabe melhor na largura do celular)
    fig.update_layout(margin=dict(t=95), legend=dict(y=1.02))
    return fig


def fig_hist(precos, ticker, anos=3, bins=30, altura=None):
    """
    Distribuição anual de preços, adaptada ao celular: as `anos` colunas do notebook
    viram linhas empilhadas. Painel 1: preço com cada ano numa cor. Painéis seguintes:
    um histograma por ano, todos com o mesmo eixo de preço e a mesma escala de
    contagem, para comparar diretamente. O preço atual aparece em vermelho.
    """
    s = serie(precos, ticker)
    atual, fim = s.iloc[-1], s.index[-1]
    minimo, maximo = s.min() * 0.95, s.max() * 1.05
    tam_bin = (maximo - minimo) / (bins - 1)

    # Recorta as janelas anuais (mais antiga primeiro)
    janelas = []
    for i in range(anos):
        k = anos - i
        ini_j = fim - pd.DateOffset(years=k)
        fim_j = fim - pd.DateOffset(years=k - 1)
        janela = s[(s.index > ini_j) & (s.index <= fim_j)]
        if not janela.empty:
            janelas.append((janela, CORES_ANOS[i % len(CORES_ANOS)]))
    n = len(janelas)

    fig = make_subplots(rows=1 + n, cols=1, vertical_spacing=0.035,
                        row_heights=[0.34] + [0.66 / n] * n)
    linha_atual = dict(color="red", dash="dashdot", width=1)

    # Painel 1 — preço por ano
    for janela, cor in janelas:
        nome = f"{janela.index[0]:%d/%m/%y} a {janela.index[-1]:%d/%m/%y}"
        fig.add_trace(go.Scatter(x=janela.index, y=janela, name=nome,
                                 line=dict(color=cor, width=1.5),
                                 hovertemplate="%{y:,.2f}"), row=1, col=1)
    fig.add_hline(y=atual, line=linha_atual, row=1, col=1)

    # Painéis 2..n+1 — um histograma por ano, eixos compartilhados entre eles
    for r, (janela, cor) in enumerate(janelas, start=2):
        rotulo = f"{janela.index[0]:%m/%y}–{janela.index[-1]:%m/%y}"
        fig.add_trace(go.Histogram(x=janela, name=rotulo, showlegend=False,
                                   xbins=dict(start=minimo, end=maximo, size=tam_bin),
                                   marker=dict(color=cor, line=dict(color="black", width=0.5)),
                                   hovertemplate=(f"{rotulo}<br>%{{x}}<br>"
                                                  "%{y} pregões<extra></extra>")),
                      row=r, col=1)
        fig.add_vline(x=atual, line=linha_atual, row=r, col=1)
        fig.add_annotation(text=f"<b>{rotulo}</b>", font=dict(color=cor, size=11),
                           xref="x domain" if r == 1 else f"x{r} domain",
                           yref=f"y{r} domain", x=0.01, y=0.95,
                           xanchor="left", yanchor="top", showarrow=False)
        fig.update_xaxes(matches="x2", range=[minimo, maximo], tickformat=",.0f",
                         showticklabels=(r == n + 1), row=r, col=1)
        fig.update_yaxes(matches="y2", row=r, col=1)

    fig.update_layout(
        **_LAYOUT_BASE,
        height=altura or 300 + 150 * n,
        bargap=0.05,
        title=dict(text=(f"<b>{ticker}</b> — distribuição anual de preços<br>"
                         f"<sup>Atual: {moeda(atual)}</sup>"), x=0, font=dict(size=15)),
    )
    fig.update_layout(hovermode="closest", margin=dict(t=95), legend=dict(y=1.02))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_yaxes(range=[minimo, maximo], tickformat=",.0f", row=1, col=1)
    fig.update_yaxes(title_text="pregões", title_font=dict(size=10),
                     row=1 + (n + 1) // 2, col=1)
    return fig
