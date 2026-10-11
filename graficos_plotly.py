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

from analise_ativos import (BANDAS, EMA_FAST, EMA_LONG, EMA_MEDIUM, EMA_SLOW,
                            JANELA_PLOT, MMAS, difusor, mma, moeda, serie)

# Cores equivalentes às do matplotlib ("seaborn-v0_8-colorblind" e nomes curtos)
COR_PRECO = "#0173B2"
CORES_MPL = {"k": "black", "r": "red", "b": "blue", "g": "green", "y": "goldenrod"}
CORES_ANOS = ["#0173B2", "#029E73", "#DE8F05"]   # ano −3, −2, −1

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
                                 line=dict(color=cor, width=1.2)), row=1, col=1)

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
        ("medium", f"Medium ({EMA_MEDIUM})",           "#1f77b4",                1),
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


def fig_hist(precos, ticker, anos=3, bins=30, altura=620):
    """
    Distribuição anual de preços, adaptada ao celular: em vez de `anos` colunas lado
    a lado, cada ano vira uma cor — linha do preço (painel 1) e histogramas
    sobrepostos (painel 2), com o preço atual marcado nos dois.
    """
    s = serie(precos, ticker)
    atual, fim = s.iloc[-1], s.index[-1]
    minimo, maximo = s.min() * 0.95, s.max() * 1.05
    tam_bin = (maximo - minimo) / (bins - 1)

    fig = make_subplots(rows=2, cols=1, row_heights=[0.5, 0.5], vertical_spacing=0.10)

    for i in range(anos):
        k = anos - i
        ini_j = fim - pd.DateOffset(years=k)
        fim_j = fim - pd.DateOffset(years=k - 1)
        janela = s[(s.index > ini_j) & (s.index <= fim_j)]
        if janela.empty:
            continue
        cor = CORES_ANOS[i % len(CORES_ANOS)]
        nome = f"{janela.index[0]:%d/%m/%y} a {janela.index[-1]:%d/%m/%y}"

        fig.add_trace(go.Scatter(x=janela.index, y=janela, name=nome, legendgroup=nome,
                                 line=dict(color=cor, width=1.5),
                                 hovertemplate="%{y:,.2f}"), row=1, col=1)
        fig.add_trace(go.Histogram(x=janela, name=nome, legendgroup=nome, showlegend=False,
                                   xbins=dict(start=minimo, end=maximo, size=tam_bin),
                                   marker=dict(color=cor, line=dict(color="black", width=0.5)),
                                   opacity=0.55,
                                   hovertemplate="%{x}<br>%{y} pregões<extra></extra>"),
                      row=2, col=1)

    linha_atual = dict(color="red", dash="dashdot", width=1)
    fig.add_hline(y=atual, line=linha_atual, row=1, col=1)
    fig.add_vline(x=atual, line=linha_atual, row=2, col=1)

    fig.update_layout(
        **_LAYOUT_BASE,
        height=altura,
        barmode="overlay",
        title=dict(text=(f"<b>{ticker}</b> — distribuição anual de preços<br>"
                         f"<sup>Atual: {moeda(atual)}</sup>"), x=0, font=dict(size=15)),
    )
    fig.update_layout(margin=dict(t=95), legend=dict(y=1.02))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_yaxes(range=[minimo, maximo], tickformat=",.2f", row=1, col=1)
    fig.update_xaxes(range=[minimo, maximo], tickformat=",.2f", row=2, col=1)
    fig.update_yaxes(title_text="pregões", row=2, col=1)
    # No histograma o hover por eixo x unificado não faz sentido
    fig.update_xaxes(hoverformat=",.2f", row=2, col=1)
    return fig
