"""
analise_ativos.py — download de cotações (yfinance) e gráficos de análise técnica.

Notebook:
    from analise_ativos import *
    precos = baixar_precos(TICKERS)
    plot_ativo(precos, "IBOV")
    plot_hist(precos, "IBOV")
    tabela_resumo(precos)
    rodape_data(precos)

Streamlit: chame as funções de plot com mostrar=False e use a figura retornada.
"""
import re
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# ----------------------------------------------------------------------------
# Configuração
# ----------------------------------------------------------------------------
TICKERS = ["ITUB4", "BBAS3", "PETR4", "VALE3", "ABEV3", "EGIE3", "VIVT3",
           "WEGE3", "RENT3", "BBSE3", "BOVA11", "^BVSP", "^GSPC", "USDBRL=X"]

NOMES = {"^BVSP": "IBOV", "^GSPC": "SP500", "USDBRL=X": "USDBRL"}
NAO_ACOES = list(NOMES.values())

DIAS_UTEIS_ANO = 252   # anualização
JANELA_PLOT = 248      # período exibido nos gráficos

MMAS = [(200, "darkgreen"), (84, "orange"), (63, "fuchsia")]

BANDAS = [  # (multiplicador do desvio, cor, espessura); None = verde acima / vermelho abaixo
    (0.5, "k",  0.5),
    (1,   None, 1.4),
    (1.5, "k",  0.5),
    (2,   None, 0.8),
]

EMA_FAST, EMA_SLOW, EMA_MEDIUM, EMA_LONG = 21, 55, 34, 89

plt.style.use("seaborn-v0_8-colorblind")

_RE_B3 = re.compile(r"[A-Z]{4}\d{1,2}")


# ----------------------------------------------------------------------------
# Dados
# ----------------------------------------------------------------------------
def normalizar_tickers(tickers):
    """Acrescenta '.SA' a códigos da B3 sem sufixo (ex.: PETR4 -> PETR4.SA)."""
    return [f"{t}.SA" if _RE_B3.fullmatch(t) else t for t in tickers]


def baixar_precos(tickers=TICKERS, periodo="3y"):
    """
    Baixa fechamentos ajustados. Mantém o calendário de cada mercado:
    linhas só são removidas quando TODOS os ativos estão vazios; os NaN de
    feriados de outros mercados são tratados por série em `serie()`.
    """
    tickers = normalizar_tickers(tickers)
    dados = yf.download(tickers, period=periodo, auto_adjust=True,
                        progress=False, threads=True)["Close"]
    if isinstance(dados, pd.Series):
        dados = dados.to_frame(tickers[0])

    vazios = dados.columns[dados.isna().all()].tolist()
    if vazios:
        raise ValueError(f"Sem dados para: {vazios}")

    dados.columns = dados.columns.str.removesuffix(".SA")
    return dados.rename(columns=NOMES).dropna(how="all").sort_index()


def serie(precos, ticker):
    """Série limpa de um ativo, no calendário do próprio mercado."""
    if ticker not in precos:
        raise KeyError(f"Ticker '{ticker}' não está em precos: {list(precos.columns)}")
    s = precos[ticker].dropna()
    if s.empty:
        raise ValueError(f"Série vazia para {ticker}")
    return s


def tabela_resumo(precos, dias=JANELA_PLOT * 2, excluir=NAO_ACOES):
    """Estatísticas descritivas das ações (exclui índices/câmbio) nos últimos `dias`."""
    cols = precos.columns.difference(excluir, sort=False)
    return precos[cols].tail(dias).describe().round(2)


# ----------------------------------------------------------------------------
# Indicadores
# ----------------------------------------------------------------------------
def mma(s, periodo):
    return s.rolling(periodo).mean()


def difusor(precos, ticker, period=DIAS_UTEIS_ANO, percentual=False):
    """Difusor de fluxo: EMA21-EMA55 e suas EMAs de 34 e 89 (série cheia para aquecimento)."""
    s = serie(precos, ticker)
    fast = s.ewm(span=EMA_FAST, adjust=False).mean()
    slow = s.ewm(span=EMA_SLOW, adjust=False).mean()
    short = (fast / slow - 1) * 100 if percentual else fast - slow
    return pd.DataFrame({
        "short":  short,
        "medium": short.ewm(span=EMA_MEDIUM, adjust=False).mean(),
        "long":   short.ewm(span=EMA_LONG, adjust=False).mean(),
    }).tail(period)


# ----------------------------------------------------------------------------
# Formatação
# ----------------------------------------------------------------------------
_TRANS_BR = str.maketrans(",.", ".,")


def moeda(valor):
    return f"{valor:,.2f}".translate(_TRANS_BR)


def rodape_data(precos):
    """Notebook: mostra a data do último dado disponível (não a data de hoje)."""
    from IPython.display import HTML, display   # import local: não é dependência do app
    ultimo = precos.index[-1].strftime("%d/%m/%Y")
    display(HTML(f"<div style='text-align:right'><h4>Dados até: {ultimo}</h4></div>"))


def _finalizar(fig, mostrar):
    fig.tight_layout()
    if mostrar:
        plt.show()
    return fig


# ----------------------------------------------------------------------------
# Gráficos
# ----------------------------------------------------------------------------
def plot_ativo(precos, ticker, period=JANELA_PLOT, percentual=False,
               figsize=(16, 8), mostrar=True):
    s_full = serie(precos, ticker)
    s = s_full.tail(period)
    media, std = s.mean(), s.std()

    fig, (ax_p, ax_d) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})

    # Preço e médias móveis (calculadas na série cheia, exibidas na janela)
    ax_p.plot(s, label="Fechamento")
    for janela, cor in MMAS:
        ax_p.plot(mma(s_full, janela).tail(period), color=cor, label=f"MMA {janela}")

    # Bandas estáticas da janela: média ± k·desvio
    ax_p.axhline(media, ls="-.", color="b", lw=1.4)
    for k, cor, lw in BANDAS:
        ax_p.axhline(media + k * std, ls="-.", color=cor or "darkgreen", lw=lw)
        ax_p.axhline(media - k * std, ls="-.", color=cor or "r", lw=lw)

    ax_p.set_title(
        f"{ticker} — Cotação: {moeda(s.iloc[-1])} ({s.index[-1]:%d/%m/%y})\n"
        f"Mín: {moeda(s.min())} | Máx: {moeda(s.max())} | "
        f"Méd: {moeda(media)} | Std: {moeda(std)}",
        fontsize=14)
    ax_p.legend(loc="upper left")
    ax_p.grid(False)

    # Difusor
    dif = difusor(precos, ticker, period, percentual)
    ax_d.plot(dif["long"],   color="#808080F0", lw=3, label=f"Long ({EMA_LONG})")
    ax_d.plot(dif["medium"], color="tab:blue",  lw=1, label=f"Medium ({EMA_MEDIUM})")
    ax_d.plot(dif["short"],  color="red",       lw=2, label=f"Short ({EMA_FAST}-{EMA_SLOW})")
    ax_d.axhline(0, color="k", lw=0.6)
    ax_d.legend(loc="upper left")
    ax_d.grid(False)

    return _finalizar(fig, mostrar)


def plot_hist(precos, ticker, anos=3, bins=30, figsize=(16, 6), mostrar=True):
    s = serie(precos, ticker)
    atual, fim = s.iloc[-1], s.index[-1]
    minimo, maximo = s.min() * 0.95, s.max() * 1.05
    faixas = np.linspace(minimo, maximo, bins)
    cores = cycle(["b", "g", "y"])

    fig, ax = plt.subplots(2, anos, figsize=figsize, sharey="row", squeeze=False)

    for i, cor in zip(range(anos), cores):
        k = anos - i                                    # anos, ..., 2, 1
        ini_j = fim - pd.DateOffset(years=k)
        fim_j = fim - pd.DateOffset(years=k - 1)
        janela = s[(s.index > ini_j) & (s.index <= fim_j)]

        if janela.empty:
            ax[0, i].set_title(f"Ano −{k}: sem dados")
            ax[1, i].set_visible(False)
            continue

        ax[0, i].plot(janela, color=cor)
        ax[0, i].axhline(atual, ls="-.", color="r", lw=1)
        ax[0, i].set_ylim(minimo, maximo)
        ax[0, i].set_xticks([])
        ax[0, i].set_title(f"{janela.index[0]:%d/%m/%y} a {janela.index[-1]:%d/%m/%y}")

        ax[1, i].hist(janela, bins=faixas, color=cor, edgecolor="black")
        ax[1, i].axvline(atual, ls="-.", color="r", lw=1)
        ax[1, i].set_xlim(minimo, maximo)
        ax[1, i].grid(False)

    fig.suptitle(f"{ticker} — distribuição anual de preços | Atual: {moeda(atual)}", fontsize=14)
    return _finalizar(fig, mostrar)
