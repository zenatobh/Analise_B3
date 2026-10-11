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
TICKERS = ["ITUB4", "BBAS3", "PETR4", "VALE3", "ABEV3" , "VIVT3","BBSE3", "BOVA11", "^BVSP", "^GSPC", "USDBRL=X"]

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

# Operação (bandas móveis de JANELA_PLOT pregões: média μ e desvio σ até cada dia)
#   Entrada:    difusor short > long  E  MMA63 > MMA84  E  μ + K_STOP_LOSS·σ < preço < μ + K_SIGMA·σ
#   Stop gain:  fechamento acima de μ + K_STOP_GAIN·σ
#   Stop loss:  fechamento abaixo de μ + K_STOP_LOSS·σ
MMA_RAPIDA, MMA_LENTA = 63, 84
K_SIGMA = 1          # teto da entrada (+1σ)
K_STOP_GAIN = 2      # +2σ
K_STOP_LOSS = -1     # −1σ (também é o piso da entrada)

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


def tabela_painel(precos, dias=JANELA_PLOT, k_gain=K_STOP_GAIN, k_loss=K_STOP_LOSS):
    """
    Visão geral de todos os ativos nos últimos `dias` pregões de cada um:
    operação aberta (há quantos pregões e resultado até agora), último preço, variação no dia e em
    12 meses, posição do preço na faixa
    mín–máx do período (0 = na mínima, 1 = na máxima) e distância da média em σ.
    """
    linhas = {}
    for ticker in precos.columns:
        s = serie(precos, ticker)
        j = s.tail(dias)
        ultimo, minimo, maximo = j.iloc[-1], j.min(), j.max()
        st_op = status_operacao(precos, ticker, k_gain, k_loss)
        aberta = st_op["aberta"]
        linhas[ticker] = {
            "Comprado": aberta,
            "Há":       st_op["op"]["dias"] if aberta else None,
            "Result.":  st_op["op"]["resultado"] if aberta else np.nan,
            "Último":   ultimo,
            "Dia":      ultimo / s.iloc[-2] - 1 if len(s) > 1 else np.nan,
            "12 meses": ultimo / s.iloc[-min(len(s), DIAS_UTEIS_ANO)] - 1,
            "Faixa":    (ultimo - minimo) / (maximo - minimo) if maximo > minimo else np.nan,
            "σ média":  (ultimo - j.mean()) / j.std() if j.std() > 0 else np.nan,
            "Mín":      minimo,
            "Máx":      maximo,
        }
    return pd.DataFrame.from_dict(linhas, orient="index")


# ----------------------------------------------------------------------------
# Indicadores
# ----------------------------------------------------------------------------
def mma(s, periodo):
    return s.rolling(periodo).mean()


def difusor(precos, ticker, period=DIAS_UTEIS_ANO):
    """Difusor de fluxo: EMA21-EMA55 e suas EMAs de 34 e 89 (série cheia para aquecimento)."""
    s = serie(precos, ticker)
    fast = s.ewm(span=EMA_FAST, adjust=False).mean()
    slow = s.ewm(span=EMA_SLOW, adjust=False).mean()
    short = fast - slow
    return pd.DataFrame({
        "short":  short,
        "medium": short.ewm(span=EMA_MEDIUM, adjust=False).mean(),
        "long":   short.ewm(span=EMA_LONG, adjust=False).mean(),
    }).tail(period)


def bandas_moveis(s):
    """Média e desvio móveis de JANELA_PLOT pregões (a partir de MMA_LENTA no início da série)."""
    janela = s.rolling(JANELA_PLOT, min_periods=MMA_LENTA)
    return janela.mean(), janela.std()


def sinal_compra(precos, ticker, k_loss=K_STOP_LOSS):
    """
    Condições de entrada dia a dia (série completa do ativo):
      difusor  — short (EMA21−EMA55) acima do long (EMA89 do short)
      mma      — MMA63 acima da MMA84
      preco    — preço entre μ + k_loss·σ e μ + 1σ (bandas móveis, sem olhar o futuro;
                 no último pregão são as mesmas bandas do gráfico em 1A)
      ativo    — as três condições ao mesmo tempo
    Dias sem MMA84 calculada (início da série) contam como False.
    """
    s = serie(precos, ticker)
    dif = difusor(precos, ticker, period=len(s))
    mu, sd = bandas_moveis(s)
    return pd.DataFrame({
        "difusor": dif["short"] > dif["long"],
        "mma":     mma(s, MMA_RAPIDA) > mma(s, MMA_LENTA),
        "preco":   (s > mu + k_loss * sd) & (s < mu + K_SIGMA * sd),
    }).assign(ativo=lambda d: d["difusor"] & d["mma"] & d["preco"])


def operacoes(precos, ticker, k_gain=K_STOP_GAIN, k_loss=K_STOP_LOSS):
    """
    Simula as operações pelos fechamentos:
      - entra no fechamento do primeiro dia com sinal ativo, se não houver operação aberta;
      - a partir do pregão seguinte, sai no primeiro fechamento acima de μ + k_gain·σ
        (stop gain) ou abaixo de μ + k_loss·σ (stop loss);
      - enquanto a operação está aberta, o sinal de entrada é ignorado (pode ligar e
        desligar sem efeito).
    Retorna um DataFrame com uma linha por operação: entrada, preco_entrada, saida,
    preco_saida, motivo ("gain", "loss" ou None se aberta), resultado (fração) e dias.
    """
    s = serie(precos, ticker)
    ativo = sinal_compra(precos, ticker, k_loss)["ativo"].to_numpy()
    mu, sd = bandas_moveis(s)
    teto = (mu + k_gain * sd).to_numpy()
    piso = (mu + k_loss * sd).to_numpy()
    p, idx = s.to_numpy(), s.index

    ops, aberta = [], None
    for i in range(len(p)):
        if aberta is None:
            if ativo[i]:
                aberta = i
            continue
        motivo = "gain" if p[i] > teto[i] else "loss" if p[i] < piso[i] else None
        if motivo:
            ops.append((aberta, i, motivo))
            aberta = None
    if aberta is not None:
        ops.append((aberta, None, None))

    linhas = []
    for e, x, motivo in ops:
        fim = x if x is not None else len(p) - 1
        linhas.append({
            "entrada":       idx[e],
            "preco_entrada": p[e],
            "saida":         idx[x] if x is not None else pd.NaT,
            "preco_saida":   p[x] if x is not None else np.nan,
            "motivo":        motivo,                 # None = operação aberta
            "resultado":     p[fim] / p[e] - 1,      # aberta: resultado até o último fechamento
            "dias":          fim - e + 1,            # pregões, contando a entrada
        })
    return pd.DataFrame(linhas, columns=["entrada", "preco_entrada", "saida", "preco_saida",
                                         "motivo", "resultado", "dias"])


def status_operacao(precos, ticker, k_gain=K_STOP_GAIN, k_loss=K_STOP_LOSS):
    """
    Situação no último pregão:
      aberta     — bool (operação em andamento, inclusive se entrou hoje)
      op         — a operação aberta (Series) ou a última fechada (ou None se nunca houve)
      stop_gain  — nível atual de μ + k_gain·σ
      stop_loss  — nível atual de μ + k_loss·σ
      faltam     — se não há operação aberta: condições de entrada não atendidas hoje
    """
    s = serie(precos, ticker)
    mu, sd = bandas_moveis(s)
    ops = operacoes(precos, ticker, k_gain, k_loss)
    op = ops.iloc[-1] if len(ops) else None
    aberta = op is not None and pd.isna(op["saida"])

    faltam = []
    if not aberta:
        hoje = sinal_compra(precos, ticker, k_loss).iloc[-1]
        if not hoje["difusor"]:
            faltam.append("difusor short > long")
        if not hoje["mma"]:
            faltam.append(f"MMA{MMA_RAPIDA} > MMA{MMA_LENTA}")
        if not hoje["preco"]:
            faixa = f"entre {k_loss:+g}σ e +{K_SIGMA:g}σ".replace(".", ",").replace("-", "−")
            faltam.append(f"preço {faixa}")
    return {
        "aberta":    aberta,
        "op":        op,
        "stop_gain": mu.iloc[-1] + k_gain * sd.iloc[-1],
        "stop_loss": mu.iloc[-1] + k_loss * sd.iloc[-1],
        "faltam":    faltam,
    }


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
def plot_ativo(precos, ticker, period=JANELA_PLOT,
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
    dif = difusor(precos, ticker, period)
    ax_d.plot(dif["long"],   color="#808080F0", lw=3, label=f"Long ({EMA_LONG})")
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
