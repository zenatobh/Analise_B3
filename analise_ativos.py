"""
analise_ativos.py — download de cotações (yfinance), indicadores, regra de operação
e gráficos matplotlib (notebook). Toda regra de negócio fica aqui, para app, alerta
e notebook nunca divergirem.

Notebook:
    from analise_ativos import *
    precos = baixar_precos(TICKERS)
    plot_ativo(precos, "IBOV")
    plot_hist(precos, "IBOV")
    tabela_painel(precos)
    tabela_resumo(precos)
    rodape_data(precos)

Streamlit: chame as funções de plot com mostrar=False e use a figura retornada.
"""
import re
from datetime import datetime, time
from itertools import cycle
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# ----------------------------------------------------------------------------
# Configuração
# ----------------------------------------------------------------------------
TICKERS = ["ITUB4", "BBAS3", "PETR4", "VALE3", "ABEV3", "VIVT3", "BBSE3",
           "BOVA11", "^BVSP", "^GSPC", "USDBRL=X"]

NOMES = {"^BVSP": "IBOV", "^GSPC": "SP500", "USDBRL=X": "USDBRL"}
NAO_ACOES = list(NOMES.values())                 # índices e câmbio (fora da aba Resumo)
# Só acompanhamento: aparecem no app (gráficos, Painel), mas sem sinais de COMPRAR /
# STOP GAIN e fora do aporte do mês (modo B)
ACOMPANHAMENTO = NAO_ACOES + ["BOVA11"]


def operavel(ticker):
    """True se o ativo recebe sinais (modo A) e entra no aporte do mês (modo B)."""
    return ticker not in ACOMPANHAMENTO

DIAS_UTEIS_ANO = 252   # pregões em 12 meses
JANELA_PLOT = 248      # janela das bandas (μ, σ) e período padrão dos gráficos

MMAS = [(200, "darkgreen"), (84, "orange"), (63, "fuchsia")]

BANDAS = [  # (multiplicador do desvio, cor, espessura); None = verde acima / vermelho abaixo
    (0.5, "k",  0.5),
    (1,   None, 1.4),
    (1.5, "k",  0.5),
    (2,   None, 0.8),
]

EMA_FAST, EMA_SLOW, EMA_MEDIUM, EMA_LONG = 21, 55, 34, 89   # EMA_MEDIUM não é mais usada

# Regra de operação (bandas móveis de JANELA_PLOT pregões: média μ e desvio σ até cada dia)
#   Entrada:    difusor short > long  E  MMA63 > MMA84  E  preço < μ + K_SIGMA·σ
#   Saída:      stop gain — fechamento acima de μ + K_STOP_GAIN·σ (não há stop loss)
MMA_RAPIDA, MMA_LENTA = 63, 84
K_SIGMA = 1          # teto da entrada (+1σ)
K_STOP_GAIN = 1.8    # +1,8σ

# Antes deste horário (Brasília), uma barra com a data de hoje é tratada como
# provisória: pregão em andamento, o "fechamento" ainda é o preço corrente.
FUSO = ZoneInfo("America/Sao_Paulo")
HORA_CONSOLIDADO = time(18, 30)

plt.style.use("seaborn-v0_8-colorblind")

_RE_B3 = re.compile(r"[A-Z]{4}\d{1,2}")


# ----------------------------------------------------------------------------
# Dados
# ----------------------------------------------------------------------------
def normalizar_tickers(tickers):
    """Acrescenta '.SA' a códigos da B3 sem sufixo (ex.: PETR4 -> PETR4.SA)."""
    return [f"{t}.SA" if _RE_B3.fullmatch(t) else t for t in tickers]


def _nome(ticker):
    """Nome exibido de um ticker do Yahoo (PETR4.SA -> PETR4, ^BVSP -> IBOV)."""
    return NOMES.get(ticker, ticker.removesuffix(".SA"))


def baixar_precos(tickers=TICKERS, periodo="3y"):
    """
    Baixa fechamentos ajustados. Mantém o calendário de cada mercado: linhas só
    são removidas quando TODOS os ativos estão vazios; os NaN de feriados de outros
    mercados são tratados por série em `serie()`.

    Ativos sem dados (falha do Yahoo, ticker inválido) são descartados e listados
    em `precos.attrs["falhas"]`; só lança erro se nenhum ativo vier.
    """
    tickers = normalizar_tickers(tickers)
    dados = yf.download(tickers, period=periodo, auto_adjust=True,
                        progress=False, threads=True)["Close"]
    if isinstance(dados, pd.Series):
        dados = dados.to_frame(tickers[0])

    dados = dados.reindex(columns=tickers)                 # ticker ausente vira coluna vazia
    vazios = dados.columns[dados.isna().all()].tolist()
    dados = dados.drop(columns=vazios)
    if dados.empty:
        raise ValueError(f"Sem dados para nenhum ativo: {[_nome(t) for t in vazios]}")

    precos = (dados.rename(columns=_nome)
                   .dropna(how="all")
                   .sort_index())
    precos.attrs["falhas"] = [_nome(t) for t in vazios]
    return precos


def falhas(precos):
    """Ativos que não vieram no último download (lista vazia se todos vieram)."""
    return list(precos.attrs.get("falhas", []))


def serie(precos, ticker):
    """Série limpa de um ativo, no calendário do próprio mercado."""
    if ticker not in precos:
        raise KeyError(f"Ticker '{ticker}' não está em precos: {list(precos.columns)}")
    s = precos[ticker].dropna()
    if s.empty:
        raise ValueError(f"Série vazia para {ticker}")
    return s


def provisorio(precos, ticker, agora=None):
    """
    True se o último dado do ativo é de hoje e ainda não passou de HORA_CONSOLIDADO
    (Brasília): o pregão está em andamento e o sinal pode mudar até o fechamento.
    """
    agora = agora or datetime.now(FUSO)
    return serie(precos, ticker).index[-1].date() == agora.date() \
        and agora.time() < HORA_CONSOLIDADO


def base_12m(s):
    """Preço de 12 meses atrás (DIAS_UTEIS_ANO pregões), ou o primeiro disponível."""
    return s.iloc[-(DIAS_UTEIS_ANO + 1)] if len(s) > DIAS_UTEIS_ANO else s.iloc[0]


def _var_12m(s):
    """Variação em 12 meses."""
    return s.iloc[-1] / base_12m(s) - 1


def tabela_resumo(precos, dias=JANELA_PLOT * 2, excluir=NAO_ACOES):
    """
    Estatísticas descritivas das ações (exclui índices/câmbio) nos últimos `dias`
    pregões de cada uma (recorte por série: feriados de outros mercados não contam).
    """
    cols = precos.columns.difference(excluir, sort=False)
    return pd.DataFrame({t: serie(precos, t).tail(dias).describe() for t in cols}).round(2)


# ----------------------------------------------------------------------------
# Indicadores
# ----------------------------------------------------------------------------
def mma(s, periodo):
    return s.rolling(periodo).mean()


def difusor(precos, ticker, period=DIAS_UTEIS_ANO):
    """Difusor de fluxo: short = EMA21 − EMA55; long = EMA89 do short (série cheia)."""
    s = serie(precos, ticker)
    short = (s.ewm(span=EMA_FAST, adjust=False).mean()
             - s.ewm(span=EMA_SLOW, adjust=False).mean())
    return pd.DataFrame({
        "short": short,
        "long":  short.ewm(span=EMA_LONG, adjust=False).mean(),
    }).tail(period)


def bandas_moveis(s):
    """Média e desvio móveis de JANELA_PLOT pregões (a partir de MMA_LENTA no início da série)."""
    janela = s.rolling(JANELA_PLOT, min_periods=MMA_LENTA)
    return janela.mean(), janela.std()


def bandas_fixas(s):
    """
    μ e σ dos últimos JANELA_PLOT pregões: as bandas retas dos gráficos, iguais em
    qualquer período exibido e iguais às bandas móveis da regra no último pregão.
    """
    j = s.tail(JANELA_PLOT)
    return j.mean(), j.std()


def sinal_compra(precos, ticker):
    """
    Condições de entrada dia a dia (série completa do ativo):
      difusor  — short (EMA21−EMA55) acima do long (EMA89 do short)
      mma      — MMA63 acima da MMA84
      preco    — preço abaixo de μ + 1σ (banda móvel, sem olhar o futuro; no último
                 pregão é a mesma banda do gráfico). Sem limite inferior.
      ativo    — as três condições ao mesmo tempo
    Dias sem MMA84 calculada (início da série) contam como False.
    """
    s = serie(precos, ticker)
    dif = difusor(precos, ticker, period=len(s))
    mu, sd = bandas_moveis(s)
    return pd.DataFrame({
        "difusor": dif["short"] > dif["long"],
        "mma":     mma(s, MMA_RAPIDA) > mma(s, MMA_LENTA),
        "preco":   s < mu + K_SIGMA * sd,
    }).assign(ativo=lambda d: d["difusor"] & d["mma"] & d["preco"])


def operacoes(precos, ticker, k_gain=K_STOP_GAIN):
    """
    Simula as operações pelos fechamentos (uso interno, para o stop gain):
      - entra no fechamento do primeiro dia com sinal ativo, se não houver operação aberta;
      - a partir do pregão seguinte, sai no primeiro fechamento acima de μ + k_gain·σ
        (stop gain). Não há stop loss: a operação fica aberta até o stop gain;
      - enquanto a operação está aberta, o sinal de entrada é ignorado.
    Retorna um DataFrame com uma linha por operação: entrada, preco_entrada, saida,
    preco_saida, motivo ("gain" ou None se aberta), resultado (fração) e dias.
    """
    s = serie(precos, ticker)
    ativo = sinal_compra(precos, ticker)["ativo"].to_numpy()
    mu, sd = bandas_moveis(s)
    teto = (mu + k_gain * sd).to_numpy()
    p, idx = s.to_numpy(), s.index

    ops, aberta = [], None
    for i in range(len(p)):
        if aberta is None:
            if ativo[i]:
                aberta = i
            continue
        if p[i] > teto[i]:
            ops.append((aberta, i, "gain"))
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
            "resultado":     p[fim] / p[e] - 1,
            "dias":          fim - e + 1,            # pregões, contando a entrada
        })
    return pd.DataFrame(linhas, columns=["entrada", "preco_entrada", "saida", "preco_saida",
                                         "motivo", "resultado", "dias"])


def _dias_seguidos(cond):
    """Quantos pregões seguidos, terminando no último, a condição vale (0 se não vale hoje)."""
    cond = np.asarray(cond, bool)
    if not len(cond) or not cond[-1]:
        return 0
    falsos = np.flatnonzero(~cond)
    return len(cond) - (falsos[-1] + 1 if len(falsos) else 0)


def sinal_hoje(precos, ticker, k_gain=K_STOP_GAIN):
    """
    Sinal do último pregão do ativo:
      sinal      — "gain" (a operação fechou hoje no stop gain; prevalece sobre a compra),
                   "comprar" (as três condições de entrada valem hoje) ou None
      data       — data do último pregão
      preco      — último fechamento
      stop_gain  — nível de hoje de μ + k_gain·σ;  ate_gain — distância do preço (fração)
      dia_n      — "comprar": há quantos pregões seguidos as condições valem (1 = hoje)
      op         — "gain": a operação que fechou hoje (entrada, resultado…)
      faltam     — None: condições de entrada não atendidas hoje
      acompanhamento — True para ativos só de acompanhamento (sinal sempre None)
    """
    s = serie(precos, ticker)
    mu, sd = bandas_moveis(s)
    preco, data = s.iloc[-1], s.index[-1]
    gain = mu.iloc[-1] + k_gain * sd.iloc[-1]
    if not operavel(ticker):                 # só acompanhamento: nunca há sinal
        return {"sinal": None, "data": data, "preco": preco, "stop_gain": gain,
                "ate_gain": gain / preco - 1, "dia_n": None, "op": None, "faltam": [],
                "acompanhamento": True}

    cond = sinal_compra(precos, ticker)
    ativo = cond["ativo"].to_numpy()
    ops = operacoes(precos, ticker, k_gain)
    ult = ops.iloc[-1] if len(ops) else None

    res = {"sinal": None, "data": data, "preco": preco,
           "stop_gain": gain, "ate_gain": gain / preco - 1,
           "dia_n": None, "op": None, "faltam": [], "acompanhamento": False}

    if ult is not None and ult["saida"] == data:
        res.update(sinal=ult["motivo"], op=ult)
    elif ativo[-1]:
        res.update(sinal="comprar", dia_n=_dias_seguidos(ativo))
    else:
        hoje = cond.iloc[-1]
        if not hoje["difusor"]:
            res["faltam"].append("difusor short > long")
        if not hoje["mma"]:
            res["faltam"].append(f"MMA{MMA_RAPIDA} > MMA{MMA_LENTA}")
        if not hoje["preco"]:
            res["faltam"].append(f"preço abaixo de +{K_SIGMA:g}σ".replace(".", ","))
    return res


ROTULO_SINAL = {"comprar": "🟢 COMPRAR", "gain": "✅ STOP GAIN", None: "⚪ —",
                "acompanhamento": "👁 acompanhamento"}


def tabela_painel(precos, dias=JANELA_PLOT, k_gain=K_STOP_GAIN):
    """
    Visão geral de todos os ativos: sinal de hoje (com Nº dia e distância até o
    stop gain quando é COMPRAR), último preço, variação no dia e em 12 meses, posição na
    faixa mín–máx dos últimos `dias` pregões (0 = mínima, 1 = máxima) e distância
    da média em σ.
    """
    linhas = {}
    for ticker in precos.columns:
        s = serie(precos, ticker)
        j = s.tail(dias)
        ultimo, minimo, maximo = j.iloc[-1], j.min(), j.max()
        sh = sinal_hoje(precos, ticker, k_gain)
        comprar = sh["sinal"] == "comprar"
        mu, sd = bandas_fixas(s)
        linhas[ticker] = {
            "Sinal":    ROTULO_SINAL["acompanhamento" if sh["acompanhamento"]
                                     else sh["sinal"]],
            "Dia nº":   sh["dia_n"] if comprar else None,
            "Até gain": sh["ate_gain"] if comprar else np.nan,
            "Último":   ultimo,
            "Dia":      ultimo / s.iloc[-2] - 1 if len(s) > 1 else np.nan,
            "12 meses": _var_12m(s),
            "Faixa":    (ultimo - minimo) / (maximo - minimo) if maximo > minimo else np.nan,
            "σ média":  (ultimo - mu) / sd if sd > 0 else np.nan,
            "Mín":      minimo,
            "Máx":      maximo,
        }
    painel = pd.DataFrame.from_dict(linhas, orient="index")
    painel["Dia nº"] = painel["Dia nº"].astype("Int64")      # inteiro com vazio
    return painel


# ----------------------------------------------------------------------------
# Modo B — aporte do mês (acumulação: compra uma vez por mês e nunca vende)
# ----------------------------------------------------------------------------
# No dia do aporte, entre as ações operáveis (sem os ativos de ACOMPANHAMENTO):
#   - se algum tem COMPRAR hoje (as três condições de entrada), o aporte vai para ele
#     (dividido igualmente se houver mais de um);
#   - senão, vai para o de σ da média mais negativo (o mais abaixo da média de 1 ano).
# É o "aporte dirigido" do backtest_carteira.py. O STOP GAIN não vale para estas compras.

def ranking_aporte(precos, ativos=None):
    """
    Fila do modo B: primeiro os ativos com COMPRAR hoje, depois do mais ao menos
    abaixo da média (σ da média = (preço − μ)/σ dos últimos JANELA_PLOT pregões).
    Colunas: comprar, dia_n (pregões seguidos de COMPRAR), sigma, preco, data.
    """
    ativos = [a for a in (ativos if ativos is not None else precos.columns)
              if a in precos.columns and operavel(a)]
    linhas = {}
    for a in ativos:
        s = serie(precos, a)
        mu, sd = bandas_moveis(s)
        dia_n = _dias_seguidos(sinal_compra(precos, a)["ativo"].to_numpy())
        linhas[a] = {"comprar": dia_n > 0, "dia_n": dia_n or None,
                     "sigma": (s.iloc[-1] - mu.iloc[-1]) / sd.iloc[-1],
                     "preco": s.iloc[-1], "data": s.index[-1]}
    rk = pd.DataFrame.from_dict(linhas, orient="index",
                                columns=["comprar", "dia_n", "sigma", "preco", "data"])
    return rk.sort_values(["comprar", "sigma"], ascending=[False, True])


def aporte_do_mes(precos, ativos=None):
    """
    Indicação do modo B para hoje:
      ativos  — onde aportar (todos os com COMPRAR, dividindo; ou o mais abaixo da média)
      motivo  — "comprar" ou "desconto"
      ranking — a fila completa (ranking_aporte)
    """
    rk = ranking_aporte(precos, ativos)
    if rk.empty:
        return {"ativos": [], "motivo": None, "ranking": rk}
    if rk["comprar"].any():
        return {"ativos": list(rk.index[rk["comprar"]]), "motivo": "comprar", "ranking": rk}
    return {"ativos": [rk.index[0]], "motivo": "desconto", "ranking": rk}


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
    media, std = bandas_fixas(s_full)          # sempre os últimos 248 pregões (= regra)

    fig, (ax_p, ax_d) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})

    # Preço e médias móveis (calculadas na série cheia, exibidas na janela)
    ax_p.plot(s, label="Fechamento")
    for janela, cor in MMAS:
        ax_p.plot(mma(s_full, janela).tail(period), color=cor, label=f"MMA {janela}")

    # Bandas dos últimos 248 pregões: média ± k·desvio. Em períodos curtos a escala
    # segue os dados (as bandas de 1A achatariam o gráfico); as de fora ficam cortadas.
    ylim = ax_p.get_ylim()
    ax_p.axhline(media, ls="-.", color="b", lw=1.4)
    for k, cor, lw in BANDAS:
        ax_p.axhline(media + k * std, ls="-.", color=cor or "darkgreen", lw=lw)
        ax_p.axhline(media - k * std, ls="-.", color=cor or "r", lw=lw)
    if period < JANELA_PLOT:
        ax_p.set_ylim(ylim)

    ax_p.set_title(
        f"{ticker} — Cotação: {moeda(s.iloc[-1])} ({s.index[-1]:%d/%m/%y})\n"
        f"Mín: {moeda(s.min())} | Máx: {moeda(s.max())} | "
        f"Méd 1A: {moeda(media)} | Std 1A: {moeda(std)}",
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
