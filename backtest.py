"""
backtest.py — testa a regra de entrada e saída do app com dados reais (Yahoo + CDI).

Usa as mesmas funções do analise_ativos.py (sinal_compra, bandas_moveis, K_STOP_GAIN),
então testa exatamente a regra do app e do alerta:
    entrada:  difusor short > long, MMA63 > MMA84 e preço < μ + 1σ
    saída:    fechamento acima de μ + K_STOP_GAIN·σ (sem stop loss)

Para cada ação compara, no mesmo período:
    estratégia   — comprado durante as operações; fora delas, o dinheiro rende CDI
    buy & hold   — comprar no início e segurar (fechamento ajustado: inclui proventos)
    CDI          — só CDI (série 12 do SGS / Banco Central)
e testa a entrada no 1º, 2º, 3º e 5º dia seguido de COMPRAR.

Uso no Colab (com analise_ativos.py e backtest.py na pasta):
    !python backtest.py
ou, para explorar:
    from backtest import *
    res = rodar()
    mostrar(res)
    plot_curvas(res, "PETR4")
    res["ops"][1]            # todas as operações com entrada no 1º dia

Cuidados: as ações testadas são as grandes de hoje (viés de sobrevivência); custos
aproximados; IR fora; resultado passado não garante resultado futuro.
"""
import os
import time

import numpy as np
import pandas as pd
import requests

from analise_ativos import (JANELA_PLOT, K_STOP_GAIN, TICKERS, baixar_precos, bandas_moveis,
                            serie, sinal_compra)

PERIODO = "10y"
CUSTO = 0.0005               # por ordem (emolumentos/taxas B3); corretagem zero
DIAS_ENTRADA = (1, 2, 3, 5)  # entrar no Nº dia seguido de COMPRAR
INICIO = JANELA_PLOT         # 1º pregão avaliado: bandas μ/σ já com 248 pregões
ACOES = [t for t in TICKERS if not t.startswith("^") and not t.endswith("=X")]

SGS_CDI = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados"   # % ao dia útil
SGS_CDI_MES = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.4391/dados"  # % acum. no mês
CACHE_CDI = "cdi_cache.csv"   # cópia local das taxas (evita depender do Banco Central)


# ----------------------------------------------------------------------------
# Dados
# ----------------------------------------------------------------------------
def _consultar_sgs(ini, fim, tentativas=3, espera=10, url=SGS_CDI):
    """Uma consulta ao SGS, com novas tentativas (o servidor às vezes dá 502/504)."""
    for k in range(tentativas):
        try:
            r = requests.get(url, timeout=60, params={
                "formato": "json", "dataInicial": f"{ini:%d/%m/%Y}",
                "dataFinal": f"{fim:%d/%m/%Y}"})
            if r.status_code == 404:          # nenhum dado no intervalo (ex.: só hoje)
                return []
            r.raise_for_status()
            dados = r.json()
            # Às vezes o SGS responde 200 com outro conteúdo (aviso, erro em JSON)
            if not isinstance(dados, list) or (dados and not (
                    isinstance(dados[0], dict) and {"data", "valor"} <= dados[0].keys())):
                raise ValueError(f"resposta inesperada do Banco Central: {str(dados)[:80]}")
            return dados
        except (requests.RequestException, ValueError) as e:
            if k == tentativas - 1:
                raise
            print(f"CDI {ini:%m/%Y}–{fim:%m/%Y}: falha ({e}); nova tentativa em {espera} s")
            time.sleep(espera)


def _baixar_taxas(inicio, fim, anos=2):
    """Taxas diárias do CDI (fração) em blocos de `anos` anos."""
    partes, ini = [], pd.Timestamp(inicio)
    while ini <= fim:
        f = min(ini + pd.DateOffset(years=anos) - pd.Timedelta(days=1), pd.Timestamp(fim))
        partes += _consultar_sgs(ini, f)
        ini = f + pd.Timedelta(days=1)
    if not partes:
        return pd.Series(dtype=float)
    df = pd.DataFrame(partes)
    return pd.Series(pd.to_numeric(df["valor"]).to_numpy() / 100,
                     index=pd.to_datetime(df["data"], dayfirst=True))


def _taxas_do_mensal(inicio, fim):
    """
    Alternativa quando a série diária falha: CDI acumulado no mês (série 4391),
    distribuído igualmente pelos dias úteis de cada mês. O total de cada mês é exato.
    """
    dados = _consultar_sgs(inicio, fim, url=SGS_CDI_MES)
    taxas = []
    for item in dados:
        m0 = pd.to_datetime(item["data"], dayfirst=True)
        m1 = min(m0 + pd.offsets.MonthEnd(0), pd.Timestamp(fim))   # mês corrente: até `fim`
        dias = pd.bdate_range(max(m0, pd.Timestamp(inicio)), m1)
        if len(dias):
            diaria = (1 + float(item["valor"]) / 100) ** (1 / len(pd.bdate_range(m0, m1))) - 1
            taxas.append(pd.Series(diaria, index=dias))
    return pd.concat(taxas) if taxas else pd.Series(dtype=float)


def baixar_cdi(inicio, fim, cache=CACHE_CDI):
    """
    Índice acumulado do CDI (1,0 no primeiro dia) entre as datas. Guarda as taxas em
    `cache` (CSV na pasta atual) e só consulta o Banco Central o que faltar. Se a série
    diária estiver fora do ar, usa a mensal (sem gravar no cache).
    """
    inicio, fim = pd.Timestamp(inicio), pd.Timestamp(fim)
    try:
        return _baixar_cdi_diario(inicio, fim, cache)
    except (requests.RequestException, ValueError) as e:
        print(f"CDI diário indisponível ({e}); usando o CDI mensal do Banco Central.")
        taxa = _taxas_do_mensal(inicio, fim).loc[inicio:fim]
        if taxa.empty:
            raise ValueError("o Banco Central não devolveu dados do CDI no período")
        return (1 + taxa).cumprod()


def _baixar_cdi_diario(inicio, fim, cache):
    taxa = pd.Series(dtype=float)
    if cache and os.path.exists(cache):
        taxa = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
    novos = []
    if taxa.empty:
        novos.append(_baixar_taxas(inicio, fim))
    else:
        if (taxa.index[0] - inicio).days > 5:      # fim de semana/feriado no início: ok
            novos.append(_baixar_taxas(inicio, taxa.index[0] - pd.Timedelta(days=1)))
        if taxa.index[-1] < fim:
            try:
                novos.append(_baixar_taxas(taxa.index[-1] + pd.Timedelta(days=1), fim))
            except (requests.RequestException, ValueError) as e:
                # Cache quase em dia: segue com ele (faltam só os últimos dias)
                if (fim - taxa.index[-1]).days > 10:
                    raise
                print(f"CDI dos últimos dias indisponível ({e}); usando o cache até "
                      f"{taxa.index[-1]:%d/%m/%Y}.")
    novos = [n for n in novos if not n.empty]
    if novos:
        taxa = pd.concat(([taxa] if not taxa.empty else []) + novos)
        taxa = taxa[~taxa.index.duplicated()].sort_index()
        if cache:
            taxa.rename("cdi").to_csv(cache)
    taxa = taxa.loc[inicio:fim]
    if taxa.empty:
        raise ValueError("o Banco Central não devolveu dados do CDI no período")
    return (1 + taxa).cumprod()


def _cdi_nas_datas(cdi, datas):
    """Retorno diário do CDI no calendário do ativo (0 se o CDI não estiver disponível)."""
    if cdi is None:
        return pd.Series(0.0, index=datas)
    idx = cdi.reindex(cdi.index.union(datas)).ffill().reindex(datas)
    return idx.pct_change().fillna(0.0)


# ----------------------------------------------------------------------------
# Simulação
# ----------------------------------------------------------------------------
def simular(s, ativo, teto, n_dia=1, inicio=INICIO):
    """
    Operações pelos fechamentos. Entra quando não há operação aberta e as condições
    valem há `n_dia` pregões seguidos ou mais (n_dia=1 é a regra do app); sai, a partir
    do pregão seguinte, no 1º fechamento acima do teto (μ + k·σ do dia).
    Só abre operações a partir do índice `inicio`. Devolve [(i_entrada, i_saida|None)].
    """
    p = s.to_numpy()
    ops, aberta, seguidos = [], None, 0
    for i in range(len(p)):
        seguidos = seguidos + 1 if ativo[i] else 0
        if aberta is None:
            if i >= inicio and seguidos >= n_dia:
                aberta = i
            continue
        if p[i] > teto[i]:
            ops.append((aberta, i))
            aberta = None
    if aberta is not None:
        ops.append((aberta, None))
    return ops


def tabela_ops(s, ops, custo=CUSTO):
    """Uma linha por operação: datas, preços, resultado bruto e líquido, dias, pior queda."""
    p, idx, linhas = s.to_numpy(), s.index, []
    for e, x in ops:
        fim = x if x is not None else len(p) - 1
        bruto = p[fim] / p[e] - 1
        linhas.append({
            "entrada": idx[e], "preco_entrada": p[e],
            "saida": idx[x] if x is not None else pd.NaT,
            "preco_saida": p[x] if x is not None else np.nan,
            "aberta": x is None,
            "bruto": bruto,
            # custo na compra e na venda (a aberta é avaliada como se vendesse hoje)
            "liquido": (1 + bruto) * (1 - custo) ** 2 - 1,
            "dias": fim - e + 1,
            "pior_queda": p[e:fim + 1].min() / p[e] - 1,      # pior fechamento na operação
        })
    return pd.DataFrame(linhas, columns=["entrada", "preco_entrada", "saida", "preco_saida",
                                         "aberta", "bruto", "liquido", "dias", "pior_queda"])


def curva_estrategia(s, ops, r_cdi, inicio=INICIO, custo=CUSTO):
    """
    Capital diário (1,0 no início): comprado do pregão seguinte à entrada até a saída
    (inclusive); fora das operações rende CDI. Custo aplicado nos dias de compra e venda.
    """
    r_ativo = s.pct_change().fillna(0.0).to_numpy()
    comprado = np.zeros(len(s), bool)
    custos = np.ones(len(s))
    for e, x in ops:
        fim = x if x is not None else len(s) - 1
        comprado[e + 1:fim + 1] = True
        custos[e] *= 1 - custo
        if x is not None:
            custos[x] *= 1 - custo
    r = np.where(comprado, r_ativo, r_cdi.to_numpy())
    fator = 1 + r
    fator[:inicio + 1] = 1.0                       # avaliação começa em `inicio`
    fator = fator * custos                         # inclui a compra feita em `inicio`
    # Começa em 1,0 (menos o custo, se comprar já em `inicio`): não renormalizar
    eq = pd.Series(np.cumprod(fator), index=s.index).iloc[inicio:]
    return eq, pd.Series(comprado, index=s.index).iloc[inicio:]


def _anual(curva):
    anos = (curva.index[-1] - curva.index[0]).days / 365.25
    return curva.iloc[-1] ** (1 / anos) - 1 if anos > 0 else np.nan


def _queda_max(curva):
    return (curva / curva.cummax() - 1).min()


# ----------------------------------------------------------------------------
# Execução
# ----------------------------------------------------------------------------
def avaliar(precos, ticker, cdi, n_dia=1, k_gain=K_STOP_GAIN):
    """Operações e curvas de um ativo para um dia de entrada."""
    s = serie(precos, ticker)
    ativo = sinal_compra(precos, ticker)["ativo"].to_numpy()
    mu, sd = bandas_moveis(s)
    teto = (mu + k_gain * sd).to_numpy()
    ops = simular(s, ativo, teto, n_dia)
    r_cdi = _cdi_nas_datas(cdi, s.index)
    eq, comprado = curva_estrategia(s, ops, r_cdi)
    bh = s.iloc[INICIO:] / s.iloc[INICIO]
    so_cdi = (1 + r_cdi.iloc[INICIO:]).cumprod()
    so_cdi = so_cdi / so_cdi.iloc[0] if cdi is not None else so_cdi * np.nan
    return {"ops": tabela_ops(s, ops).assign(ativo=ticker), "estrategia": eq,
            "buy_hold": bh, "cdi": so_cdi, "comprado": comprado}


def resumo(av):
    """Linha de métricas de um ativo."""
    ops = av["ops"]
    fechadas, aberta = ops[~ops["aberta"]], ops[ops["aberta"]]
    eq, bh = av["estrategia"], av["buy_hold"]
    return {
        "Início":          eq.index[0],
        "Operações":       len(fechadas),
        "Com lucro":       (fechadas["liquido"] > 0).mean() if len(fechadas) else np.nan,
        "Ganho médio":     fechadas["liquido"].mean() if len(fechadas) else np.nan,
        "Pregões (méd.)":  fechadas["dias"].mean() if len(fechadas) else np.nan,
        "Pior queda":      ops["pior_queda"].min() if len(ops) else np.nan,
        "Aberta (dias)":   aberta["dias"].iloc[0] if len(aberta) else np.nan,
        "Aberta (result.)": aberta["liquido"].iloc[0] if len(aberta) else np.nan,
        "Tempo comprado":  av["comprado"].mean(),
        "Estratégia a.a.": _anual(eq),
        "Buy&hold a.a.":   _anual(bh),
        "CDI a.a.":        _anual(av["cdi"]),
        "Queda máx. estr.": _queda_max(eq),
        "Queda máx. B&H":  _queda_max(bh),
    }


def rodar(acoes=ACOES, periodo=PERIODO, dias_entrada=DIAS_ENTRADA, precos=None, cdi=None):
    """
    Baixa os dados (se não vierem prontos) e roda o teste. Devolve um dicionário com:
      tabelas[n]   — métricas por ação, entrada no nº dia
      comparativo  — médias por dia de entrada
      ops[n]       — todas as operações, entrada no nº dia
      av[n][ativo] — curvas (estratégia, buy & hold, CDI) para gráficos
    """
    if precos is None:
        precos = baixar_precos(acoes, periodo=periodo)
    aviso_cdi = None
    if cdi is None:
        try:
            cdi = baixar_cdi(precos.index[0], precos.index[-1])
        except Exception as e:            # sem CDI: fora das operações o dinheiro fica parado
            aviso_cdi = f"CDI indisponível ({e}); fora das operações o capital rende 0%."
    res = {"tabelas": {}, "ops": {}, "av": {}, "comparativo": None,
           "falhas": list(precos.attrs.get("falhas", [])), "aviso_cdi": aviso_cdi}
    linhas_comp = {}
    for n in dias_entrada:
        avs = {t: avaliar(precos, t, cdi, n) for t in precos.columns
               if len(serie(precos, t)) > INICIO + 20}
        tab = pd.DataFrame({t: resumo(a) for t, a in avs.items()}).T
        todas = pd.concat([a["ops"] for a in avs.values()], ignore_index=True)
        fechadas = todas[~todas["aberta"]]
        res["tabelas"][n], res["ops"][n], res["av"][n] = tab, todas, avs
        linhas_comp[f"{n}º dia"] = {
            "Operações":        len(fechadas),
            "Com lucro":        (fechadas["liquido"] > 0).mean(),
            "Ganho médio":      fechadas["liquido"].mean(),
            "Pregões (méd.)":   fechadas["dias"].mean(),
            "Pior queda":       todas["pior_queda"].min(),
            "Abertas c/ perda": int((todas["aberta"] & (todas["liquido"] < 0)).sum()),
            "Estratégia a.a.":  tab["Estratégia a.a."].mean(),
            "Buy&hold a.a.":    tab["Buy&hold a.a."].mean(),
            "CDI a.a.":         tab["CDI a.a."].mean(),
            "Ações > B&H":      int((tab["Estratégia a.a."] > tab["Buy&hold a.a."]).sum()),
            "Ações > CDI":      (int((tab["Estratégia a.a."] > tab["CDI a.a."]).sum())
                                 if cdi is not None else np.nan),
        }
    res["comparativo"] = pd.DataFrame(linhas_comp).T
    return res


# ----------------------------------------------------------------------------
# Apresentação
# ----------------------------------------------------------------------------
PCT = ["Com lucro", "Ganho médio", "Pior queda", "Aberta (result.)", "Tempo comprado",
       "Estratégia a.a.", "Buy&hold a.a.", "CDI a.a.", "Queda máx. estr.", "Queda máx. B&H"]


def _fmt(df):
    """Tabela em texto no formato brasileiro (percentuais com vírgula)."""
    out = df.copy().astype(object)
    for c in out.columns:
        for i in out.index:
            v = df.at[i, c]
            if isinstance(v, pd.Timestamp):
                out.at[i, c] = f"{v:%m/%Y}"
            elif pd.isna(v):
                out.at[i, c] = "—"
            elif c in PCT:
                out.at[i, c] = f"{v:+.1%}".replace(".", ",") if c not in (
                    "Com lucro", "Tempo comprado") else f"{v:.0%}"
            elif isinstance(v, (float, np.floating)):
                out.at[i, c] = f"{v:.0f}"
            else:
                out.at[i, c] = str(v)
    return out


def mostrar(res):
    gain = f"{K_STOP_GAIN:g}".replace(".", ",")
    custo = f"{CUSTO:.2%}".replace(".", ",")
    print(f"Regra: entrada (difusor, MMA63>MMA84, preço < +1σ); saída stop gain "
          f"+{gain}σ, sem stop loss. Custo {custo} por ordem.")
    if res["falhas"]:
        print("⚠️ Sem dados do Yahoo para:", ", ".join(res["falhas"]))
    if res["aviso_cdi"]:
        print("⚠️", res["aviso_cdi"])
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n=== Por ação — entrada no 1º dia de COMPRAR (regra do app) ===")
    print(_fmt(res["tabelas"][min(res["tabelas"])]).to_string())
    print("\n=== Comparativo por dia de entrada (médias entre as ações) ===")
    print(_fmt(res["comparativo"]).to_string())
    print("\nOperações = fechadas no stop gain. 'Aberta' = operação ainda em curso no fim "
          "(resultado não realizado).\nPior queda = pior fechamento durante uma operação "
          "em relação ao preço de entrada.")


def plot_curvas(res, ticker, n_dia=1):
    """Capital da estratégia vs buy & hold vs CDI (escala log), com as operações."""
    import matplotlib.pyplot as plt
    av = res["av"][n_dia][ticker]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(av["estrategia"], label="Estratégia (CDI fora das operações)", lw=1.6)
    ax.plot(av["buy_hold"], label="Buy & hold", lw=1.2)
    ax.plot(av["cdi"], label="CDI", lw=1.2, color="gray")
    for _, op in av["ops"].iterrows():
        ax.axvspan(op["entrada"], op["saida"] if pd.notna(op["saida"]) else av["estrategia"].index[-1],
                   color="green", alpha=0.08)
    ax.set_yscale("log")
    ax.set_title(f"{ticker} — entrada no {n_dia}º dia (faixas verdes = operações)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    plt.show()
    return fig


if __name__ == "__main__":
    mostrar(rodar())
