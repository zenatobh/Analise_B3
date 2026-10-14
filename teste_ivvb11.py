"""
teste_ivvb11.py — o IVVB11 cabe nas regras atuais? Testa sem ajustar nenhum parâmetro.

1) IVVB11 sozinho no modo A (operação com stop gain +1,8σ, CDI fora das operações),
   em todo o histórico (desde 07/2014), contra comprar e segurar e contra o CDI.
2) Carteira com aporte mensal (mesmas premissas do backtest_carteira.py: R$ 1.000/mês,
   custos e IR — IVVB11 é ETF: sem isenção dos R$ 20 mil), de 08/2015 até hoje,
   comparando o universo das 7 ações COM e SEM o IVVB11 nos modos A e B.

Uso no Colab (pasta Analise_B3):
    !python teste_ivvb11.py
"""
import pandas as pd

import backtest as bt
import backtest_carteira as bc
from analise_ativos import baixar_precos

INICIO_COMUM = "2015-08-01"    # IVVB11 desde 07/07/2014 + 248 pregões para as bandas
LINHAS = ["Regra do app", "2) Aporte dirigido", "Tudo em IVVB11", "Tudo em BOVA11",
          "Tudo no CDI"]
ROTULO = {"Regra do app": "Modo A (regra do app)", "2) Aporte dirigido": "Modo B (aporte do mês)"}


def _pct(v, sinal=True):
    return "—" if pd.isna(v) else (f"{v:+.1%}" if sinal else f"{v:.1%}").replace(".", ",")


def rodar(precos=None, cdi=None):
    if precos is None:
        precos = baixar_precos(bt.ACOES + ["IVVB11"], periodo="max")
    if "IVVB11" not in precos.columns:
        raise SystemExit("Sem dados do IVVB11 no Yahoo; tente de novo mais tarde.")
    if cdi is None:
        cdi = bt.baixar_cdi(precos.index[0], precos.index[-1])

    # 1) IVVB11 sozinho no modo A
    so = precos[["IVVB11"]].copy()
    so.attrs["falhas"] = []
    print("=" * 100 + "\n1) IVVB11 SOZINHO NO MODO A — todo o histórico\n" + "=" * 100)
    bt.mostrar(bt.rodar(precos=so, cdi=cdi))

    # 2) Carteira com e sem IVVB11 no universo operável
    com = precos.copy()
    sem = precos.drop(columns="IVVB11")
    for p in (com, sem):
        p.attrs["falhas"] = []
    r_com = bc.rodar_carteira(precos=com, cdi=cdi, inicio=INICIO_COMUM, referencias=("IVVB11",))
    r_sem = bc.rodar_carteira(precos=sem, cdi=cdi, inicio=INICIO_COMUM)
    ini, fim = r_com["periodo"]
    print("\n" + "=" * 100 + f"\n2) CARTEIRA COM APORTE MENSAL — {ini:%m/%Y} a {fim:%m/%Y}, "
          "com e sem IVVB11 entre os operáveis\n" + "=" * 100)
    tab = pd.DataFrame(index=[ROTULO.get(n, n) for n in LINHAS])
    for rotulo, res in (("sem IVVB11", r_sem), ("COM IVVB11", r_com)):
        t = res["tabela"]
        tab[f"TIR {rotulo}"] = [_pct(t.at[n, "TIR líq. a.a."]) if n in t.index else "—"
                                for n in LINHAS]
        tab[f"Queda {rotulo}"] = [_pct(t.at[n, "Pior queda"], False) if n in t.index else "—"
                                  for n in LINHAS]
    print(tab.to_string())

    # Quanto o IVVB11 participou
    o_b = r_com["series"]["2) Aporte dirigido"]["ordens"]
    o_b = o_b[o_b["ordem"] == "compra"]
    fatia_b = o_b.loc[o_b["ativo"] == "IVVB11", "valor"].sum() / o_b["valor"].sum()
    o_a = r_com["series"]["Regra do app"]["ordens"]
    ops_a = int(((o_a["ativo"] == "IVVB11") & (o_a["ordem"] == "venda")).sum())
    print(f"\nModo B: {fatia_b:.0%} dos aportes foram para o IVVB11. "
          f"Modo A: {ops_a} operação(ões) do IVVB11 fechada(s) no stop gain.")
    print("TIR = rentabilidade anual líquida (custos e IR, vendendo tudo no fim). "
          "Queda = pior queda da cota.")
    return {"modo_a": so, "com": r_com, "sem": r_sem, "tabela": tab}


if __name__ == "__main__":
    rodar()
