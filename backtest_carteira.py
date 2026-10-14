"""
backtest_carteira.py — carteira única com aporte mensal: compara formas de usar a regra
do app com comprar e segurar e com o CDI, incluindo custos e imposto de renda.

Premissas (fixadas antes de ver os resultados):
    aporte       R$ 1.000 no 1º pregão de cada mês; ações fracionárias
    custo        0,05% por ordem (compra e venda)
    caixa        dinheiro livre aplicado no CDI (lotes com IR regressivo: 22,5% até
                 180 dias, 20% até 360, 17,5% até 720, 15% acima; resgate do mais antigo)
    IR ações     15% sobre o lucro do mês; vendas de ações até R$ 20 mil no mês isentas
                 (não vale para ETF — BOVA11/IBOV); prejuízo compensado nos meses seguintes
    fim          patrimônio líquido = como se vendesse tudo no último pregão, pagando o IR

Estratégias:
    Regra do app      — dinheiro no CDI; em cada COMPRAR de ativo sem posição compra
                        caixa ÷ nº de ativos sem posição; vende no stop gain (+1,8σ)
    3) Regra c/ peso  — idem, com a fatia multiplicada pelo espaço até o stop gain:
                        (μ + 1,8σ − preço) / (1,8σ), entre 0 e 2× (1× = preço na média)
    1) Núcleo+satélite— 50% ou 70% do aporte compra o núcleo (BOVA11) todo mês e nunca
                        vende; o resto segue a regra do app nas outras ações
    2) Dirigido       — o aporte inteiro compra o ativo com COMPRAR no dia (dividido se
                        houver mais de um) ou, sem sinal, o de σ da média mais negativo;
                        nunca vende
    Referências       — tudo no núcleo todo mês; tudo no CDI; 1/N em cada ação (teórico)

Uso no Colab (pasta Analise_B3, com analise_ativos.py e backtest.py):
    !python backtest_carteira.py           # out/2017 → hoje (núcleo BOVA11)
    !python backtest_carteira.py 2007      # 2007 → out/2017 (núcleo IBOV: BOVA11 é de 2008)
ou
    from backtest_carteira import *
    res = rodar_carteira(); mostrar_carteira(res); plot_carteira(res)
"""
import copy
import sys

import numpy as np
import pandas as pd

from analise_ativos import K_STOP_GAIN, baixar_precos, bandas_moveis, serie, sinal_compra
from backtest import ACOES, CUSTO, INICIO, _cdi_nas_datas, baixar_cdi

APORTE = 1000.0
ISENCAO_ACOES = 20_000.0      # vendas de ações no mês até este valor: lucro isento
ALIQ_RV = 0.15                # IR sobre lucro em ações/ETF (operações comuns)
ETFS = {"BOVA11", "IBOV"}     # sem isenção dos R$ 20 mil
PESO_MAX = 2.0                # alternativa 3: no máximo 2× a fatia igual


def aliquota_rf(dias):
    """IR regressivo da renda fixa pelo prazo em dias corridos."""
    return 0.225 if dias <= 180 else 0.20 if dias <= 360 else 0.175 if dias <= 720 else 0.15


# ----------------------------------------------------------------------------
# Carteira: caixa no CDI (lotes), posições, IR
# ----------------------------------------------------------------------------
class Carteira:
    def __init__(self, ir=True, custo=CUSTO):
        self.ir, self.custo = ir, custo
        self.lotes = []          # [data da aplicação, principal, valor atual]
        self.pos = {}            # ativo -> [quantidade, custo total (com taxa), índice da compra]
        self.prejuizo = 0.0      # prejuízo a compensar (renda variável)
        self.ir_pago = 0.0
        self.vendas = 0
        self.ordens = []         # (índice do pregão, ativo, "compra"/"venda", valor)
        self._zerar_mes()

    def _zerar_mes(self):
        self.mes = {"vendas_acoes": 0.0, "res_acoes": 0.0, "res_etf": 0.0}

    # --- caixa (CDI) ---
    def render(self, r):
        for lote in self.lotes:
            lote[2] *= 1 + r

    def aplicar(self, data, valor):
        if valor > 1e-9:
            self.lotes.append([data, valor, valor])

    def _ir_lote(self, lote, data):
        if not self.ir:
            return 0.0
        return aliquota_rf((data - lote[0]).days) * max(lote[2] - lote[1], 0.0)

    def caixa_bruto(self):
        return sum(l[2] for l in self.lotes)

    def caixa_liquido(self, data):
        return sum(l[2] - self._ir_lote(l, data) for l in self.lotes)

    def resgatar(self, data, liquido):
        """Resgata do lote mais antigo até obter `liquido` (já sem IR). Devolve o obtido."""
        obtido = 0.0
        while self.lotes and obtido < liquido - 1e-9:
            lote = self.lotes[0]
            imposto = self._ir_lote(lote, data)
            liq = lote[2] - imposto
            falta = liquido - obtido
            if liq <= falta:
                obtido += liq
                self.ir_pago += imposto
                self.lotes.pop(0)
            else:
                f = falta / liq
                self.ir_pago += f * imposto
                lote[1] *= 1 - f
                lote[2] *= 1 - f
                obtido += falta
        return obtido

    # --- ações / ETF ---
    def comprar(self, ativo, preco, valor, i):
        if valor <= 1e-9:
            return
        self.ordens.append((i, ativo, "compra", valor))
        p = self.pos.setdefault(ativo, [0.0, 0.0, i])
        p[0] += valor * (1 - self.custo) / preco
        p[1] += valor                       # custo de aquisição inclui a taxa

    def vender(self, ativo, preco, i=None):
        q, custo_total, _ = self.pos.pop(ativo)
        bruto = q * preco
        liquido = bruto * (1 - self.custo)
        resultado = liquido - custo_total
        if ativo in ETFS:
            self.mes["res_etf"] += resultado
        else:
            self.mes["vendas_acoes"] += bruto
            self.mes["res_acoes"] += resultado
        self.vendas += 1
        self.ordens.append((i, ativo, "venda", liquido))
        return liquido

    def apurar_ir(self):
        """Fim do mês: IR sobre o resultado em renda variável (isenção e prejuízo)."""
        m = self.mes
        self._zerar_mes()
        if not self.ir:
            return 0.0
        isento = m["vendas_acoes"] <= ISENCAO_ACOES
        # Em mês isento o lucro com ações não é tributado, mas o prejuízo é compensável
        base = m["res_etf"] + (min(m["res_acoes"], 0.0) if isento else m["res_acoes"])
        base -= self.prejuizo
        if base > 0:
            self.prejuizo = 0.0
            return ALIQ_RV * base
        self.prejuizo = -base
        return 0.0

    def patrimonio(self, precos):
        return self.caixa_bruto() + sum(q * precos[a] for a, (q, _, _) in self.pos.items())

    def liquidar(self, data, precos, ir_devido=0.0):
        """Valor líquido se vendesse tudo hoje: IR da renda variável e do CDI pagos."""
        c = copy.deepcopy(self)
        recebido = sum(c.vender(a, precos[a]) for a in list(c.pos))
        return c.caixa_liquido(data) + recebido - c.apurar_ir() - ir_devido


# ----------------------------------------------------------------------------
# Dados
# ----------------------------------------------------------------------------
def preparar(precos, ativos):
    """Sinais e níveis da regra por ativo, no calendário comum (B3)."""
    cal = precos.index
    d = {"px": precos[ativos].ffill()}
    for nome in ("ativo", "pronto", "teto", "dist", "z"):
        d[nome] = {}
    for a in ativos:
        s = serie(precos, a)
        mu, sd = bandas_moveis(s)
        teto = mu + K_STOP_GAIN * sd
        d["ativo"][a] = sinal_compra(precos, a)["ativo"].reindex(cal, fill_value=False)
        d["pronto"][a] = pd.Series(np.arange(len(s)) >= INICIO, index=s.index
                                   ).reindex(cal, fill_value=False)   # bandas completas
        d["teto"][a] = teto.reindex(cal)
        d["dist"][a] = ((teto - s) / sd).reindex(cal)                 # espaço até o gain, em σ
        d["z"][a] = ((s - mu) / sd).reindex(cal)                      # σ da média
    for nome in ("ativo", "pronto", "teto", "dist", "z"):
        d[nome] = pd.DataFrame(d[nome])
    return d


def estrategias(nucleo, universo):
    sat = [a for a in universo if a != nucleo]
    return {
        "Regra do app": dict(tipo="regra", peso="igual", universo=universo),
        "3) Regra, peso pelo espaço": dict(tipo="regra", peso="espaco", universo=universo),
        f"1) 50% {nucleo} + 50% regra": dict(tipo="nucleo", frac=0.5, universo=sat),
        f"1) 70% {nucleo} + 30% regra": dict(tipo="nucleo", frac=0.7, universo=sat),
        "2) Aporte dirigido": dict(tipo="dirigido", universo=universo),
        f"Tudo em {nucleo}": dict(tipo="dca", pesos={nucleo: 1.0}),
        "Tudo no CDI": dict(tipo="cdi"),
        "1/N em cada ação (teórico)": dict(tipo="dca", pesos={a: 1.0 for a in universo
                                                             if a not in ETFS}),
    }


# ----------------------------------------------------------------------------
# Simulação
# ----------------------------------------------------------------------------
def simular(est, d, cal, r_cdi, nucleo, ir=True, aporte=APORTE):
    """Simula uma estratégia dia a dia. Devolve patrimônio, cota e métricas."""
    c = Carteira(ir=ir)
    px, ativo, pronto = d["px"], d["ativo"], d["pronto"]
    meses = cal.to_period("M")
    primeiro = np.r_[True, meses[1:] != meses[:-1]]
    ultimo = np.r_[meses[:-1] != meses[1:], True]
    tipo, universo = est["tipo"], est.get("universo", [])
    devido, cotas = 0.0, 0.0
    pat, cota, caixa_frac, fluxos = [], [], [], []

    def disponiveis(i, lista):
        return [a for a in lista if pronto.at[cal[i], a] and pd.notna(px.at[cal[i], a])]

    def compra_do_caixa(a, valor, i):
        obtido = c.resgatar(cal[i], valor)
        c.comprar(a, px.at[cal[i], a], obtido, i)

    for i, t in enumerate(cal):
        c.render(r_cdi[i])
        p_t = px.loc[t]

        # 1) Saídas pelo stop gain (só posições da regra; a partir do pregão seguinte)
        if tipo in ("regra", "nucleo"):
            for a in [a for a in c.pos if a in universo]:
                if i > c.pos[a][2] and p_t[a] > d["teto"].at[t, a]:
                    c.aplicar(t, c.vender(a, p_t[a], i))

        # 2) Aporte do mês (primeiro abate IR que ficou devido)
        if primeiro[i]:
            p_antes = c.patrimonio(p_t)
            cotas += aporte if cotas == 0 else aporte / (p_antes / cotas)
            fluxos.append((t, aporte))
            pago = min(aporte, devido)
            devido -= pago
            c.ir_pago += pago
            a_liq = aporte - pago
            if tipo in ("regra", "cdi"):
                c.aplicar(t, a_liq)
            elif tipo == "nucleo":
                c.comprar(nucleo, p_t[nucleo], a_liq * est["frac"], i)
                c.aplicar(t, a_liq * (1 - est["frac"]))
            elif tipo == "dca":
                alvo = {a: w for a, w in est["pesos"].items() if pd.notna(p_t.get(a, np.nan))}
                soma = sum(alvo.values())
                for a, w in alvo.items():
                    c.comprar(a, p_t[a], a_liq * w / soma, i)
            elif tipo == "dirigido":
                cand = disponiveis(i, universo)
                sinais = [a for a in cand if ativo.at[t, a]]
                if sinais:
                    for a in sinais:
                        c.comprar(a, p_t[a], a_liq / len(sinais), i)
                elif cand:
                    a = min(cand, key=lambda x: d["z"].at[t, x])
                    c.comprar(a, p_t[a], a_liq, i)

        # 3) Entradas da regra: COMPRAR em ativo sem posição
        if tipo in ("regra", "nucleo"):
            for a in disponiveis(i, universo):
                if a in c.pos or not ativo.at[t, a]:
                    continue
                livres = [x for x in disponiveis(i, universo) if x not in c.pos]
                caixa = c.caixa_liquido(t)
                valor = caixa / len(livres)
                if est.get("peso") == "espaco":
                    fator = float(np.clip(d["dist"].at[t, a] / K_STOP_GAIN, 0.0, PESO_MAX))
                    valor = min(caixa, valor * fator)
                if valor >= 1.0:
                    compra_do_caixa(a, valor, i)

        # 4) IR de renda variável no fim do mês (pago do caixa; o que faltar, do aporte)
        if ultimo[i]:
            devido += c.apurar_ir()
            if devido > 0:
                obtido = c.resgatar(t, devido)
                c.ir_pago += obtido
                devido -= obtido

        valor = c.patrimonio(p_t)
        pat.append(valor)
        cota.append(valor / cotas if cotas else 1.0)
        caixa_frac.append(c.caixa_bruto() / valor if valor else 1.0)

    liquido = c.liquidar(cal[-1], px.loc[cal[-1]], devido)
    ordens = [(cal[i] if i is not None else None, a, o, v) for i, a, o, v in c.ordens]
    return {"patrimonio": pd.Series(pat, index=cal), "cota": pd.Series(cota, index=cal),
            "caixa": pd.Series(caixa_frac, index=cal), "fluxos": fluxos,
            "liquido": liquido, "ir_pago": c.ir_pago, "vendas": c.vendas,
            "ordens": pd.DataFrame(ordens, columns=["data", "ativo", "ordem", "valor"])}


def tir(fluxos, data_final, valor_final):
    """Rentabilidade anual equivalente dos aportes (TIR por bisseção)."""
    def vp(r):
        return valor_final - sum(v * (1 + r) ** ((data_final - d).days / 365.25)
                                 for d, v in fluxos)
    lo, hi = -0.99, 5.0
    if vp(lo) * vp(hi) > 0:
        return np.nan
    for _ in range(200):
        mid = (lo + hi) / 2
        if vp(lo) * vp(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def rodar_carteira(periodo="10y", inicio=None, fim=None, nucleo="BOVA11", ir=True,
                   precos=None, cdi=None, aporte=APORTE):
    """
    Roda todas as estratégias no mesmo período. `inicio`/`fim` (texto 'AAAA-MM-DD')
    recortam a avaliação; os indicadores usam todo o histórico anterior.
    """
    if precos is None:
        tickers = ACOES + (["^BVSP"] if nucleo == "IBOV" else [])
        precos = baixar_precos(tickers, periodo=periodo)
    universo = [a for a in precos.columns if a != "IBOV"]
    ativos = universo + (["IBOV"] if nucleo == "IBOV" else [])
    d = preparar(precos, ativos)

    # Avaliação: a partir do 1º pregão com o núcleo e algum ativo do universo prontos
    ok = d["pronto"][nucleo] & d["pronto"][universo].any(axis=1)
    cal = precos.index[ok.to_numpy()]
    if inicio:
        cal = cal[cal >= pd.Timestamp(inicio)]
    if fim:
        cal = cal[cal <= pd.Timestamp(fim)]
    for k in ("px", "ativo", "pronto", "teto", "dist", "z"):
        d[k] = d[k].loc[cal]

    aviso_cdi = None
    if cdi is None:
        try:
            cdi = baixar_cdi(cal[0] - pd.Timedelta(days=10), cal[-1])
        except Exception as e:
            aviso_cdi = f"CDI indisponível ({e}); caixa rendendo 0%."
    r_cdi = _cdi_nas_datas(cdi, cal).to_numpy()

    linhas, series = {}, {}
    for nome, est in estrategias(nucleo, universo).items():
        r = simular(est, d, cal, r_cdi, nucleo, ir=ir, aporte=aporte)
        aportado = sum(v for _, v in r["fluxos"])
        linhas[nome] = {
            "Aportado":        aportado,
            "Patrimônio":      r["patrimonio"].iloc[-1],
            "Líquido (vende)": r["liquido"],
            "TIR líq. a.a.":   tir(r["fluxos"], cal[-1], r["liquido"]),
            "Pior queda":      (r["cota"] / r["cota"].cummax() - 1).min(),
            "IR pago":         r["ir_pago"],
            "Vendas":          r["vendas"],
            "Caixa médio":     r["caixa"].mean(),
        }
        series[nome] = r
    tabela = pd.DataFrame(linhas).T
    return {"tabela": tabela, "series": series, "periodo": (cal[0], cal[-1]),
            "nucleo": nucleo, "ir": ir, "aviso_cdi": aviso_cdi,
            "falhas": list(precos.attrs.get("falhas", []))}


# ----------------------------------------------------------------------------
# Apresentação
# ----------------------------------------------------------------------------
def _reais(v):
    return f"{v:,.0f}".replace(",", ".")


def mostrar_carteira(res):
    ini, fim = res["periodo"]
    print(f"Carteira com aporte de R$ {_reais(APORTE)} no 1º pregão de cada mês, "
          f"{ini:%m/%Y} a {fim:%m/%Y}. Stop gain +{K_STOP_GAIN:g}σ, sem stop loss; "
          f"custo {CUSTO:.2%} por ordem; IR {'incluído' if res['ir'] else 'fora'}."
          .replace("0.05%", "0,05%").replace("+1.8σ", "+1,8σ"))
    if res["falhas"]:
        print("⚠️ Sem dados do Yahoo para:", ", ".join(res["falhas"]))
    if res["aviso_cdi"]:
        print("⚠️", res["aviso_cdi"])
    t = res["tabela"].sort_values("TIR líq. a.a.", ascending=False)
    out = pd.DataFrame(index=t.index)
    for c in ("Aportado", "Patrimônio", "Líquido (vende)", "IR pago"):
        out[c] = t[c].map(lambda v: "R$ " + _reais(v))
    out["TIR líq. a.a."] = t["TIR líq. a.a."].map(lambda v: f"{v:+.1%}".replace(".", ","))
    out["Pior queda"] = t["Pior queda"].map(lambda v: f"{v:.1%}".replace(".", ","))
    out["Vendas"] = t["Vendas"].astype(int)
    out["Caixa médio"] = t["Caixa médio"].map(lambda v: f"{v:.0%}")
    pd.set_option("display.width", 250)
    print()
    print(out.to_string())
    print("\nLíquido (vende) = patrimônio se vendesse tudo no último pregão, pagando o IR. "
          "TIR = rentabilidade anual equivalente dos aportes, líquida.\nPior queda = "
          "maior queda da cota (sem o efeito dos aportes). Caixa médio = parte média do "
          "patrimônio no CDI.")


def plot_carteira(res, nomes=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 6))
    for nome, r in res["series"].items():
        if nomes is None or nome in nomes:
            ax.plot(r["patrimonio"], label=nome, lw=1.4)
    aportes = pd.Series([v for _, v in next(iter(res["series"].values()))["fluxos"]],
                        index=[d for d, _ in next(iter(res["series"].values()))["fluxos"]])
    ax.plot(aportes.cumsum(), color="black", ls="--", lw=1, label="Total aportado")
    ax.set_title("Patrimônio bruto — aporte mensal")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    plt.show()
    return fig


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "2007":
        mostrar_carteira(rodar_carteira(periodo="max", inicio="2007-01-01",
                                        fim="2017-10-13", nucleo="IBOV"))
    else:
        mostrar_carteira(rodar_carteira())
