"""
alerta.py — envia pelo Telegram os sinais do pregão do dia.

Roda de segunda a sexta às 19h47 pelo GitHub Actions (.github/workflows/alerta.yml),
com a mesma regra do app (analise_ativos.sinal_hoje), então alerta e app não divergem.

A mensagem separa as duas estratégias:
    🅰️ Modo A — operação com stop gain (stop gain primeiro):
        🟢 COMPRAR     — todo dia em que as três condições de entrada valem
        ✅ STOP GAIN   — a operação fechou acima de μ + 1,8σ hoje (não há stop loss);
                         vale só para o que foi comprado pelo modo A
    IBOV, S&P 500, dólar e BOVA11 são só acompanhamento: nunca geram sinal.
    IVVB11 não gera sinal do modo A, mas concorre no aporte do mês (modo B).
    🅱️ Modo B — aporte do mês (só no 1º pregão da B3 a partir do dia 15 de cada mês;
       dia 15 em fim de semana ou feriado → pregão seguinte — DIA_APORTE):
        💰 onde aportar: COMPRAR hoje (dividindo) ou o mais abaixo da média; nunca vende
Pregão de referência = data de 12 h atrás (Brasília): a execução das 19h47 aponta
para o próprio dia e uma execução que o GitHub atrase para a madrugada continua
apontando para o pregão certo. Ativos sem dado nessa data ficam de fora, para não
repetir sinais do dia anterior; os da B3 nessa situação são avisados na mensagem.

Toda execução agendada envia uma mensagem — sinais, "sem sinais" ou "sem pregão na
B3" —, então silêncio no Telegram significa que o alerta não rodou.

Variáveis de ambiente (segredos do repositório no GitHub):
    TELEGRAM_TOKEN    token do bot (BotFather)
    TELEGRAM_CHAT_ID  id do chat que recebe as mensagens
    BRAPI_TOKEN       opcional: reserva da cotação do dia quando o Yahoo atrasa

Uso:
    python alerta.py            # sinais do pregão de referência (sempre envia mensagem)
    python alerta.py --teste    # envia a situação de todos os ativos (teste)
"""
import os
import sys
import time
from datetime import datetime, timedelta

import requests

from analise_ativos import (DIA_APORTE, FUSO, K_STOP_GAIN, SO_MODO_B, TICKERS,
                            aporte_do_mes, baixar_precos, e_dia_de_aporte, falhas, moeda,
                            no_modo_b, reserva, serie, sinal_hoje)

URL_APP = "https://ativos-b3.streamlit.app"   # ajuste se o endereço do app for outro

FORA_B3 = {"SP500", "USDBRL"}       # negociados fora da B3 (outro calendário)
ATRASO_MAX = timedelta(hours=12)    # tolerância a atrasos do agendamento do GitHub


def pregao_referencia(agora=None):
    """Data do pregão que o alerta avalia: o dia de 12 h atrás, em Brasília."""
    return ((agora or datetime.now(FUSO)) - ATRASO_MAX).date()


def pct(x):
    return f"{x:+.2%}".replace(".", ",").replace("-", "−")


def sigma(k):
    return f"{k:+g}σ".replace(".", ",").replace("-", "−")


def enviar(texto):
    # .strip(): um "\n" colado no fim do segredo gera 404 na API do Telegram
    token = os.environ["TELEGRAM_TOKEN"].strip()
    chat = os.environ["TELEGRAM_CHAT_ID"].strip()
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id": chat, "text": texto, "parse_mode": "HTML",
                            "disable_web_page_preview": True}, timeout=30)
    r.raise_for_status()


def baixar_com_retentativa(tentativas=3, espera=60):
    """O Yahoo às vezes limita acessos vindos de servidores na nuvem."""
    for i in range(tentativas):
        try:
            return baixar_precos(TICKERS)
        except Exception as e:
            if i == tentativas - 1:
                raise
            print(f"Falha no download ({e}); nova tentativa em {espera}s")
            time.sleep(espera)


def em_sigma(v):
    v = round(v, 1) + 0.0        # evita "−0,0σ"
    return f"{v:+.1f}σ".replace(".", ",").replace("-", "−").replace("+0,0", "0,0")


def dia_txt(n):
    return "1º dia" if n == 1 else f"{n}º dia seguido"


def texto_sinal(ticker, sh):
    """Mensagem de um ativo com sinal (None se não houver sinal)."""
    if sh["sinal"] == "comprar":
        return (f"🟢 <b>COMPRAR {ticker}</b> a {moeda(sh['preco'])} · {dia_txt(sh['dia_n'])}\n"
                f"   stop gain ({sigma(K_STOP_GAIN)}): {moeda(sh['stop_gain'])} "
                f"({pct(sh['ate_gain'])})")
    if sh["sinal"] == "gain":
        op = sh["op"]
        return (f"✅ <b>STOP GAIN {ticker}</b> a {moeda(op['preco_saida'])}\n"
                f"   compra {op['entrada']:%d/%m/%y} a {moeda(op['preco_entrada'])} · "
                f"resultado {pct(op['resultado'])} em {op['dias']} pregões")
    return None


def tem_pregao(precos, ticker, dia):
    """True se o último dado do ativo é do dia de referência."""
    return serie(precos, ticker).index[-1].date() == dia


def sinais_do_dia(precos, dia):
    """Mensagens dos ativos com pregão no dia e algum sinal; STOP GAIN antes de COMPRAR."""
    stops, compras = [], []
    for ticker in precos.columns:
        if not tem_pregao(precos, ticker, dia):
            continue
        sh = sinal_hoje(precos, ticker)
        msg = texto_sinal(ticker, sh)
        if msg:
            (compras if sh["sinal"] == "comprar" else stops).append(msg)
    return stops + compras


TITULO_A = "🅰️ <b>Modo A — operação com stop gain</b>"
TITULO_B = "🅱️ <b>Modo B — aporte do mês</b>"
NOTA_A = "<i>Venda no STOP GAIN só o que foi comprado pelo modo A.</i>"
NOTA_B = "<i>Comprar e não vender — o STOP GAIN do modo A não vale aqui.</i>"


def bloco_a(msgs):
    """Bloco do modo A: sinais do dia (ou 'sem sinais')."""
    if not msgs:
        return f"{TITULO_A}\nSem sinais hoje."
    tem_stop = any("STOP GAIN" in m for m in msgs)
    return "\n\n".join([TITULO_A] + msgs + ([NOTA_A] if tem_stop else []))


def bloco_b(precos, ativos):
    """Bloco do modo B: onde aportar hoje e as 3 seguintes da fila."""
    ap = aporte_do_mes(precos, ativos)
    if not ap["ativos"]:
        return None
    rk = ap["ranking"]
    if ap["motivo"] == "comprar":
        onde = " e ".join(f"<b>{a}</b>" for a in ap["ativos"])
        dias = ", ".join(f"{a} {dia_txt(int(rk.at[a, 'dia_n']))}" for a in ap["ativos"])
        motivo = f"COMPRAR hoje ({dias})" + (
            "; dividir o aporte igualmente" if len(ap["ativos"]) > 1 else "")
    else:
        onde = f"<b>{ap['ativos'][0]}</b>"
        motivo = f"a mais abaixo da média ({em_sigma(rk.at[ap['ativos'][0], 'sigma'])})"
    seguintes = [a for a in rk.index if a not in ap["ativos"]][:3]
    fila = " · ".join(f"{a} ({em_sigma(rk.at[a, 'sigma'])})" for a in seguintes)
    return (f"{TITULO_B}\n💰 Aportar em {onde} — {motivo}\n"
            f"   seguintes: {fila}\n{NOTA_B}")


def ativos_b3_no_dia(precos, dia):
    return [t for t in precos.columns if t not in FORA_B3 and tem_pregao(precos, t, dia)]


def resumo_teste(precos):
    linhas = ["🧪 <b>Teste do alerta</b> — situação de todos os ativos", TITULO_A]
    for ticker in precos.columns:
        sh = sinal_hoje(precos, ticker)
        data = f" ({sh['data']:%d/%m})"
        msg = texto_sinal(ticker, sh)
        if sh["acompanhamento"]:
            linhas.append(f"👁 {ticker}: " + ("só no aporte do mês (modo B)"
                          if ticker in SO_MODO_B else "só acompanhamento") + data)
        else:
            linhas.append(msg + data if msg else
                          f"⚪ {ticker}: sem sinal{data} — falta {' e '.join(sh['faltam'])}")
    b = bloco_b(precos, [t for t in precos.columns if t not in FORA_B3])
    if b:
        linhas.append(b.replace("aporte do mês</b>",
                                "aporte do mês</b> (no teste, sempre; de verdade, só no "
                                f"1º pregão a partir do dia {DIA_APORTE})"))
    return "\n\n".join(linhas)


def aviso_falhas(precos):
    f, r = falhas(precos), reserva(precos)
    partes = ([f"⚠️ Sem dados do Yahoo para: {', '.join(f)}"] if f else []) + (
        [f"ℹ️ Pregão de hoje via brapi (o Yahoo ainda não tinha): {', '.join(r)}"] if r else [])
    return "\n".join(partes)


def b3_sem_pregao(precos, dia):
    """Ativos da B3 sem dado no dia de referência."""
    return [t for t in precos.columns if t not in FORA_B3 and not tem_pregao(precos, t, dia)]


def main():
    precos = baixar_com_retentativa()
    ultimo = precos.index[-1].date()
    dia = pregao_referencia()
    aviso = aviso_falhas(precos)

    if "--teste" in sys.argv:
        enviar(resumo_teste(precos) + (f"\n\n{aviso}" if aviso else "")
               + f"\n\nDados até {ultimo:%d/%m/%Y}")
        print("Mensagem de teste enviada.")
        return

    msgs = sinais_do_dia(precos, dia)      # inclui SP500/USDBRL se tiveram pregão no dia
    faltando = b3_sem_pregao(precos, dia)
    b3 = [t for t in precos.columns if t not in FORA_B3]
    sem_b3 = bool(b3) and len(faltando) == len(b3)
    aporte = False
    if sem_b3:
        # Nenhum ativo da B3 com dado do dia: feriado ou Yahoo ainda sem o fechamento
        ultimo_b3 = max(serie(precos, t).index[-1] for t in b3)
        partes = [f"📭 <b>Sem pregão na B3 em {dia:%d/%m/%Y}</b>\n"
                  f"Feriado ou dados ainda não publicados pelo Yahoo "
                  f"(B3 até {ultimo_b3:%d/%m/%Y})."]
        if msgs:                            # (hoje nunca: S&P 500 e dólar não têm sinais)
            partes.append(bloco_a(msgs))
    else:
        partes = [f"📈 <b>{dia:%d/%m/%Y}</b>", bloco_a(msgs)]
        if e_dia_de_aporte(precos, dia):
            b = bloco_b(precos, ativos_b3_no_dia(precos, dia))
            if b:
                partes.append(b)
                aporte = True
    faltando_sinais = [t for t in faltando if no_modo_b(t)]    # IBOV não tem sinais
    if faltando_sinais and not sem_b3:
        partes.append(f"⚠️ Sem o pregão de {dia:%d/%m} no Yahoo: {', '.join(faltando_sinais)} "
                      "(sinais desses ativos não avaliados)")
    if aviso:
        partes.append(aviso)
    enviar("\n\n".join(partes) + f"\n\n<a href='{URL_APP}'>Abrir o app</a>")
    print(f"{dia:%d/%m/%Y}: {len(msgs)} sinal(is) do modo A"
          + ("; aporte do mês (modo B)" if aporte else "")
          + ("; sem pregão na B3" if sem_b3 else
             f"; sem pregão: {', '.join(faltando)}" if faltando else "")
          + (f"; {aviso}" if aviso else "") + ".")


if __name__ == "__main__":
    main()
