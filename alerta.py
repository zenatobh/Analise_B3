"""
alerta.py — envia pelo Telegram os sinais do pregão do dia.

Roda de segunda a sexta às 19h30 pelo GitHub Actions (.github/workflows/alerta.yml),
com a mesma regra do app (analise_ativos.sinal_hoje), então alerta e app não divergem.

Sinais (stop gain primeiro):
    🟢 COMPRAR     — todo dia em que as três condições de entrada valem
    ✅ STOP GAIN   — a operação fechou acima de μ + 1,8σ hoje (não há stop loss)
Ativos cujo último pregão não é hoje (feriado no mercado deles, dado ainda não
publicado) ficam de fora, para não repetir sinais do dia anterior.

Variáveis de ambiente (segredos do repositório no GitHub):
    TELEGRAM_TOKEN    token do bot (BotFather)
    TELEGRAM_CHAT_ID  id do chat que recebe as mensagens

Uso:
    python alerta.py            # envia só se houver algum sinal hoje (ou falha de dados)
    python alerta.py --teste    # envia a situação de todos os ativos (teste)
"""
import os
import sys
import time
from datetime import datetime

import requests

from analise_ativos import (FUSO, K_STOP_GAIN, TICKERS, baixar_precos, falhas, moeda,
                            serie, sinal_hoje)

URL_APP = "https://ativos-b3.streamlit.app"   # ajuste se o endereço do app for outro


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


def texto_sinal(ticker, sh):
    """Mensagem de um ativo com sinal (None se não houver sinal)."""
    if sh["sinal"] == "comprar":
        dia = "1º dia" if sh["dia_n"] == 1 else f"{sh['dia_n']}º dia seguido"
        return (f"🟢 <b>COMPRAR {ticker}</b> a {moeda(sh['preco'])} · {dia}\n"
                f"   stop gain ({sigma(K_STOP_GAIN)}): {moeda(sh['stop_gain'])} "
                f"({pct(sh['ate_gain'])})")
    if sh["sinal"] == "gain":
        op = sh["op"]
        return (f"✅ <b>STOP GAIN {ticker}</b> a {moeda(op['preco_saida'])}\n"
                f"   compra {op['entrada']:%d/%m/%y} a {moeda(op['preco_entrada'])} · "
                f"resultado {pct(op['resultado'])} em {op['dias']} pregões")
    return None


def sinais_do_dia(precos, hoje):
    """Mensagens dos ativos com pregão hoje e algum sinal; STOP GAIN antes de COMPRAR."""
    stops, compras = [], []
    for ticker in precos.columns:
        if serie(precos, ticker).index[-1].date() != hoje:
            continue
        sh = sinal_hoje(precos, ticker)
        msg = texto_sinal(ticker, sh)
        if msg:
            (compras if sh["sinal"] == "comprar" else stops).append(msg)
    return stops + compras


def resumo_teste(precos):
    linhas = ["🧪 <b>Teste do alerta</b> — situação de todos os ativos:"]
    for ticker in precos.columns:
        sh = sinal_hoje(precos, ticker)
        data = f" ({sh['data']:%d/%m})"
        msg = texto_sinal(ticker, sh)
        linhas.append(msg + data if msg
                      else f"⚪ {ticker}: sem sinal{data} — falta {' e '.join(sh['faltam'])}")
    return "\n\n".join(linhas)


def aviso_falhas(precos):
    f = falhas(precos)
    return f"⚠️ Sem dados do Yahoo para: {', '.join(f)}" if f else ""


def main():
    precos = baixar_com_retentativa()
    ultimo = precos.index[-1].date()
    hoje = datetime.now(FUSO).date()
    aviso = aviso_falhas(precos)

    if "--teste" in sys.argv:
        enviar(resumo_teste(precos) + (f"\n\n{aviso}" if aviso else "")
               + f"\n\nDados até {ultimo:%d/%m/%Y}")
        print("Mensagem de teste enviada.")
        return

    msgs = sinais_do_dia(precos, hoje)
    if not msgs and not aviso:
        print(f"Sem sinais em {hoje:%d/%m/%Y} (dados até {ultimo:%d/%m/%Y}).")
        return
    titulo = "Sinais de" if msgs else "Sem sinais em"
    partes = [f"📈 <b>{titulo} {hoje:%d/%m/%Y}</b>"] + msgs
    if aviso:
        partes.append(aviso)
    enviar("\n\n".join(partes) + f"\n\n<a href='{URL_APP}'>Abrir o app</a>")
    print(f"{len(msgs)} sinal(is) enviado(s)" + (f"; {aviso}" if aviso else "") + ".")


if __name__ == "__main__":
    main()
