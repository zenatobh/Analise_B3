"""
alerta.py — envia pelo Telegram as entradas e saídas do último pregão.

Roda todo dia útil pelo GitHub Actions (.github/workflows/alerta.yml), com a
mesma regra do app (analise_ativos.operacoes), então alerta e gráfico não divergem.

Variáveis de ambiente (segredos do repositório no GitHub):
    TELEGRAM_TOKEN    token do bot (BotFather)
    TELEGRAM_CHAT_ID  id do chat que recebe as mensagens

Uso:
    python alerta.py            # envia só se houve entrada/saída hoje
    python alerta.py --teste    # envia o status atual de todos os ativos (teste)
"""
import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from analise_ativos import (K_STOP_GAIN, K_STOP_LOSS, TICKERS, baixar_precos,
                            moeda, operacoes, status_operacao)

FUSO = ZoneInfo("America/Sao_Paulo")
URL_APP = "https://ativos-b3.streamlit.app"   # ajuste se o endereço do app for outro


def pct(x):
    return f"{x:+.2%}".replace(".", ",").replace("-", "−")


def sigma(k):
    return f"{k:+g}σ".replace(".", ",").replace("-", "−")


def enviar(texto):
    token, chat = os.environ["TELEGRAM_TOKEN"], os.environ["TELEGRAM_CHAT_ID"]
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


def eventos_do_dia(precos, hoje):
    """Entradas e saídas cujo pregão é `hoje` (data local de São Paulo)."""
    msgs = []
    for ticker in precos.columns:
        ops = operacoes(precos, ticker)
        if ops.empty:
            continue
        op = ops.iloc[-1]
        st = status_operacao(precos, ticker)
        if op["entrada"].date() == hoje and pd.isna(op["saida"]):
            msgs.append(
                f"🟢 <b>ENTRADA {ticker}</b> a {moeda(op['preco_entrada'])}\n"
                f"   stop gain ({sigma(K_STOP_GAIN)}): {moeda(st['stop_gain'])}\n"
                f"   stop loss ({sigma(K_STOP_LOSS)}): {moeda(st['stop_loss'])}")
        elif pd.notna(op["saida"]) and op["saida"].date() == hoje:
            icone, nome = ("✅", "stop gain") if op["motivo"] == "gain" else ("🔴", "stop loss")
            msgs.append(
                f"{icone} <b>SAÍDA {ticker}</b> ({nome}) a {moeda(op['preco_saida'])}\n"
                f"   entrada {op['entrada']:%d/%m/%y} a {moeda(op['preco_entrada'])} · "
                f"resultado {pct(op['resultado'])}")
    return msgs


def resumo_teste(precos):
    linhas = ["🧪 <b>Teste do alerta</b> — situação atual:"]
    for ticker in precos.columns:
        st = status_operacao(precos, ticker)
        if st["aberta"]:
            op = st["op"]
            linhas.append(f"🟢 {ticker}: comprado desde {op['entrada']:%d/%m/%y} "
                          f"({pct(op['resultado'])})")
        else:
            linhas.append(f"⚪ {ticker}: sem operação")
    return "\n".join(linhas)


def main():
    precos = baixar_com_retentativa()
    ultimo = precos.index[-1].date()
    hoje = datetime.now(FUSO).date()

    if "--teste" in sys.argv:
        enviar(resumo_teste(precos) + f"\n\nDados até {ultimo:%d/%m/%Y}")
        print("Mensagem de teste enviada.")
        return

    msgs = eventos_do_dia(precos, hoje)
    if not msgs:
        print(f"Sem entradas/saídas em {hoje:%d/%m/%Y} (dados até {ultimo:%d/%m/%Y}).")
        return
    enviar(f"📈 <b>Alertas de {hoje:%d/%m/%Y}</b>\n\n" + "\n\n".join(msgs)
           + f"\n\n<a href='{URL_APP}'>Abrir o app</a>")
    print(f"{len(msgs)} alerta(s) enviado(s).")


if __name__ == "__main__":
    main()
