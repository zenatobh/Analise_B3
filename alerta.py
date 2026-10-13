"""
alerta.py — envia pelo Telegram os sinais do pregão do dia.

Roda de segunda a sexta às 19h30 pelo GitHub Actions (.github/workflows/alerta.yml),
com a mesma regra do app (analise_ativos.sinal_hoje), então alerta e app não divergem.

Sinais (stop gain primeiro):
    🟢 COMPRAR     — todo dia em que as três condições de entrada valem
    ✅ STOP GAIN   — a operação fechou acima de μ + 1,8σ hoje (não há stop loss)
Pregão de referência = data de 12 h atrás (Brasília): a execução das 19h30 aponta
para o próprio dia e uma execução que o GitHub atrase para a madrugada continua
apontando para o pregão certo. Ativos sem dado nessa data ficam de fora, para não
repetir sinais do dia anterior; os da B3 nessa situação são avisados na mensagem.

Toda execução agendada envia uma mensagem — sinais, "sem sinais" ou "sem pregão na
B3" —, então silêncio no Telegram significa que o alerta não rodou.

Variáveis de ambiente (segredos do repositório no GitHub):
    TELEGRAM_TOKEN    token do bot (BotFather)
    TELEGRAM_CHAT_ID  id do chat que recebe as mensagens

Uso:
    python alerta.py            # sinais do pregão de referência (sempre envia mensagem)
    python alerta.py --teste    # envia a situação de todos os ativos (teste)
"""
import os
import sys
import time
from datetime import datetime, timedelta

import requests

from analise_ativos import (FUSO, K_STOP_GAIN, TICKERS, baixar_precos, falhas, moeda,
                            serie, sinal_hoje)

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
    if sem_b3:
        # Nenhum ativo da B3 com dado do dia: feriado ou Yahoo ainda sem o fechamento
        ultimo_b3 = max(serie(precos, t).index[-1] for t in b3)
        partes = [f"📭 <b>Sem pregão na B3 em {dia:%d/%m/%Y}</b>\n"
                  f"Feriado ou dados ainda não publicados pelo Yahoo "
                  f"(B3 até {ultimo_b3:%d/%m/%Y})."] + msgs
    else:
        titulo = "Sinais de" if msgs else "Sem sinais em"
        partes = [f"📈 <b>{titulo} {dia:%d/%m/%Y}</b>"] + msgs
    if faltando and not sem_b3:
        partes.append(f"⚠️ Sem o pregão de {dia:%d/%m} no Yahoo: {', '.join(faltando)} "
                      "(sinais desses ativos não avaliados)")
    if aviso:
        partes.append(aviso)
    enviar("\n\n".join(partes) + f"\n\n<a href='{URL_APP}'>Abrir o app</a>")
    print(f"{dia:%d/%m/%Y}: {len(msgs)} sinal(is) enviado(s)"
          + ("; sem pregão na B3" if sem_b3 else
             f"; sem pregão: {', '.join(faltando)}" if faltando else "")
          + (f"; {aviso}" if aviso else "") + ".")


if __name__ == "__main__":
    main()
