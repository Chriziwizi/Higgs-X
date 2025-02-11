#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Código unificado de Higgs (modo headless) para desplegar en Railway,
usando la API gratuita de CoinGecko para obtener datos de mercado,
calculando indicadores técnicos y generando señales de trading para el timeframe 1h
basadas en condiciones definidas (cruce de Bollinger Bands, medias móviles y RSI).
Incluye:
- Configuración global.
- Funciones para obtener datos OHLCV desde CoinGecko (usando days=7 para velas horarias).
- Funciones de indicadores técnicos.
- Nueva lógica de trading para señales de entrada en 1h con cálculo de Take Profit y Stop Loss.
- Funciones para generar gráficos y enviar mensajes por Telegram.
- Bot de Telegram para procesar mensajes (incluyendo consultas a OpenAI).
- Función principal start() que arranca los procesos en hilos.
"""

# =======================
# Sección 1: Configuración
# =======================
import sys
import time
import threading
import io
import re
from datetime import datetime
import pytz
import requests
import pandas as pd
import xgboost as xgb  # Se mantiene, pero la nueva lógica no depende de ML
import openai
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI
import matplotlib.pyplot as plt
import mplfinance as mpf

# Imprimir la IP pública (para test)
try:
    ip = requests.get("https://ifconfig.me").text.strip()
    print(f"La IP pública actual es: {ip}")
except Exception as e:
    print(f"No se pudo obtener la IP pública: {e}")

# Variables de configuración global
feature_columns = ['open', 'high', 'low', 'close', 'volume', 'sma_25', 'bb_low', 'bb_medium', 'bb_high']

# (Claves de Binance se mantienen para referencia y serán ignoradas)
API_KEY_BINANCE = 'C7xBOQLYAf597cakk21IldpGzTSvQ0CDoTPjoG9ZvssDXCjd21Y18IwbSj9fJuhP'
API_SECRET_BINANCE = 'khp4f2IdWOqloP98QU0mZz6VkmtJNfdAL9yL21RgZXGmppp75UmYvfWdpFS7ePL3'

# Configuración de Telegram
TELEGRAM_TOKEN = '8066635436:AAH2E-ZnwNvf7G-fskKOTZD3oVvuLt05v8U'
TELEGRAM_CHAT_ID = '-1002402692277'

# Clave API de CoinGecko (demo, se pone directo para pruebas)
COINGECKO_API_KEY = 'CG-9vur1PrpF89UrwBLERLsjEUL'

# OpenAI API key
OPENAI_API_KEY = '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'

# Otros parámetros
SYMBOL = 'BTC/USDT'      # Se mapea a "bitcoin" para CoinGecko
TIMEFRAME = '1h'         # Indicativo: usaremos datos de los últimos 7 días para obtener velas horarias
MAX_RETRIES = 5

# Para señales de trading, reducimos el cooldown a 5 minutos
last_trade_signal_time = None
COOLDOWN_SECONDS = 300  # 5 minutos

# Configuración de OpenAI
openai.api_key = OPENAI_API_KEY

# Tiempo de inicio para filtrar mensajes antiguos (Unix timestamp)
START_TIME = int(time.time())

# Diccionario de mapeo para convertir SYMBOL a coin_id de CoinGecko
COIN_ID_MAPPING = {
    "BTC/USDT": "bitcoin",
    "ETH/USDT": "ethereum"
}

# ================================
# Sección 2: Funciones para obtener datos desde CoinGecko
# ================================
def fetch_data(symbol=SYMBOL, timeframe=TIMEFRAME, days=7):
    """
    Obtiene datos OHLC y volúmenes para la criptomoneda usando la API gratuita de CoinGecko.
    Se usa days=7 para obtener velas horarias.
    Retorna un DataFrame con columnas: timestamp, open, high, low, close, volume.
    """
    coin_id = COIN_ID_MAPPING.get(symbol, "bitcoin")
    # Endpoint para OHLC (gratuito)
    url_ohlc = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days={days}"
    response = requests.get(url_ohlc)
    if response.status_code != 200:
        raise Exception(f"Error en CoinGecko OHLC: {response.text}")
    ohlc_data = response.json()
    df_ohlc = pd.DataFrame(ohlc_data, columns=["timestamp", "open", "high", "low", "close"])
    df_ohlc["timestamp"] = pd.to_datetime(df_ohlc["timestamp"], unit="ms")
    
    # Endpoint para market_chart (volúmenes, etc.) - gratuito
    url_chart = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}"
    response2 = requests.get(url_chart)
    if response2.status_code != 200:
        raise Exception(f"Error en CoinGecko market_chart: {response2.text}")
    chart_data = response2.json()
    volumes = chart_data.get("total_volumes", [])
    df_vol = pd.DataFrame(volumes, columns=["timestamp", "volume"])
    df_vol["timestamp"] = pd.to_datetime(df_vol["timestamp"], unit="ms")
    
    # Fusionar los datos OHLC y volúmenes usando merge_asof
    df_ohlc = df_ohlc.sort_values("timestamp")
    df_vol = df_vol.sort_values("timestamp")
    df = pd.merge_asof(df_ohlc, df_vol, on="timestamp", direction="nearest")
    return df

def fetch_chart_data(symbol=SYMBOL, timeframe="1h", days=7, limit=None):
    """
    Obtiene un DataFrame de datos y, si se especifica, limita el número de filas.
    """
    df = fetch_data(symbol, timeframe, days)
    if limit is not None:
        df = df.tail(limit)
    return df

# ================================
# Sección 3: Indicadores Técnicos
# ================================
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, ADXIndicator
from ta.volume import ChaikinMoneyFlowIndicator
from ta.volatility import BollingerBands

def calculate_indicators(data):
    """Calcula indicadores técnicos (RSI, ADX, SMAs, MACD, Bollinger Bands, CMF)."""
    close = data['close']
    high = data['high']
    low = data['low']
    volume = data['volume']
    
    cmf = ChaikinMoneyFlowIndicator(high, low, close, volume).chaikin_money_flow().iloc[-1]
    volume_level = "Alto" if cmf > 0.1 else "Bajo" if cmf < -0.1 else "Moderado"
    
    sma_10 = SMAIndicator(close, window=10).sma_indicator().iloc[-1]
    sma_25 = SMAIndicator(close, window=25).sma_indicator().iloc[-1]
    sma_50 = SMAIndicator(close, window=50).sma_indicator().iloc[-1]
    
    macd_indicator = MACD(close)
    macd = macd_indicator.macd().iloc[-1]
    macd_signal = macd_indicator.macd_signal().iloc[-1]
    
    rsi = RSIIndicator(close, window=14).rsi().iloc[-1]
    adx = ADXIndicator(high, low, close).adx().iloc[-1]
    
    bb_indicator = BollingerBands(close, window=20, window_dev=2)
    bb_low = bb_indicator.bollinger_lband().iloc[-1]
    bb_medium = bb_indicator.bollinger_mavg().iloc[-1]
    bb_high = bb_indicator.bollinger_hband().iloc[-1]
    
    indicators = {
        'price': close.iloc[-1],
        'rsi': rsi,
        'adx': adx,
        'macd': macd,
        'macd_signal': macd_signal,
        'sma_10': sma_10,
        'sma_25': sma_25,
        'sma_50': sma_50,
        'cmf': cmf,
        'volume_level': volume_level,
        'bb_low': bb_low,
        'bb_medium': bb_medium,
        'bb_high': bb_high,
        'prev_close': close.iloc[-2] if len(close) >= 2 else None
    }
    return indicators

# ================================
# Sección 4: Modelo ML (XGBoost)
# ================================
from ta.volatility import AverageTrueRange

MODEL = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

def add_extra_features(data):
    """Agrega indicadores extra (SMA25, Bollinger Bands, ATR) al DataFrame."""
    data = data.copy()
    data['sma_25'] = SMAIndicator(data['close'], window=25).sma_indicator()
    bb = BollingerBands(data['close'], window=20, window_dev=2)
    data['bb_low'] = bb.bollinger_lband()
    data['bb_medium'] = bb.bollinger_mavg()
    data['bb_high'] = bb.bollinger_hband()
    atr = AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=14)
    data['atr'] = atr.average_true_range()
    return data

def train_ml_model(data):
    """Entrena el modelo ML con datos históricos y características extra."""
    data = add_extra_features(data)
    features = data[feature_columns].pct_change().dropna()
    target = (features['close'] > 0).astype(int)
    MODEL.fit(features, target)

def predict_ml(data):
    """
    Predice la dirección (subida o caída) utilizando ML.
    (Esta función se mantiene para consulta, pero la nueva lógica usa condiciones técnicas.)
    """
    global LAST_STABLE_PREDICTION
    data = add_extra_features(data)
    features = data[feature_columns].pct_change().dropna().iloc[-1:][feature_columns]
    prob = MODEL.predict_proba(features)[0]
    if 0.45 < prob[1] < 0.55 and LAST_STABLE_PREDICTION is not None:
        prediction = LAST_STABLE_PREDICTION
    else:
        prediction = 1 if prob[1] >= 0.55 else 0
        LAST_STABLE_PREDICTION = prediction
    return '📈 Dirección xML: Subida Esperada' if prediction == 1 else '📉 Dirección xML: Caída Esperada'

# ================================
# Sección 5: Nueva Lógica de Trading para 1h
# ================================
def generate_trade_signal(data):
    """
    Genera una señal de trading basada en condiciones técnicas para velas de 1h.
    Condiciones para LONG:
      - El low de la última vela supera la banda superior (bb_high).
      - La SMA10 es mayor que SMA25.
      - RSI > 50.
    Condiciones para SHORT:
      - El high de la última vela está por debajo de la banda inferior (bb_low).
      - La SMA10 es menor que SMA25.
      - RSI < 50.
    Calcula el precio de entrada, stop loss y take profit (relación 1:2) y retorna el mensaje.
    Si no se cumplen, retorna None.
    """
    indicators = calculate_indicators(data)
    price = indicators['price']
    rsi = indicators['rsi']
    sma_10 = indicators['sma_10']
    sma_25 = indicators['sma_25']
    bb_low = indicators['bb_low']
    bb_high = indicators['bb_high']
    
    last_low = data['low'].iloc[-1]
    last_high = data['high'].iloc[-1]
    
    signal_message = None
    # Condición para LONG
    if last_low > bb_high and sma_10 > sma_25 and rsi > 50:
        entry = price
        stop_loss = last_low
        risk = entry - stop_loss
        take_profit = entry + 2 * risk
        signal_message = (f"🚀 Señal LONG (1h):\n"
                          f"Entrada: ${entry:.2f}\n"
                          f"Stop Loss: ${stop_loss:.2f}\n"
                          f"Take Profit: ${take_profit:.2f}\n"
                          f"(RSI: {rsi:.2f}, SMA10: {sma_10:.2f}, SMA25: {sma_25:.2f})")
    # Condición para SHORT
    elif last_high < bb_low and sma_10 < sma_25 and rsi < 50:
        entry = price
        stop_loss = last_high
        risk = stop_loss - entry
        take_profit = entry - 2 * risk
        signal_message = (f"📉 Señal SHORT (1h):\n"
                          f"Entrada: ${entry:.2f}\n"
                          f"Stop Loss: ${stop_loss:.2f}\n"
                          f"Take Profit: ${take_profit:.2f}\n"
                          f"(RSI: {rsi:.2f}, SMA10: {sma_10:.2f}, SMA25: {sma_25:.2f})")
    return signal_message

# ================================
# Sección 6: Gráficos y Envío a Telegram
# ================================
def send_graphic(chat_id, timeframe_input="1h", chart_type="line", days=7):
    """
    Genera un gráfico (lineal o de velas) a partir de los datos obtenidos vía CoinGecko
    y lo envía a Telegram.
    """
    try:
        df = fetch_chart_data(SYMBOL, timeframe_input, days=days, limit=100)
        support = df['close'].min()
        resistance = df['close'].max()
        sma20 = df['close'].rolling(window=20).mean()
        sma50 = df['close'].rolling(window=50).mean()
        buf = io.BytesIO()
        caption = f"Gráfico de {SYMBOL} - Últimos {days} día(s)"
        
        if chart_type.lower() == "candlestick":
            mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
            s  = mpf.make_mpf_style(marketcolors=mc, gridstyle="--")
            ap0 = mpf.make_addplot(sma20, color='blue', width=1.0, linestyle='-')
            ap1 = mpf.make_addplot(sma50, color='orange', width=1.0, linestyle='-')
            sr_support = [support] * len(df)
            sr_resistance = [resistance] * len(df)
            ap2 = mpf.make_addplot(sr_support, color='green', linestyle='--', width=0.8)
            ap3 = mpf.make_addplot(sr_resistance, color='red', linestyle='--', width=0.8)
            fig, _ = mpf.plot(
                df,
                type='candle',
                style=s,
                title=caption,
                volume=False,
                addplot=[ap0, ap1, ap2, ap3],
                returnfig=True
            )
            fig.suptitle(caption, y=0.95, fontsize=14)
            fig.savefig(buf, dpi=100, format='png')
            plt.close(fig)
        else:
            plt.figure(figsize=(10, 6))
            plt.plot(df.index, df['close'], label="Precio", color='black')
            plt.plot(df.index, sma20, label="SMA20", color='blue')
            plt.plot(df.index, sma50, label="SMA50", color='orange')
            plt.axhline(support, color='green', linestyle='--', label="Soporte")
            plt.axhline(resistance, color='red', linestyle='--', label="Resistencia")
            plt.title(caption, fontsize=14)
            plt.xlabel("Tiempo")
            plt.ylabel("Precio")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.savefig(buf, format="png")
            plt.close()
        
        buf.seek(0)
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        files = {'photo': buf}
        data_payload = {'chat_id': chat_id, 'caption': caption}
        response = requests.post(url, data=data_payload, files=files)
        if response.status_code != 200:
            print(f"Error al enviar el gráfico: {response.text}")
    except Exception as e:
        print(f"Error en send_graphic: {e}")

def send_telegram_message(message, chat_id=None):
    """Envía un mensaje a Telegram."""
    if not chat_id:
        chat_id = TELEGRAM_CHAT_ID
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': chat_id, 'text': message, 'parse_mode': 'Markdown'}
    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            print(f"Error al enviar mensaje a Telegram: {response.text}")
    except Exception as e:
        print(f"Error en la conexión con Telegram: {e}")

def handle_telegram_message(update):
    """
    Procesa los mensajes recibidos en Telegram.
    Si se detecta una petición de gráfico, llama a send_graphic;
    en otro caso, si se trata de una consulta, usa OpenAI para responder.
    """
    message_obj = update.get("message", {})
    message_text = message_obj.get("text", "").strip()
    chat_id = message_obj.get("chat", {}).get("id")
    username = message_obj.get("from", {}).get("username", "Agente")
    message_date = message_obj.get("message", {}).get("date", 0)
    if message_date < START_TIME:
        return
    if not message_text or not chat_id:
        return
    lower_msg = message_text.lower()
    
    if any(phrase in lower_msg for phrase in ["grafico", "gráfico"]):
        timeframe = "1h"  # Para gráficos, siempre 1h
        chart_type = "line"
        if any(keyword in lower_msg for keyword in ["vela", "velas", "candlestick", "japonesas"]):
            chart_type = "candlestick"
        send_graphic(chat_id, timeframe, chart_type, days=7)
        return

    # Si se trata de una consulta para OpenAI:
    context = (
        f"Hola agente @{username}, aquí Higgs X. Indicadores técnicos de {SYMBOL} (1h):\n"
        f"(Consulta: {message_text})"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": (
                    "Eres Higgs X, el agente encargado de vigilar el mercado. Responde de forma concisa, seria y con un toque de misterio."
                )},
                {"role": "user", "content": context}
            ],
            max_tokens=500,
            temperature=0.7
        )
        answer = response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        answer = f"⚠️ Error al procesar la solicitud: {e}"
    send_telegram_message(answer, chat_id)

def get_updates():
    """Obtiene las actualizaciones (mensajes) de Telegram."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            updates = response.json().get("result", [])
            return [upd for upd in updates if upd.get("message", {}).get("date", 0) >= START_TIME]
        else:
            print(f"Error al obtener actualizaciones: {response.text}")
            return []
    except Exception as e:
        print(f"Error en la conexión con Telegram: {e}")
        return []

# ================================
# Sección 7: Bot de Telegram (Bucle)
# ================================
def telegram_bot_loop():
    """Bucle que escucha actualizaciones y procesa cada mensaje."""
    last_update_id = None
    while True:
        try:
            updates = get_updates()
            if updates:
                for update in updates:
                    update_id = update.get("update_id")
                    if last_update_id is None or update_id > last_update_id:
                        handle_telegram_message(update)
                        last_update_id = update_id
            time.sleep(3)
        except Exception as e:
            print(f"Error en el bucle del bot: {e}")
            time.sleep(10)

# ================================
# Sección 8: Monitor de Mercado y Señales de Trading
# ================================
def monitor_market():
    """
    Función principal de monitoreo:
    - Obtiene datos (velas horarias, usando days=7) y calcula indicadores.
    - Genera una señal de trading basada en condiciones técnicas para 1h.
    - Envía la señal si se cumple la condición y no hay cooldown activo (5 minutos).
    """
    global last_trade_signal_time
    print("Obteniendo datos históricos para entrenamiento inicial...")
    data = fetch_data(SYMBOL, TIMEFRAME, days=7)
    train_ml_model(data)  # Se entrena el modelo, aunque la nueva señal usa condiciones técnicas
    print("Datos obtenidos. Comenzando monitoreo en 1h...")
    while True:
        try:
            data = fetch_data(SYMBOL, TIMEFRAME, days=7)
            signal = generate_trade_signal(data)
            now = datetime.now()
            if signal is not None:
                # Enviar señal si ha pasado el cooldown (5 minutos) o si es la primera señal
                if last_trade_signal_time is None or (now - last_trade_signal_time).total_seconds() >= COOLDOWN_SECONDS:
                    send_telegram_message(signal)
                    last_trade_signal_time = now
                else:
                    print("Cooldown activo, señal no enviada.")
            else:
                print("No se cumplen condiciones para una señal de trading.")
            # Revisión cada 5 minutos
            time.sleep(300)
        except Exception as e:
            print(f"Error en monitor_market: {e}")
            time.sleep(300)

# ================================
# Sección 9: Función Start (Punto de Entrada)
# ================================
def start():
    """
    Función de inicio para Railway:
      - Inicia el bucle del bot de Telegram.
      - Inicia el monitor de mercado (con señales de trading en 1h).
    Se ejecutan en hilos separados.
    """
    print("Iniciando Higgs en modo headless (trading 1h)...")
    bot_thread = threading.Thread(target=telegram_bot_loop, daemon=True)
    market_thread = threading.Thread(target=monitor_market, daemon=True)
    bot_thread.start()
    market_thread.start()
    while True:
        time.sleep(60)

# ================================
# Sección 10: Punto de Entrada
# ================================
if __name__ == '__main__':
    start()
