import numpy as np
import pandas as pd
import pywt
from scipy import stats
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import warnings
import requests
from datetime import datetime, timedelta
import sqlite3
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import threading
import time
import json
import logging
import os
from config import config

warnings.filterwarnings('ignore')

# Replace your existing to_scalar function with this:
def to_scalar(x):
    """Convert input to scalar value, handling arrays and edge cases"""
    if x is None:
        return 0.0
    
    # Handle already scalar values
    if np.isscalar(x):
        return float(x)
    
    # Convert to numpy array
    arr = np.asarray(x)
    
    # Handle empty arrays
    if arr.size == 0:
        return 0.0
    
    # Handle single element arrays
    if arr.size == 1:
        return float(arr.item())
    
    # Handle arrays with multiple identical values
    if np.all(arr == arr[0]):
        return float(arr[0])
    
    # For arrays with different values, take the last one (or mean/median)
    # You can change this logic based on your needs
    return float(arr[-1])  # or use np.mean(arr) or np.median(arr)

# Configure logging using config
config.setup_logging()
logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    signal: str
    confidence: float
    entry_price: float
    tp_levels: List[float]
    sl_level: float
    position_size: float
    timeframe_alignment: Dict[str, str]
    timestamp: str = ""

class WaveletTransform:
    def __init__(self, wavelet_type=None):
        self.wavelet_type = wavelet_type or config.WAVELET_TYPE
        self.timeframes = ['1min', '5min', '15min', '1hr', '4hr', 'daily']

    def decompose(self, price_data: np.array, levels: int = 6) -> Dict:
        try:
            if len(price_data) < 2**levels:
                padding_size = 2**levels - len(price_data)
                price_data = np.pad(price_data, (0, padding_size), mode='edge')
            coeffs = pywt.wavedec(price_data, self.wavelet_type, level=levels)
            components = {}
            for i, timeframe in enumerate(self.timeframes[:len(coeffs)-1]):
                temp_coeffs = [np.zeros_like(c) for c in coeffs]
                temp_coeffs[i+1] = coeffs[i+1]
                reconstructed = pywt.waverec(temp_coeffs, self.wavelet_type)
                components[timeframe] = reconstructed[:len(price_data)]
            temp_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
            trend = pywt.waverec(temp_coeffs, self.wavelet_type)
            components['trend'] = trend[:len(price_data)]
            return components
        except Exception as e:
            logger.error(f"Wavelet decomposition error: {e}")
            return {}

class KalmanFilter:
    def __init__(self, process_noise=None, measurement_noise=None):
        self.process_noise = process_noise or config.PROCESS_NOISE
        self.measurement_noise = measurement_noise or config.MEASUREMENT_NOISE
        self.reset()

    def reset(self):
        self.state = np.array([0.0, 0.0, 0.0])
        self.covariance = np.eye(3) * 1000
        self.F = np.array([[1, 1, 0.5], [0, 1, 1], [0, 0, 0.95]])
        self.H = np.array([[1, 0, 0]])
        self.Q = np.eye(3) * self.process_noise
        self.R = np.array([[self.measurement_noise]])

    def update(self, measurement: float) -> Dict:
        predicted_state = self.F @ self.state
        predicted_covariance = self.F @ self.covariance @ self.F.T + self.Q
        residual = measurement - self.H @ predicted_state
        residual_covariance = self.H @ predicted_covariance @ self.H.T + self.R
        if residual_covariance[0, 0] <= 0:
            residual_covariance[0, 0] = 1e-6
        kalman_gain = predicted_covariance @ self.H.T / residual_covariance[0, 0]
        self.state = predicted_state + kalman_gain * residual
        self.covariance = (np.eye(3) - kalman_gain @ self.H) @ predicted_covariance
        return {
            'price': to_scalar(self.state[0]),
            'velocity': to_scalar(self.state[1]),
            'acceleration': to_scalar(self.state[2]),
            'volatility': to_scalar(np.sqrt(max(self.covariance[0, 0], 1e-6)))
        }

    def process_timeframe(self, price_series: np.array) -> Dict:
        self.reset()
        results = []
        for price in price_series:
            val = to_scalar(price)
            if not np.isnan(val) and np.isfinite(val):
                result = self.update(val)
                results.append(result)
        if not results:
            return {'price': 0, 'velocity': 0, 'acceleration': 0, 'volatility': 0.01}
        return results[-1]

class BayesianInference:
    def __init__(self, risk_tolerance=None):
        self.risk_tolerance = risk_tolerance or config.RISK_TOLERANCE
        self.prior_bull = 0.5

    def calculate_signal_probability(self, kalman_states: Dict) -> Dict:
        signals = {}
        for timeframe, state in kalman_states.items():
            if timeframe == 'trend':
                continue
            velocity = state.get('velocity', 0)
            acceleration = state.get('acceleration', 0)
            volatility = max(state.get('volatility', 0.01), 0.01)
            momentum_score = velocity / volatility
            acceleration_score = acceleration / volatility
            combined_score = momentum_score + acceleration_score * 0.5
            likelihood_bull = 0.5 + 0.5 * np.tanh(combined_score)
            likelihood_bear = 1 - likelihood_bull
            posterior_bull = (likelihood_bull * self.prior_bull) / (
                likelihood_bull * self.prior_bull + likelihood_bear * (1 - self.prior_bull))
            signals[timeframe] = {
                'bull_prob': float(posterior_bull),
                'bear_prob': float(1 - posterior_bull),
                'momentum': float(momentum_score),
                'strength': float(abs(momentum_score))
            }
        return signals

    def generate_trading_signal(self, kalman_states: Dict, current_price: float) -> TradingSignal:
        signal_probs = self.calculate_signal_probability(kalman_states)
        weights = {'1min': 0.1, '5min': 0.15, '15min': 0.2, '1hr': 0.25, '4hr': 0.3}
        weighted_bull_prob = 0
        weighted_bear_prob = 0
        timeframe_alignment = {}
        for timeframe, prob_data in signal_probs.items():
            if timeframe in weights:
                weight = weights[timeframe]
                weighted_bull_prob += prob_data['bull_prob'] * weight
                weighted_bear_prob += prob_data['bear_prob'] * weight
                if prob_data['bull_prob'] > 0.6:
                    timeframe_alignment[timeframe] = 'BULLISH'
                elif prob_data['bear_prob'] > 0.6:
                    timeframe_alignment[timeframe] = 'BEARISH'
                else:
                    timeframe_alignment[timeframe] = 'NEUTRAL'
        confidence = max(weighted_bull_prob, weighted_bear_prob) * 100
        if weighted_bull_prob > 0.65:
            signal = 'BUY'
        elif weighted_bear_prob > 0.65:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        volatilities = [state.get('volatility', 0.01) for state in kalman_states.values() if 'volatility' in state]
        avg_volatility = np.mean(volatilities) if volatilities else 0.01
        if signal == 'BUY':
            tp_levels = [
                current_price + avg_volatility * 1.5,
                current_price + avg_volatility * 2.5,
                current_price + avg_volatility * 4.0
            ]
            sl_level = current_price - avg_volatility * 2.0
        elif signal == 'SELL':
            tp_levels = [
                current_price - avg_volatility * 1.5,
                current_price - avg_volatility * 2.5,
                current_price - avg_volatility * 4.0
            ]
            sl_level = current_price + avg_volatility * 2.0
        else:
            tp_levels = [current_price]
            sl_level = current_price
        position_size = min(confidence / 100 * self.risk_tolerance, self.risk_tolerance)
        return TradingSignal(
            signal=signal,
            confidence=to_scalar(confidence),
            entry_price=to_scalar(current_price),
            tp_levels=[to_scalar(x) for x in tp_levels],
            sl_level=to_scalar(sl_level),
            position_size=to_scalar(position_size),
            timeframe_alignment=timeframe_alignment,
            timestamp=datetime.now().isoformat()
        )

class AdvancedTradingSystem:
    def __init__(self, wavelet_type=None, process_noise=None, measurement_noise=None, risk_tolerance=None):
        self.wavelet = WaveletTransform(wavelet_type)
        self.kalman_filters = {tf: KalmanFilter(process_noise, measurement_noise) for tf in self.wavelet.timeframes}
        self.bayesian = BayesianInference(risk_tolerance)

    def analyze(self, price_data: np.array) -> TradingSignal:
        if len(price_data) < config.MIN_DATA_POINTS:
            raise ValueError(f"Need at least {config.MIN_DATA_POINTS} data points for analysis")
        wavelet_components = self.wavelet.decompose(price_data)
        if not wavelet_components:
            raise ValueError("Wavelet decomposition failed")
        kalman_states = {}
        for timeframe in self.wavelet.timeframes:
            if timeframe in wavelet_components:
                component_data = wavelet_components[timeframe]
                kalman_states[timeframe] = self.kalman_filters[timeframe].process_timeframe(component_data)
        current_price = to_scalar(price_data[-1])
        trading_signal = self.bayesian.generate_trading_signal(kalman_states, current_price)
        return trading_signal

class TwelveDataProvider:
    def __init__(self, api_key='fef3c30aa26c4831924fdb142f87550d'):
        self.api_key = api_key
        self.base_url = 'https://api.twelvedata.com'
        self.session = requests.Session()
        self.symbol_mapping = {
            'EURUSD': 'EUR/USD',
            'GBPUSD': 'GBP/USD',
            'USDJPY': 'USD/JPY',
            'USDCHF': 'USD/CHF',
            'AUDUSD': 'AUD/USD',
            'USDCAD': 'USD/CAD',
            'NZDUSD': 'NZD/USD',
            'EURJPY': 'EUR/JPY',
            'GBPJPY': 'GBP/JPY',
            'EURGBP': 'EUR/GBP',
            'XAUUSD': 'XAU/USD',
            'XAGUSD': 'XAG/USD',
            'BTCUSD': 'BTC/USD',
            'ETHUSD': 'ETH/USD',
            'BTC-USD': 'BTC/USD',
            'ETH-USD': 'ETH/USD'
        }

    def get_twelve_data_symbol(self, symbol: str) -> str:
        symbol = symbol.upper().replace('-USD', 'USD').replace('_USD', 'USD')
        if symbol in self.symbol_mapping:
            return self.symbol_mapping[symbol]
        if len(symbol) == 6 and symbol.isalpha():
            return f"{symbol[:3]}/{symbol[3:]}"
        return symbol

    def get_current_price(self, symbol: str) -> Dict:
        try:
            twelve_data_symbol = self.get_twelve_data_symbol(symbol)
            quote_url = f"{self.base_url}/price"
            quote_params = {'symbol': twelve_data_symbol, 'apikey': self.api_key}
            quote_response = self.session.get(quote_url, params=quote_params, timeout=10)
            quote_response.raise_for_status()
            quote_data = quote_response.json()
            if 'price' not in quote_data:
                raise ValueError(f"No price data returned for {symbol}")
            current_price = to_scalar(quote_data['price'])
            quote_detail_url = f"{self.base_url}/quote"
            quote_detail_params = {'symbol': twelve_data_symbol, 'apikey': self.api_key}
            try:
                detail_response = self.session.get(quote_detail_url, params=quote_detail_params, timeout=10)
                detail_response.raise_for_status()
                detail_data = detail_response.json()
                previous_close = to_scalar(detail_data.get('previous_close', current_price))
                high = to_scalar(detail_data.get('high', current_price))
                low = to_scalar(detail_data.get('low', current_price))
                open_price = to_scalar(detail_data.get('open', current_price))
                volume = int(detail_data.get('volume', 0))
            except:
                previous_close = current_price
                high = current_price
                low = current_price
                open_price = current_price
                volume = 0
            change = current_price - previous_close
            change_percent = (change / previous_close * 100) if previous_close != 0 else 0
            return {
                'symbol': symbol,
                'twelve_data_symbol': twelve_data_symbol,
                'price': current_price,
                'change': change,
                'change_percent': change_percent,
                'high': high,
                'low': low,
                'open': open_price,
                'previous_close': previous_close,
                'timestamp': datetime.now().isoformat(),
                'volume': volume
            }
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return {
                'symbol': symbol,
                'price': 0,
                'change': 0,
                'change_percent': 0,
                'timestamp': datetime.now().isoformat(),
                'volume': 0,
                'error': str(e)
            }

    def get_historical_data(self, symbol: str, days: int = None) -> List[float]:
        if days is None:
            days = config.DEFAULT_HISTORICAL_DAYS
        days = min(days, config.MAX_HISTORICAL_DAYS)
        try:
            twelve_data_symbol = self.get_twelve_data_symbol(symbol)
            ts_url = f"{self.base_url}/time_series"
            ts_params = {
                'symbol': twelve_data_symbol,
                'interval': '1day',
                'outputsize': str(days),
                'apikey': self.api_key
            }
            ts_response = self.session.get(ts_url, params=ts_params, timeout=15)
            ts_response.raise_for_status()
            ts_data = ts_response.json()
            if 'values' in ts_data and ts_data['values']:
                prices = [to_scalar(item['close']) for item in reversed(ts_data['values'])]
                return prices
            else:
                current_data = self.get_current_price(symbol)
                base_price = current_data['price'] if current_data['price'] > 0 else 1.0
                prices = []
                current_price = base_price
                for i in range(days):
                    change_percent = np.random.normal(0, 0.02)
                    trend = 0.001 * np.sin(i * 0.1)
                    current_price *= (1 + change_percent + trend)
                    prices.append(to_scalar(current_price))
                return prices
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return [to_scalar(1.0 + i * 0.01 + np.random.normal(0, 0.02)) for i in range(days)]

    def search_symbol(self, query: str) -> List[Dict]:
        try:
            search_url = f"{self.base_url}/symbol_search"
            search_params = {'symbol': query, 'apikey': self.api_key}
            response = self.session.get(search_url, params=search_params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if 'data' in data:
                return data['data']
            else:
                return []
        except Exception as e:
            logger.error(f"Error searching for {query}: {e}")
            return []

class DatabaseManager:
    def __init__(self, db_path=None):
        self.db_path = db_path or config.DATABASE_PATH
        self.init_database()

    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                signal TEXT NOT NULL,
                confidence REAL NOT NULL,
                entry_price REAL NOT NULL,
                tp_levels TEXT NOT NULL,
                sl_level REAL NOT NULL,
                position_size REAL NOT NULL,
                timeframe_alignment TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
                change_val REAL,
                change_percent REAL,
                volume INTEGER,
                timestamp TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()

    def save_signal(self, symbol: str, signal: TradingSignal):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO signals (symbol, signal, confidence, entry_price, tp_levels, 
                               sl_level, position_size, timeframe_alignment, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol,
            signal.signal,
            to_scalar(signal.confidence),
            to_scalar(signal.entry_price),
            json.dumps([to_scalar(x) for x in signal.tp_levels]),
            to_scalar(signal.sl_level),
            to_scalar(signal.position_size),
            json.dumps(signal.timeframe_alignment),
            signal.timestamp
        ))
        conn.commit()
        conn.close()

    def save_price_data(self, price_data: Dict):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO price_history (symbol, price, change_val, change_percent, volume, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            price_data['symbol'],
            to_scalar(price_data['price']),
            to_scalar(price_data['change']),
            to_scalar(price_data['change_percent']),
            int(price_data['volume']),
            price_data['timestamp']
        ))
        conn.commit()
        conn.close()

app = Flask(__name__)
CORS(app)

data_provider = TwelveDataProvider()
db_manager = DatabaseManager()
trading_systems = {}

@app.route('/api/analyze', methods=['POST'])
def analyze_symbol():
    try:
        data = request.json
        symbol = data.get('symbol', 'AAPL')
        timeframe = data.get('timeframe', '1hr')
        risk_tolerance = float(data.get('risk_tolerance', config.RISK_TOLERANCE))
        wavelet_type = data.get('wavelet_type', config.WAVELET_TYPE)
        system_key = f"{wavelet_type}_{risk_tolerance}"
        if system_key not in trading_systems:
            trading_systems[system_key] = AdvancedTradingSystem(
                wavelet_type=wavelet_type,
                risk_tolerance=risk_tolerance
            )
        trading_system = trading_systems[system_key]
        current_data = data_provider.get_current_price(symbol)
        historical_prices = data_provider.get_historical_data(symbol, config.DEFAULT_HISTORICAL_DAYS)
        if len(historical_prices) < config.MIN_DATA_POINTS:
            return jsonify({'error': f'Insufficient historical data. Need at least {config.MIN_DATA_POINTS} points'}), 400
        price_array = np.array([to_scalar(x) for x in historical_prices])
        signal = trading_system.analyze(price_array)
        db_manager.save_signal(symbol, signal)
        db_manager.save_price_data(current_data)
        response = {
            'signal': asdict(signal),
            'current_data': current_data,
            'historical_data': [to_scalar(x) for x in historical_prices][-50:],
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(response)
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/price/<symbol>', methods=['GET'])
def get_price(symbol):
    try:
        price_data = data_provider.get_current_price(symbol)
        db_manager.save_price_data(price_data)
        return jsonify(price_data)
    except Exception as e:
        logger.error(f"Price fetch error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/historical/<symbol>', methods=['GET'])
def get_historical(symbol):
    try:
        days = int(request.args.get('days', config.DEFAULT_HISTORICAL_DAYS))
        days = min(days, config.MAX_HISTORICAL_DAYS)
        historical_data = data_provider.get_historical_data(symbol, days)
        return jsonify({
            'symbol': symbol,
            'data': [to_scalar(x) for x in historical_data],
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Historical data error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/search/<query>', methods=['GET'])
def search_symbols(query):
    try:
        results = data_provider.search_symbol(query)
        return jsonify({
            'query': query,
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/signals/<symbol>', methods=['GET'])
def get_signals_history(symbol):
    try:
        conn = sqlite3.connect(db_manager.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM signals WHERE symbol = ? 
            ORDER BY timestamp DESC LIMIT 10
        ''', (symbol,))
        signals = cursor.fetchall()
        conn.close()
        signal_list = []
        for signal in signals:
            signal_dict = {
                'id': signal[0],
                'symbol': signal[1],
                'signal': signal[2],
                'confidence': to_scalar(signal[3]),
                'entry_price': to_scalar(signal[4]),
                'tp_levels': [to_scalar(x) for x in json.loads(signal[5])],
                'sl_level': to_scalar(signal[6]),
                'position_size': to_scalar(signal[7]),
                'timeframe_alignment': json.loads(signal[8]),
                'timestamp': signal[9]
            }
            signal_list.append(signal_dict)
        return jsonify({'signals': signal_list})
    except Exception as e:
        logger.error(f"Signals history error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0',
        'data_provider': 'Twelve Data'
    })

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

def background_updater():
    symbols = ['EUR/USD', 'XAU/USD', 'USD/JPY']
    while True:
        try:
            for symbol in symbols:
                try:
                    price_data = data_provider.get_current_price(symbol)
                    db_manager.save_price_data(price_data)
                    logger.info(f"Updated price for {symbol}: {price_data['price']}")
                    time.sleep(5)
                except Exception as e:
                    logger.error(f"Error updating {symbol}: {e}")
                    continue
            logger.info("Background update cycle completed, waiting 9 minutes...")
            time.sleep(540)
        except Exception as e:
            logger.error(f"Background update error: {e}")
            time.sleep(60)

if __name__ == '__app__':
    updater_thread = threading.Thread(target=background_updater, daemon=True)
    updater_thread.start()
    logger.info("Background updater started with 9-minute intervals")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
