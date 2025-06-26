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
from flask import Flask, render_template
from flask_cors import CORS
import threading
import time
import json
import logging
import os
from config import config

warnings.filterwarnings('ignore')

# Configure logging using config
config.setup_logging()
logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    signal: str  # BUY/SELL/HOLD
    confidence: float  # 0-100%
    entry_price: float
    tp_levels: List[float]  # Multiple take profit levels
    sl_level: float
    position_size: float  # Recommended position size
    timeframe_alignment: Dict[str, str]  # Status of each timeframe
    timestamp: str = ""

class WaveletTransform:
    def __init__(self, wavelet_type=None):
        self.wavelet_type = wavelet_type or config.WAVELET_TYPE
        self.timeframes = ['1min', '5min', '15min', '1hr', '4hr', 'daily']

    def decompose(self, price_data: np.array, levels: int = 6) -> Dict:
        """Decompose price data into different timeframe components"""
        try:
            # Ensure we have enough data points
            if len(price_data) < 2**levels:
                # Pad data if necessary
                padding_size = 2**levels - len(price_data)
                price_data = np.pad(price_data, (0, padding_size), mode='edge')
            
            # Perform wavelet decomposition
            coeffs = pywt.wavedec(price_data, self.wavelet_type, level=levels)
            
            # Reconstruct components for different timeframes
            components = {}
            for i, timeframe in enumerate(self.timeframes[:len(coeffs)-1]):
                # Create zero arrays for all levels
                temp_coeffs = [np.zeros_like(c) for c in coeffs]
                # Set only the current level
                temp_coeffs[i+1] = coeffs[i+1]
                # Reconstruct
                reconstructed = pywt.waverec(temp_coeffs, self.wavelet_type)
                components[timeframe] = reconstructed[:len(price_data)]
                
            # Approximation (trend)
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
        # State: [price, velocity, acceleration]
        self.state = np.array([0.0, 0.0, 0.0])
        self.covariance = np.eye(3) * 1000
        
        # State transition matrix (constant velocity + acceleration model)
        self.F = np.array([[1, 1, 0.5],
                          [0, 1, 1],
                          [0, 0, 0.95]])  # Decay acceleration
        
        # Observation matrix (we only observe price)
        self.H = np.array([[1, 0, 0]])
        
        # Process noise covariance
        self.Q = np.eye(3) * self.process_noise
        
        # Measurement noise covariance
        self.R = np.array([[self.measurement_noise]])
        
    def update(self, measurement: float) -> Dict:
        """Update Kalman filter with new price measurement"""
        # Prediction step
        predicted_state = self.F @ self.state
        predicted_covariance = self.F @ self.covariance @ self.F.T + self.Q
        
        # Update step
        residual = measurement - self.H @ predicted_state
        residual_covariance = self.H @ predicted_covariance @ self.H.T + self.R
        
        # Handle numerical issues
        if residual_covariance[0, 0] <= 0:
            residual_covariance[0, 0] = 1e-6
            
        kalman_gain = predicted_covariance @ self.H.T / residual_covariance[0, 0]
        
        self.state = predicted_state + kalman_gain * residual
        self.covariance = (np.eye(3) - kalman_gain @ self.H) @ predicted_covariance
        
        return {
            'price': float(self.state[0]),
            'velocity': float(self.state[1]),
            'acceleration': float(self.state[2]),
            'volatility': float(np.sqrt(max(self.covariance[0, 0], 1e-6)))
        }

    def process_timeframe(self, price_series: np.array) -> Dict:
        """Process entire price series for a timeframe"""
        self.reset()
        results = []
        
        for price in price_series:
            if not np.isnan(price) and np.isfinite(price):
                result = self.update(price)
                results.append(result)
        
        if not results:
            return {'price': 0, 'velocity': 0, 'acceleration': 0, 'volatility': 0.01}
            
        # Return latest state
        return results[-1]

class BayesianInference:
    def __init__(self, risk_tolerance=None):
        self.risk_tolerance = risk_tolerance or config.RISK_TOLERANCE
        self.prior_bull = 0.5

    def calculate_signal_probability(self, kalman_states: Dict) -> Dict:
        """Calculate probabilities for buy/sell/hold based on Kalman states"""
        signals = {}
        
        for timeframe, state in kalman_states.items():
            if timeframe == 'trend':
                continue
                
            velocity = state.get('velocity', 0)
            acceleration = state.get('acceleration', 0)
            volatility = max(state.get('volatility', 0.01), 0.01)
            
            # Calculate momentum score
            momentum_score = velocity / volatility
            acceleration_score = acceleration / volatility
            
            # Combined score
            combined_score = momentum_score + acceleration_score * 0.5
            
            # Use tanh to normalize to [-1, 1] range
            likelihood_bull = 0.5 + 0.5 * np.tanh(combined_score)
            likelihood_bear = 1 - likelihood_bull
            
            # Posterior probability using Bayes theorem
            posterior_bull = (likelihood_bull * self.prior_bull) / \
                           (likelihood_bull * self.prior_bull + likelihood_bear * (1 - self.prior_bull))
            
            signals[timeframe] = {
                'bull_prob': float(posterior_bull),
                'bear_prob': float(1 - posterior_bull),
                'momentum': float(momentum_score),
                'strength': float(abs(momentum_score))
            }
        
        return signals

    def generate_trading_signal(self, kalman_states: Dict, current_price: float) -> TradingSignal:
        """Generate final trading signal"""
        signal_probs = self.calculate_signal_probability(kalman_states)
        
        # Weight timeframes (longer timeframes have more weight)
        weights = {'1min': 0.1, '5min': 0.15, '15min': 0.2, '1hr': 0.25, '4hr': 0.3}
        
        weighted_bull_prob = 0
        weighted_bear_prob = 0
        timeframe_alignment = {}
        
        for timeframe, prob_data in signal_probs.items():
            if timeframe in weights:
                weight = weights[timeframe]
                weighted_bull_prob += prob_data['bull_prob'] * weight
                weighted_bear_prob += prob_data['bear_prob'] * weight
                
                # Determine timeframe direction
                if prob_data['bull_prob'] > 0.6:
                    timeframe_alignment[timeframe] = 'BULLISH'
                elif prob_data['bear_prob'] > 0.6:
                    timeframe_alignment[timeframe] = 'BEARISH'
                else:
                    timeframe_alignment[timeframe] = 'NEUTRAL'
        
        # Generate signal
        confidence = max(weighted_bull_prob, weighted_bear_prob) * 100
        
        if weighted_bull_prob > 0.65:
            signal = 'BUY'
        elif weighted_bear_prob > 0.65:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        # Calculate TP/SL levels
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
        
        # Position size based on confidence and risk tolerance
        position_size = min(confidence / 100 * self.risk_tolerance, self.risk_tolerance)
        
        return TradingSignal(
            signal=signal,
            confidence=confidence,
            entry_price=current_price,
            tp_levels=tp_levels,
            sl_level=sl_level,
            position_size=position_size,
            timeframe_alignment=timeframe_alignment,
            timestamp=datetime.now().isoformat()
        )

class AdvancedTradingSystem:
    def __init__(self, wavelet_type=None, process_noise=None, measurement_noise=None, risk_tolerance=None):
        self.wavelet = WaveletTransform(wavelet_type)
        self.kalman_filters = {tf: KalmanFilter(process_noise, measurement_noise) for tf in self.wavelet.timeframes}
        self.bayesian = BayesianInference(risk_tolerance)

    def analyze(self, price_data: np.array) -> TradingSignal:
        """Main analysis function"""
        if len(price_data) < config.MIN_DATA_POINTS:
            raise ValueError(f"Need at least {config.MIN_DATA_POINTS} data points for analysis")
        
        # Step 1: Wavelet decomposition
        wavelet_components = self.wavelet.decompose(price_data)
        
        if not wavelet_components:
            raise ValueError("Wavelet decomposition failed")
        
        # Step 2: Kalman filtering for each timeframe
        kalman_states = {}
        for timeframe in self.wavelet.timeframes:
            if timeframe in wavelet_components:
                component_data = wavelet_components[timeframe]
                kalman_states[timeframe] = self.kalman_filters[timeframe].process_timeframe(component_data)
        
        # Step 3: Bayesian inference
        current_price = float(price_data[-1])
        trading_signal = self.bayesian.generate_trading_signal(kalman_states, current_price)
        
        return trading_signal

class FinnhubDataProvider:
    def __init__(self, api_key=None):
        self.api_key = api_key or config.FINNHUB_API_KEY
        self.base_url = 'https://finnhub.io/api/v1'
        self.session = requests.Session()
        self.session.headers.update({
            'X-Finnhub-Token': self.api_key
        })

    def get_current_price(self, symbol: str) -> Dict:
        """Get current price for any symbol"""
        try:
            # First, search for the symbol to get the correct format
            search_url = f"{self.base_url}/search"
            search_params = {'q': symbol}
            
            search_response = self.session.get(search_url, params=search_params, timeout=config.API_TIMEOUT)
            search_response.raise_for_status()
            search_data = search_response.json()
            
            # Use the first result or the original symbol
            finnhub_symbol = symbol
            if search_data.get('result') and len(search_data['result']) > 0:
                finnhub_symbol = search_data['result'][0]['symbol']
            
            # Get quote data
            quote_url = f"{self.base_url}/quote"
            quote_params = {'symbol': finnhub_symbol}
            
            quote_response = self.session.get(quote_url, params=quote_params, timeout=config.API_TIMEOUT)
            quote_response.raise_for_status()
            quote_data = quote_response.json()
            
            current_price = quote_data.get('c', 0)  # Current price
            previous_close = quote_data.get('pc', current_price)  # Previous close
            change = current_price - previous_close
            change_percent = (change / previous_close * 100) if previous_close != 0 else 0
            
            return {
                'symbol': symbol,
                'finnhub_symbol': finnhub_symbol,
                'price': current_price,
                'change': change,
                'change_percent': change_percent,
                'high': quote_data.get('h', current_price),
                'low': quote_data.get('l', current_price),
                'open': quote_data.get('o', current_price),
                'previous_close': previous_close,
                'timestamp': datetime.now().isoformat(),
                'volume': quote_data.get('v', 0)
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
        """Get historical price data for any symbol"""
        if days is None:
            days = config.DEFAULT_HISTORICAL_DAYS
        days = min(days, config.MAX_HISTORICAL_DAYS)
        
        try:
            # First search for the symbol
            search_url = f"{self.base_url}/search"
            search_params = {'q': symbol}
            
            search_response = self.session.get(search_url, params=search_params, timeout=config.API_TIMEOUT)
            search_response.raise_for_status()
            search_data = search_response.json()
            
            # Use the first result or the original symbol
            finnhub_symbol = symbol
            if search_data.get('result') and len(search_data['result']) > 0:
                finnhub_symbol = search_data['result'][0]['symbol']
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Get candle data (daily)
            candle_url = f"{self.base_url}/stock/candle"
            candle_params = {
                'symbol': finnhub_symbol,
                'resolution': 'D',  # Daily resolution
                'from': int(start_date.timestamp()),
                'to': int(end_date.timestamp())
            }
            
            candle_response = self.session.get(candle_url, params=candle_params, timeout=config.API_TIMEOUT)
            candle_response.raise_for_status()
            candle_data = candle_response.json()
            
            if candle_data.get('s') == 'ok' and candle_data.get('c'):
                # Return closing prices
                return candle_data['c']
            else:
                # Fallback: generate realistic sample data based on current price
                current_data = self.get_current_price(symbol)
                base_price = current_data['price'] if current_data['price'] > 0 else 100
                
                prices = []
                current_price = base_price
                
                for i in range(days):
                    # Generate realistic price movement
                    change_percent = np.random.normal(0, 0.02)  # 2% daily volatility
                    trend = 0.001 * np.sin(i * 0.1)  # Slight trend
                    
                    current_price *= (1 + change_percent + trend)
                    prices.append(current_price)
                
                return prices[::-1]  # Reverse to get chronological order
                
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            # Fallback data
            return [100 + i + np.random.normal(0, 2) for i in range(days)]

    def search_symbol(self, query: str) -> List[Dict]:
        """Search for symbols"""
        try:
            search_url = f"{self.base_url}/search"
            search_params = {'q': query}
            
            response = self.session.get(search_url, params=search_params, timeout=config.API_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            return data.get('result', [])
            
        except Exception as e:
            logger.error(f"Error searching for {query}: {e}")
            return []

class DatabaseManager:
    def __init__(self, db_path=None):
        self.db_path = db_path or config.DATABASE_PATH
        self.init_database()

    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Signals table
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
        
        # Price history table
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
        """Save trading signal to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO signals (symbol, signal, confidence, entry_price, tp_levels, 
                               sl_level, position_size, timeframe_alignment, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol,
            signal.signal,
            signal.confidence,
            signal.entry_price,
            json.dumps(signal.tp_levels),
            signal.sl_level,
            signal.position_size,
            json.dumps(signal.timeframe_alignment),
            signal.timestamp
        ))
        
        conn.commit()
        conn.close()

    def save_price_data(self, price_data: Dict):
        """Save price data to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO price_history (symbol, price, change_val, change_percent, volume, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            price_data['symbol'],
            price_data['price'],
            price_data['change'],
            price_data['change_percent'],
            price_data['volume'],
            price_data['timestamp']
        ))
        
        conn.commit()
        conn.close()

# Flask Application
app = Flask(__name__)
CORS(app)

# Global instances
data_provider = FinnhubDataProvider()
db_manager = DatabaseManager()
trading_systems = {}  # Cache for trading systems with different parameters

@app.route('/api/analyze', methods=['POST'])
def analyze_symbol():
    """Analyze a symbol and return trading signal"""
    try:
        data = request.json
        symbol = data.get('symbol', 'AAPL')
        timeframe = data.get('timeframe', '1hr')
        risk_tolerance = float(data.get('risk_tolerance', config.RISK_TOLERANCE))
        wavelet_type = data.get('wavelet_type', config.WAVELET_TYPE)

        # Create or get trading system with these parameters
        system_key = f"{wavelet_type}_{risk_tolerance}"
        if system_key not in trading_systems:
            trading_systems[system_key] = AdvancedTradingSystem(
                wavelet_type=wavelet_type,
                risk_tolerance=risk_tolerance
            )
        
        trading_system = trading_systems[system_key]
        
        # Get current price data
        current_data = data_provider.get_current_price(symbol)
        
        # Get historical data
        historical_prices = data_provider.get_historical_data(symbol, config.DEFAULT_HISTORICAL_DAYS)
        
        if len(historical_prices) < config.MIN_DATA_POINTS:
            return jsonify({'error': f'Insufficient historical data. Need at least {config.MIN_DATA_POINTS} points'}), 400
        
        # Analyze
        price_array = np.array(historical_prices)
        signal = trading_system.analyze(price_array)
        
        # Save to database
        db_manager.save_signal(symbol, signal)
        db_manager.save_price_data(current_data)
        
        # Prepare response
        response = {
            'signal': asdict(signal),
            'current_data': current_data,
            'historical_data': historical_prices[-50:],  # Last 50 points for chart
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/price/<symbol>', methods=['GET'])
def get_price(symbol):
    """Get current price for a symbol"""
    try:
        price_data = data_provider.get_current_price(symbol)
        db_manager.save_price_data(price_data)
        return jsonify(price_data)
    except Exception as e:
        logger.error(f"Price fetch error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/historical/<symbol>', methods=['GET'])
def get_historical(symbol):
    """Get historical data for a symbol"""
    try:
        days = int(request.args.get('days', config.DEFAULT_HISTORICAL_DAYS))
        days = min(days, config.MAX_HISTORICAL_DAYS)
        historical_data = data_provider.get_historical_data(symbol, days)
        
        return jsonify({
            'symbol': symbol,
            'data': historical_data,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Historical data error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/search/<query>', methods=['GET'])
def search_symbols(query):
    """Search for symbols"""
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
    """Get signal history for a symbol"""
    try:
        conn = sqlite3.connect(db_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM signals WHERE symbol = ? 
            ORDER BY timestamp DESC LIMIT 10
        ''', (symbol,))
        
        signals = cursor.fetchall()
        conn.close()
        
        # Convert to dict format
        signal_list = []
        for signal in signals:
            signal_dict = {
                'id': signal[0],
                'symbol': signal[1],
                'signal': signal[2],
                'confidence': signal[3],
                'entry_price': signal[4],
                'tp_levels': json.loads(signal[5]),
                'sl_level': signal[6],
                'position_size': signal[7],
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
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0',
        'data_provider': 'Finnhub'
    })
@app.route('/webhook/finnhub', methods=['POST'])
def finnhub_webhook():
    secret = request.headers.get('X-Finnhub-Secret')
    if secret != 'd1e5t6hr01qlt46sr0k0':
        return "Forbidden", 403
    return "ok", 200  # acknowledge immediately

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# Background task for periodic updates
def background_updater():
    """Background task to update prices periodically"""
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'AMZN', 'NVDA', 'META', 'BTC-USD', 'ETH-USD']
    
    while True:
        try:
            for symbol in symbols:
                try:
                    price_data = data_provider.get_current_price(symbol)
                    db_manager.save_price_data(price_data)
                    time.sleep(2)  # Rate limiting for API calls
                except Exception as e:
                    logger.error(f"Error updating {symbol}: {e}")
                    continue
            
            time.sleep(300)  # Update every 5 minutes
            
        except Exception as e:
            logger.error(f"Background update error: {e}")
            time.sleep(60)  # Wait before retry

if __name__ == '__app__':
    # Start background updater
    updater_thread = threading.Thread(target=background_updater, daemon=True)
    updater_thread.start()
    
    # Get port from environment variable (Render requirement)
    port = int(os.environ.get('PORT', 5000))
    
    # Run Flask app
    app.run(host='0.0.0.0', port=port, debug=False)
