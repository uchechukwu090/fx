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
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import threading
import time
import json
import logging
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import hashlib
import secrets
import uuid

warnings.filterwarnings('ignore')

# --- Integrated Config Object (Replaces external file) ---
class Config:
    def __init__(self):
        self.DATABASE_PATH = 'trading_system.db'
        self.WAVELET_TYPE = 'db4'
        self.PROCESS_NOISE = 0.001
        self.MEASUREMENT_NOISE = 0.01
        self.RISK_TOLERANCE = 0.02
        self.MIN_DATA_POINTS = 50
        self.DEFAULT_HISTORICAL_DAYS = 100
        self.MAX_HISTORICAL_DAYS = 500
        self.LOG_LEVEL = 'INFO'
        self.LOG_FILE = 'app.log'

    def setup_logging(self):
        logging.basicConfig(
            level=self.LOG_LEVEL,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.LOG_FILE),
                logging.StreamHandler()
            ]
        )

config = Config()
# --- End of Integrated Config ---


# --- New Classes for Caching and Rate Limiting ---

class CacheManager:
    """In-memory cache with Time-To-Live (TTL) support."""
    def __init__(self, default_ttl_seconds=60):
        self.cache = {}
        self.default_ttl = timedelta(seconds=default_ttl_seconds)
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.cache:
                value, expiry = self.cache[key]
                if datetime.now() < expiry:
                    return value
                else:
                    # Expired item, remove it
                    del self.cache[key]
            return None

    def set(self, key, value, ttl_seconds=None):
        with self.lock:
            ttl = timedelta(seconds=ttl_seconds) if ttl_seconds else self.default_ttl
            expiry = datetime.now() + ttl
            self.cache[key] = (value, expiry)

    def clear(self):
        with self.lock:
            self.cache.clear()

class RateLimiter:
    """Simple per-minute rate limiter."""
    def __init__(self, max_calls_per_minute=20): # Increased limit
        self.max_calls = max_calls_per_minute
        self.call_log = []
        self.lock = threading.Lock()

    def is_allowed(self):
        with self.lock:
            now = datetime.now()
            one_minute_ago = now - timedelta(minutes=1)
            
            # Remove calls older than one minute
            self.call_log = [t for t in self.call_log if t > one_minute_ago]
            
            if len(self.call_log) < self.max_calls:
                self.call_log.append(now)
                return True
            return False

def to_scalar(x):
    """Convert input to scalar value, handling arrays and edge cases"""
    if x is None:
        return 0.0
    if np.isscalar(x):
        return float(x)
    arr = np.asarray(x)
    if arr.size == 0:
        return 0.0
    if arr.size == 1:
        return float(arr.item())
    if np.all(arr == arr[0]):
        return float(arr[0])
    return float(arr[-1])

# Configure logging using the integrated config
config.setup_logging()
logger = logging.getLogger(__name__)

# --- Data Classes ---

@dataclass
class PredictionLevel:
    price: float
    probability: float
    time_to_target_hours: float
    confidence_interval: Tuple[float, float]

@dataclass
class TimeframePrediction:
    timeframe: str
    direction: str
    magnitude: float
    predictions: List[PredictionLevel]
    reliability_score: float

@dataclass
class KellyPositioning:
    optimal_position_size: float
    expected_return: float
    win_probability: float
    avg_win_loss_ratio: float
    max_position_size: float

@dataclass
class EnhancedTradingSignal:
    signal: str
    confidence: float
    entry_price: float
    timeframe_predictions: Dict[str, TimeframePrediction]
    combined_prediction: TimeframePrediction
    kelly_positioning: KellyPositioning
    dynamic_tp_levels: List[PredictionLevel]
    dynamic_sl_level: PredictionLevel
    timestamp: str = ""
    estimated_entry_price: float = 0.0
    estimated_time_to_target: float = 0.0

@dataclass
class SimulatedOrder:
    order_id: str
    user_id: int
    symbol: str
    order_type: str
    quantity: float
    entry_price: float
    current_price: float
    tp_level: Optional[float] = None
    sl_level: Optional[float] = None
    status: str = 'OPEN'
    pnl: float = 0.0
    pnl_percent: float = 0.0
    created_at: str = ""
    closed_at: Optional[str] = None
    close_reason: Optional[str] = None

@dataclass
class Portfolio:
    user_id: int
    cash_balance: float
    total_equity: float
    open_positions: List[SimulatedOrder]
    total_pnl: float
    total_pnl_percent: float
    margin_used: float
    free_margin: float

# --- Core Logic Classes ---

class PredictiveEngine:
    def __init__(self):
        self.timeframe_weights = {'1min': 0.05, '5min': 0.10, '15min': 0.15, '1hr': 0.25, '4hr': 0.30, 'daily': 0.15}
        self.prediction_horizons = {'1min': [0.5, 1, 2], '5min': [1, 4, 8], '15min': [4, 12, 24], '1hr': [12, 48, 96], '4hr': [24, 96, 240], 'daily': [168, 672, 1680]}

    def predict_timeframe_movement(self, kalman_state: Dict, timeframe: str, current_price: float) -> TimeframePrediction:
        velocity, acceleration = kalman_state.get('velocity', 0), kalman_state.get('acceleration', 0)
        volatility = max(kalman_state.get('volatility', 0.01), 0.001)
        momentum = velocity + acceleration * 0.5
        direction = 'UP' if momentum > 0.001 else 'DOWN' if momentum < -0.001 else 'SIDEWAYS'
        magnitude = min(abs(momentum) / volatility * 100, 50)
        predictions = []
        for i, hours in enumerate(self.prediction_horizons.get(timeframe, [24, 72, 168])):
            time_factor, mean_reversion = np.sqrt(hours / 24), 0.95 ** (hours / 24)
            predicted_change = momentum * time_factor * mean_reversion
            predicted_price = current_price * (1 + predicted_change)
            momentum_strength, time_decay = abs(momentum) / volatility, np.exp(-hours / 168)
            base_probability = 0.5 + 0.3 * np.tanh(momentum_strength) * time_decay
            probability = max(0.1, min(0.9, base_probability * (0.9 - i * 0.15)))
            ci_width = volatility * np.sqrt(hours / 24) * 2
            predictions.append(PredictionLevel(predicted_price, probability, hours, (predicted_price * (1 - ci_width), predicted_price * (1 + ci_width))))
        reliability_score = min(0.95, 0.3 + 0.7 * np.exp(-volatility * 10))
        return TimeframePrediction(timeframe, direction, magnitude, predictions, reliability_score)

    def combine_timeframe_predictions(self, timeframe_predictions: Dict[str, TimeframePrediction], current_price: float) -> TimeframePrediction:
        direction_votes, weighted_magnitude, total_weight = {'UP': 0, 'DOWN': 0, 'SIDEWAYS': 0}, 0, 0
        for tf, pred in timeframe_predictions.items():
            weight = self.timeframe_weights.get(tf, 0.1) * pred.reliability_score
            direction_votes[pred.direction] += weight
            weighted_magnitude += pred.magnitude * weight
            total_weight += weight
        combined_direction = max(direction_votes, key=direction_votes.get)
        combined_magnitude = weighted_magnitude / total_weight if total_weight > 0 else 0
        horizon_groups = {'short': [], 'medium': [], 'long': []}
        for tf, pred in timeframe_predictions.items():
            weight = self.timeframe_weights.get(tf, 0.1) * pred.reliability_score
            for p in pred.predictions:
                if p.time_to_target_hours < 24: horizon_groups['short'].append((p, weight))
                elif p.time_to_target_hours <= 168: horizon_groups['medium'].append((p, weight))
                else: horizon_groups['long'].append((p, weight))
        combined_predictions = []
        for group in horizon_groups.values():
            if group:
                total_w = sum(w for _, w in group)
                avg_price = sum(p.price * w for p, w in group) / total_w
                avg_prob = sum(p.probability * w for p, w in group) / total_w
                avg_time = sum(p.time_to_target_hours * w for p, w in group) / total_w
                ci = (min(p.confidence_interval[0] for p, _ in group), max(p.confidence_interval[1] for p, _ in group))
                combined_predictions.append(PredictionLevel(avg_price, avg_prob, avg_time, ci))
        avg_reliability = np.mean([p.reliability_score for p in timeframe_predictions.values()])
        return TimeframePrediction('combined', combined_direction, combined_magnitude, combined_predictions, avg_reliability)

class KellyCriterionCalculator:
    def __init__(self, max_risk_per_trade: float = 0.02):
        self.max_risk_per_trade = max_risk_per_trade

    def calculate_kelly_position(self, predictions: List[PredictionLevel], entry_price: float, stop_loss: float, risk_tolerance: float) -> KellyPositioning:
        if not predictions or entry_price <= 0 or stop_loss <= 0: return KellyPositioning(0, 0, 0.5, 1, 0)
        risk_per_unit = abs(entry_price - stop_loss) / entry_price
        if risk_per_unit == 0: return KellyPositioning(0, 0, 0.5, 1, 0)
        win_scenarios, loss_scenarios, total_expected_return, total_probability = [], [], 0, 0
        for pred in predictions:
            potential_return, probability = abs(pred.price - entry_price) / entry_price, pred.probability
            if (pred.price > entry_price and entry_price > stop_loss) or (pred.price < entry_price and entry_price < stop_loss):
                win_scenarios.append((potential_return, probability))
                total_expected_return += potential_return * probability
                total_probability += probability
            else: loss_scenarios.append((potential_return, probability))
        win_probability = sum(p for _, p in win_scenarios)
        avg_win = np.mean([r for r, _ in win_scenarios]) if win_scenarios else 0.01
        win_loss_ratio = avg_win / risk_per_unit if risk_per_unit > 0 else 1
        kelly_fraction = (win_loss_ratio * win_probability - (1 - win_probability)) / win_loss_ratio if win_loss_ratio > 0 and win_probability > 0 else 0
        kelly_fraction = max(0, min(kelly_fraction, 0.25))
        max_position_by_risk = risk_tolerance / risk_per_unit
        optimal_position = min(kelly_fraction, max_position_by_risk)
        return KellyPositioning(optimal_position, total_expected_return, win_probability, win_loss_ratio, max_position_by_risk)

class WaveletTransform:
    def __init__(self, wavelet_type=None):
        self.wavelet_type = wavelet_type or config.WAVELET_TYPE
        self.timeframes = ['1min', '5min', '15min', '1hr', '4hr', 'daily']

    def decompose(self, price_data: np.array, levels: int = 6) -> Dict:
        try:
            if len(price_data) < 2**levels: price_data = np.pad(price_data, (0, 2**levels - len(price_data)), mode='edge')
            coeffs = pywt.wavedec(price_data, self.wavelet_type, level=levels)
            components = {}
            for i, timeframe in enumerate(self.timeframes[:len(coeffs)-1]):
                temp_coeffs = [np.zeros_like(c) for c in coeffs]; temp_coeffs[i+1] = coeffs[i+1]
                components[timeframe] = pywt.waverec(temp_coeffs, self.wavelet_type)[:len(price_data)]
            temp_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
            components['trend'] = pywt.waverec(temp_coeffs, self.wavelet_type)[:len(price_data)]
            return components
        except Exception as e:
            logger.error(f"Wavelet decomposition error: {e}"); return {}

class KalmanFilter:
    def __init__(self, process_noise=None, measurement_noise=None):
        self.process_noise = process_noise or config.PROCESS_NOISE
        self.measurement_noise = measurement_noise or config.MEASUREMENT_NOISE
        self.reset()

    def reset(self):
        self.state, self.covariance = np.array([0.0, 0.0, 0.0]), np.eye(3) * 1000
        self.F, self.H = np.array([[1, 1, 0.5], [0, 1, 1], [0, 0, 0.95]]), np.array([[1, 0, 0]])
        self.Q, self.R = np.eye(3) * self.process_noise, np.array([[self.measurement_noise]])

    def update(self, measurement: float) -> Dict:
        predicted_state = self.F @ self.state
        predicted_covariance = self.F @ self.covariance @ self.F.T + self.Q
        residual = measurement - self.H @ predicted_state
        residual_covariance = self.H @ predicted_covariance @ self.H.T + self.R
        if residual_covariance[0, 0] <= 0: residual_covariance[0, 0] = 1e-6
        kalman_gain = predicted_covariance @ self.H.T / residual_covariance[0, 0]
        self.state = predicted_state + kalman_gain * residual
        self.covariance = (np.eye(3) - kalman_gain @ self.H) @ predicted_covariance
        return {'price': to_scalar(self.state[0]), 'velocity': to_scalar(self.state[1]), 'acceleration': to_scalar(self.state[2]), 'volatility': to_scalar(np.sqrt(max(self.covariance[0, 0], 1e-6)))}

    def process_timeframe(self, price_series: np.array) -> Dict:
        self.reset(); results = []
        for price in price_series:
            val = to_scalar(price)
            if not np.isnan(val) and np.isfinite(val): results.append(self.update(val))
        return results[-1] if results else {'price': 0, 'velocity': 0, 'acceleration': 0, 'volatility': 0.01}

class EnhancedBayesianInference:
    def __init__(self, risk_tolerance: float = 0.02):
        self.risk_tolerance = risk_tolerance
        self.predictive_engine = PredictiveEngine()
        self.kelly_calculator = KellyCriterionCalculator(risk_tolerance)

    def estimate_time_to_target(self, predictions: List[PredictionLevel]) -> float:
        if not predictions: return 0.0
        weighted_time = sum(p.time_to_target_hours * p.probability for p in predictions)
        total_probability = sum(p.probability for p in predictions)
        return weighted_time / total_probability if total_probability > 0 else 0.0

    def estimate_entry_price(self, trend_data: np.array, signal: str, current_price: float) -> float:
        if len(trend_data) < 10: return current_price
        adj = current_price * 0.002
        if signal == 'BUY': return max(np.percentile(trend_data[-10:], 25) - adj, current_price * 0.99)
        if signal == 'SELL': return min(np.percentile(trend_data[-10:], 75) + adj, current_price * 1.01)
        return current_price

    def generate_enhanced_signal(self, kalman_states: Dict, current_price: float, risk_tolerance: float, wavelet_components: Dict) -> EnhancedTradingSignal:
        timeframe_predictions = {tf: self.predictive_engine.predict_timeframe_movement(st, tf, current_price) for tf, st in kalman_states.items() if tf != 'trend'}
        combined_prediction = self.predictive_engine.combine_timeframe_predictions(timeframe_predictions, current_price)
        signal = 'BUY' if combined_prediction.direction == 'UP' and combined_prediction.magnitude > 2 else 'SELL' if combined_prediction.direction == 'DOWN' and combined_prediction.magnitude > 2 else 'HOLD'
        confidence = combined_prediction.reliability_score * min(combined_prediction.magnitude * 2, 100)
        dynamic_tp_levels = []
        if signal != 'HOLD':
            sorted_preds = sorted(combined_prediction.predictions, key=lambda x: x.time_to_target_hours)
            for pred in sorted_preds:
                if (signal == 'BUY' and pred.price > current_price) or (signal == 'SELL' and pred.price < current_price):
                    dynamic_tp_levels.append(pred)
        volatility = np.mean([s.get('volatility', 0.01) for s in kalman_states.values()])
        sl_dist = volatility * 2.5
        sl_price = current_price * (1 - sl_dist) if signal == 'BUY' else current_price * (1 + sl_dist) if signal == 'SELL' else current_price
        dynamic_sl_level = PredictionLevel(sl_price, 0.95, 24, (sl_price * 0.99, sl_price * 1.01))
        kelly_positioning = self.kelly_calculator.calculate_kelly_position(dynamic_tp_levels, current_price, sl_price, risk_tolerance)
        
        enhanced_signal = EnhancedTradingSignal(
            signal=signal, confidence=confidence, entry_price=current_price,
            timeframe_predictions=timeframe_predictions, combined_prediction=combined_prediction,
            kelly_positioning=kelly_positioning, dynamic_tp_levels=dynamic_tp_levels,
            dynamic_sl_level=dynamic_sl_level, timestamp=datetime.now().isoformat()
        )
        enhanced_signal.estimated_time_to_target = self.estimate_time_to_target(dynamic_tp_levels)
        enhanced_signal.estimated_entry_price = self.estimate_entry_price(wavelet_components.get("trend", np.array([])), signal, current_price)
        return enhanced_signal

class AdvancedTradingSystem:
    def __init__(self, wavelet_type=None, process_noise=None, measurement_noise=None, risk_tolerance=None):
        self.wavelet = WaveletTransform(wavelet_type)
        self.kalman_filters = {tf: KalmanFilter(process_noise, measurement_noise) for tf in self.wavelet.timeframes}
        self.enhanced_bayesian = EnhancedBayesianInference(risk_tolerance or 0.02)

    def analyze_enhanced(self, price_data: np.array, user_risk_tolerance: float) -> EnhancedTradingSignal:
        if len(price_data) < config.MIN_DATA_POINTS: raise ValueError(f"Need at least {config.MIN_DATA_POINTS} data points")
        wavelet_components = self.wavelet.decompose(price_data)
        if not wavelet_components: raise ValueError("Wavelet decomposition failed")
        kalman_states = {tf: self.kalman_filters[tf].process_timeframe(comp) for tf, comp in wavelet_components.items() if tf in self.kalman_filters}
        current_price = to_scalar(price_data[-1])
        return self.enhanced_bayesian.generate_enhanced_signal(kalman_states, current_price, user_risk_tolerance, wavelet_components)

class TwelveDataProvider:
    def __init__(self, api_key='fef3c30aa26c4831924fdb142f87550d', cache_manager=None, rate_limiter=None):
        self.api_key = api_key or os.environ.get('TWELVE_DATA_API_KEY')
        self.base_url = 'https://api.twelvedata.com'
        self.session = requests.Session()
        self.cache = cache_manager or CacheManager()
        self.rate_limiter = rate_limiter or RateLimiter()
        self.symbol_mapping = {'EURUSD': 'EUR/USD', 'GBPUSD': 'GBP/USD', 'USDJPY': 'USD/JPY', 'XAUUSD': 'XAU/USD', 'BTCUSD': 'BTC/USD', 'ETHUSD': 'ETH/USD'}

    def get_twelve_data_symbol(self, symbol: str) -> str:
        s = symbol.upper().replace('-USD', 'USD').replace('_USD', 'USD')
        return self.symbol_mapping.get(s, f"{s[:3]}/{s[3:]}" if len(s) == 6 and s.isalpha() else s)

    def get_current_price(self, symbol: str) -> Dict:
        cache_key = f"price_{symbol}"
        if cached := self.cache.get(cache_key): return cached
        if not self.rate_limiter.is_allowed(): return {'error': 'Rate limit exceeded'}
        try:
            td_symbol = self.get_twelve_data_symbol(symbol)
            res = self.session.get(f"{self.base_url}/price", params={'symbol': td_symbol, 'apikey': self.api_key}, timeout=10)
            res.raise_for_status()
            price = to_scalar(res.json()['price'])
            result = {'symbol': symbol, 'price': price, 'timestamp': datetime.now().isoformat()}
            self.cache.set(cache_key, result, ttl_seconds=30)
            return result
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return {'symbol': symbol, 'price': 0, 'error': str(e)}

    def get_historical_data(self, symbol: str, days: int = None) -> List[float]:
        days = min(days or config.DEFAULT_HISTORICAL_DAYS, config.MAX_HISTORICAL_DAYS)
        cache_key = f"historical_{symbol}_{days}"
        if cached := self.cache.get(cache_key): return cached
        if not self.rate_limiter.is_allowed(): return []
        try:
            td_symbol = self.get_twelve_data_symbol(symbol)
            res = self.session.get(f"{self.base_url}/time_series", params={'symbol': td_symbol, 'interval': '1day', 'outputsize': str(days), 'apikey': self.api_key}, timeout=15)
            res.raise_for_status()
            data = res.json()
            if 'values' in data and data['values']:
                prices = [to_scalar(item['close']) for item in reversed(data['values'])]
                self.cache.set(cache_key, prices, ttl_seconds=3600)
                return prices
            return []
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return []

class TradingSimulator:
    def __init__(self, db_manager, data_provider):
        self.db_manager = db_manager
        self.data_provider = data_provider
        self.order_close_lock = threading.Lock()

    def calculate_position_value(self, order: SimulatedOrder, current_price: float) -> Dict:
        pnl = (current_price - order.entry_price if order.order_type == 'BUY' else order.entry_price - current_price) * order.quantity
        pnl_percent = (pnl / (order.entry_price * order.quantity)) * 100 if order.entry_price > 0 else 0
        return {'pnl': pnl, 'pnl_percent': pnl_percent}

    def check_tp_sl_conditions(self, order: SimulatedOrder, current_price: float) -> Optional[str]:
        if order.order_type == 'BUY':
            if order.tp_level and current_price >= order.tp_level: return 'TP_HIT'
            if order.sl_level and current_price <= order.sl_level: return 'SL_HIT'
        else: # SELL
            if order.tp_level and current_price <= order.tp_level: return 'TP_HIT'
            if order.sl_level and current_price >= order.sl_level: return 'SL_HIT'
        return None

    def place_order(self, user_id: int, symbol: str, order_type: str, quantity: float, tp_level: Optional[float] = None, sl_level: Optional[float] = None) -> Dict:
        price_data = self.data_provider.get_current_price(symbol)
        if price_data.get('error'): return {'success': False, 'message': price_data['error']}
        current_price = price_data['price']
        if current_price <= 0: return {'success': False, 'message': 'Invalid price data'}
        
        portfolio = self.get_portfolio(user_id)
        required_margin = abs(quantity * current_price)
        if required_margin > portfolio.free_margin: return {'success': False, 'message': 'Insufficient balance'}
        
        order = SimulatedOrder(order_id=str(uuid.uuid4()), user_id=user_id, symbol=symbol, order_type=order_type, quantity=quantity, entry_price=current_price, current_price=current_price, tp_level=tp_level, sl_level=sl_level, created_at=datetime.now().isoformat())
        if self.db_manager.save_simulated_order(order):
            self.db_manager.update_user_balance(user_id, -required_margin)
            logger.info(f"Order placed: {order_type} {quantity} {symbol} for user {user_id}")
            return {'success': True, 'message': 'Order placed successfully', 'order': asdict(order)}
        return {'success': False, 'message': 'Failed to save order'}

    def get_portfolio(self, user_id: int) -> Portfolio:
        cash_balance = self.db_manager.get_user_cash_balance(user_id)
        open_orders = self.db_manager.get_user_orders(user_id, status='OPEN')
        total_pnl, margin_used = 0.0, 0.0
        
        active_positions = []
        for order in open_orders:
            current_price = self.data_provider.get_current_price(order.symbol).get('price', order.entry_price)
            order.current_price = current_price
            
            if close_reason := self.check_tp_sl_conditions(order, current_price):
                self.close_order(order.order_id, current_price, close_reason)
                continue

            pnl_info = self.calculate_position_value(order, current_price)
            order.pnl, order.pnl_percent = pnl_info['pnl'], pnl_info['pnl_percent']
            total_pnl += order.pnl
            margin_used += abs(order.quantity * order.entry_price)
            active_positions.append(order)
            
        total_equity = cash_balance + total_pnl
        free_margin = cash_balance - margin_used
        total_pnl_percent = (total_pnl / cash_balance * 100) if cash_balance > 0 else 0
        return Portfolio(user_id, cash_balance, total_equity, active_positions, total_pnl, total_pnl_percent, margin_used, max(0, free_margin))

    def close_order(self, order_id: str, close_price: float, close_reason: str = 'MANUAL') -> Dict:
        with self.order_close_lock:
            order = self.db_manager.get_order_by_id(order_id)
            if not order or order.status != 'OPEN': return {'success': False, 'message': 'Order not found or already closed'}
            
            pnl_info = self.calculate_position_value(order, close_price)
            final_pnl = pnl_info['pnl']
            
            order.status = 'CLOSED' if close_reason == 'MANUAL' else close_reason
            order.pnl, order.pnl_percent = final_pnl, pnl_info['pnl_percent']
            order.closed_at, order.close_reason = datetime.now().isoformat(), close_reason
            
            self.db_manager.update_order_status(order)
            balance_change = abs(order.quantity * order.entry_price) + final_pnl
            self.db_manager.update_user_balance(order.user_id, balance_change)
            
            logger.info(f"Order {order_id} closed with P&L: {final_pnl}")
            return {'success': True, 'message': 'Position closed', 'pnl': final_pnl}

class DatabaseManager:
    def __init__(self, db_path=None):
        self.db_path = db_path or config.DATABASE_PATH
        self.init_database()
        self.get_or_create_default_user() # Ensure default user exists on startup

    def init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT UNIQUE, email TEXT, password_hash TEXT, cash_balance REAL DEFAULT 10000.0)''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS enhanced_signals (id INTEGER PRIMARY KEY, user_id INTEGER, symbol TEXT, signal TEXT, confidence REAL, entry_price REAL, kelly_position_size REAL, win_probability REAL, prediction_direction TEXT, prediction_magnitude REAL, reliability_score REAL, dynamic_tp_levels TEXT, dynamic_sl_price REAL, timeframe_predictions TEXT, timestamp TEXT, FOREIGN KEY (user_id) REFERENCES users (id))''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS simulated_orders (id INTEGER PRIMARY KEY, order_id TEXT UNIQUE, user_id INTEGER, symbol TEXT, order_type TEXT, quantity REAL, entry_price REAL, current_price REAL, tp_level REAL, sl_level REAL, status TEXT, pnl REAL, pnl_percent REAL, created_at TEXT, closed_at TEXT, close_reason TEXT, FOREIGN KEY (user_id) REFERENCES users (id))''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS balance_history (id INTEGER PRIMARY KEY, user_id INTEGER, balance_change REAL, new_balance REAL, transaction_type TEXT, description TEXT, created_at TEXT, FOREIGN KEY (user_id) REFERENCES users (id))''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_orders_user_status ON simulated_orders(user_id, status)')

    def get_or_create_default_user(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM users WHERE id = 1')
            if not cursor.fetchone():
                cursor.execute('INSERT OR IGNORE INTO users (id, username, email, cash_balance) VALUES (?, ?, ?, ?)', (1, 'default_user', 'default@user.com', 10000.0))
                cursor.execute('INSERT INTO balance_history (user_id, balance_change, new_balance, transaction_type, description, created_at) VALUES (?, ?, ?, ?, ?, ?)', (1, 10000.0, 10000.0, 'INITIAL_DEPOSIT', 'Default user creation', datetime.now().isoformat()))
                logger.info("Created default user with ID 1 and starting balance $10,000")
            return {'id': 1, 'username': 'default_user', 'risk_tolerance': 0.02}

    def save_enhanced_signal(self, symbol: str, signal: EnhancedTradingSignal, user_id: int):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''INSERT INTO enhanced_signals (user_id, symbol, signal, confidence, entry_price, kelly_position_size, win_probability, prediction_direction, prediction_magnitude, reliability_score, dynamic_tp_levels, dynamic_sl_price, timeframe_predictions, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                         (user_id, symbol, signal.signal, to_scalar(signal.confidence), to_scalar(signal.entry_price), to_scalar(signal.kelly_positioning.optimal_position_size), to_scalar(signal.kelly_positioning.win_probability), signal.combined_prediction.direction, to_scalar(signal.combined_prediction.magnitude), to_scalar(signal.combined_prediction.reliability_score), json.dumps([asdict(tp) for tp in signal.dynamic_tp_levels]), to_scalar(signal.dynamic_sl_level.price), json.dumps({tf: asdict(p) for tf, p in signal.timeframe_predictions.items()}), signal.timestamp))

    def save_simulated_order(self, order: SimulatedOrder) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('INSERT INTO simulated_orders (order_id, user_id, symbol, order_type, quantity, entry_price, current_price, tp_level, sl_level, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                             (order.order_id, order.user_id, order.symbol, order.order_type, order.quantity, order.entry_price, order.current_price, order.tp_level, order.sl_level, order.status, order.created_at))
            return True
        except Exception as e:
            logger.error(f"Error saving order: {e}"); return False

    def get_user_orders(self, user_id: int, status: str = None, limit: int = 100) -> List[SimulatedOrder]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            query = 'SELECT order_id, user_id, symbol, order_type, quantity, entry_price, current_price, tp_level, sl_level, status, pnl, pnl_percent, created_at, closed_at, close_reason FROM simulated_orders WHERE user_id = ?'
            params = [user_id]
            if status: query += ' AND status = ?'; params.append(status)
            query += ' ORDER BY created_at DESC LIMIT ?'; params.append(limit)
            cursor.execute(query, params)
            return [SimulatedOrder(*row) for row in cursor.fetchall()]

    def get_order_by_id(self, order_id: str) -> Optional[SimulatedOrder]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute('SELECT order_id, user_id, symbol, order_type, quantity, entry_price, current_price, tp_level, sl_level, status, pnl, pnl_percent, created_at, closed_at, close_reason FROM simulated_orders WHERE order_id = ?', (order_id,)).fetchone()
            return SimulatedOrder(*row) if row else None

    def update_order_status(self, order: SimulatedOrder):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('UPDATE simulated_orders SET status = ?, pnl = ?, pnl_percent = ?, closed_at = ?, close_reason = ? WHERE order_id = ?',
                         (order.status, order.pnl, order.pnl_percent, order.closed_at, order.close_reason, order.order_id))

    def get_user_cash_balance(self, user_id: int) -> float:
        with sqlite3.connect(self.db_path) as conn:
            res = conn.execute('SELECT cash_balance FROM users WHERE id = ?', (user_id,)).fetchone()
            return res[0] if res else 10000.0

    def update_user_balance(self, user_id: int, balance_change: float, transaction_type: str = 'TRADE', description: str = None):
        with sqlite3.connect(self.db_path) as conn:
            current_balance = conn.execute('SELECT cash_balance FROM users WHERE id = ?', (user_id,)).fetchone()[0]
            new_balance = current_balance + balance_change
            conn.execute('UPDATE users SET cash_balance = ? WHERE id = ?', (new_balance, user_id))
            conn.execute('INSERT INTO balance_history (user_id, balance_change, new_balance, transaction_type, description, created_at) VALUES (?, ?, ?, ?, ?, ?)',
                         (user_id, balance_change, new_balance, transaction_type, description or f"{transaction_type} operation", datetime.now().isoformat()))

# --- Flask App Initialization ---
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'a-very-secret-key')
CORS(app)

# --- Initialize Managers ---
cache_manager = CacheManager()
rate_limiter = RateLimiter()
data_provider = TwelveDataProvider(cache_manager=cache_manager, rate_limiter=rate_limiter)
db_manager = DatabaseManager()
trading_simulator = TradingSimulator(db_manager, data_provider)
trading_systems = {}

# --- Mock User Decorator (Replaces login_required) ---
def mock_user_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Attach a mock user object to the request for compatibility
        request.user = {
            'id': 1,
            'risk_tolerance': 0.02,
            'email_notifications': False # Disabled by default
        }
        return f(*args, **kwargs)
    return decorated_function

# --- API Endpoints ---

@app.route('/api/analyze_enhanced', methods=['POST'])
@mock_user_required # No login needed
def analyze_enhanced():
    try:
        data = request.json
        symbol = data.get('symbol', 'BTC/USD').upper()
        risk_tolerance = float(data.get('risk_tolerance', request.user['risk_tolerance']))
        
        system_key = f"enhanced_{risk_tolerance}"
        if system_key not in trading_systems:
            trading_systems[system_key] = AdvancedTradingSystem(risk_tolerance=risk_tolerance)
        
        trading_system = trading_systems[system_key]
        historical_prices = data_provider.get_historical_data(symbol)
        if len(historical_prices) < config.MIN_DATA_POINTS:
            return jsonify({'error': 'Insufficient historical data'}), 400
        
        enhanced_signal = trading_system.analyze_enhanced(np.array(historical_prices), risk_tolerance)
        db_manager.save_enhanced_signal(symbol, enhanced_signal, request.user['id'])
        
        # Simplified response serialization
        response = asdict(enhanced_signal)
        response['current_data'] = data_provider.get_current_price(symbol)
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Enhanced analysis error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/order', methods=['POST'])
@mock_user_required
def place_order_route():
    data = request.json
    result = trading_simulator.place_order(
        user_id=request.user['id'],
        symbol=data.get('symbol', '').upper(),
        order_type=data.get('order_type', '').upper(),
        quantity=float(data.get('quantity', 0)),
        tp_level=float(data['tp_level']) if data.get('tp_level') else None,
        sl_level=float(data['sl_level']) if data.get('sl_level') else None
    )
    return jsonify(result), 201 if result['success'] else 400

@app.route('/api/order/<order_id>/close', methods=['POST'])
@mock_user_required
def close_order_route(order_id):
    order = db_manager.get_order_by_id(order_id)
    if not order or order.user_id != request.user['id']: return jsonify({'error': 'Order not found'}), 404
    current_price = data_provider.get_current_price(order.symbol).get('price', order.entry_price)
    result = trading_simulator.close_order(order_id, current_price)
    return jsonify(result)

@app.route('/api/portfolio', methods=['GET'])
@mock_user_required
def get_portfolio_route():
    portfolio = trading_simulator.get_portfolio(request.user['id'])
    return jsonify(asdict(portfolio))

@app.route('/api/orders', methods=['GET'])
@mock_user_required
def get_orders_route():
    status = request.args.get('status')
    orders = db_manager.get_user_orders(request.user['id'], status=status)
    return jsonify([asdict(o) for o in orders])

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '5.0.0-no-auth',
        'mode': 'single_user'
    })

@app.route("/", methods=["GET"])
def home():
    # A simple landing page is still useful
    return """
    <h1>Trading AI Backend is Running</h1>
    <p>Authentication is disabled. All endpoints operate on a default user.</p>
    <p>Available endpoints:</p>
    <ul>
        <li>POST /api/analyze_enhanced</li>
        <li>POST /api/order</li>
        <li>GET /api/portfolio</li>
        <li>GET /api/orders</li>
    </ul>
    """

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # The background updater is removed as it was tied to multi-user watchlists.
    # For a single-user mode, analysis should be triggered on demand via the API.
    logger.info(f"Starting Flask app in single-user mode on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
