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
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
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
from config import config
import uuid

warnings.filterwarnings('ignore')

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
    def __init__(self, max_calls_per_minute=10):
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
class PredictionLevel:
    """Individual prediction level with probability and timing"""
    price: float
    probability: float
    time_to_target_hours: float
    confidence_interval: Tuple[float, float]  # (lower, upper)

@dataclass
class TimeframePrediction:
    """Prediction for a specific timeframe"""
    timeframe: str
    direction: str  # 'UP', 'DOWN', 'SIDEWAYS'
    magnitude: float  # Expected price movement in percentage
    predictions: List[PredictionLevel]
    reliability_score: float  # How reliable this timeframe's predictions are

@dataclass
class KellyPositioning:
    """Kelly criterion position sizing results"""
    optimal_position_size: float
    expected_return: float
    win_probability: float
    avg_win_loss_ratio: float
    max_position_size: float  # Risk-adjusted maximum

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

@dataclass
class EnhancedTradingSignal:
    """Enhanced trading signal with predictions and Kelly sizing"""
    signal: str
    confidence: float
    entry_price: float
    
    # Predictive components
    timeframe_predictions: Dict[str, TimeframePrediction]
    combined_prediction: TimeframePrediction
    
    # Kelly-based position sizing
    kelly_positioning: KellyPositioning
    
    # Enhanced TP/SL based on predictions
    dynamic_tp_levels: List[PredictionLevel]
    dynamic_sl_level: PredictionLevel
    
    timestamp: str = ""
    
    # --- NEWLY ADDED FIELDS ---
    estimated_entry_price: float = 0.0
    estimated_time_to_target: float = 0.0

@dataclass
class SimulatedOrder:
    """Represents a simulated trading order"""
    order_id: str
    user_id: int
    symbol: str
    order_type: str  # 'BUY', 'SELL'
    quantity: float
    entry_price: float
    current_price: float
    tp_level: Optional[float] = None
    sl_level: Optional[float] = None
    status: str = 'OPEN'  # 'OPEN', 'CLOSED', 'TP_HIT', 'SL_HIT'
    pnl: float = 0.0
    pnl_percent: float = 0.0
    created_at: str = ""
    closed_at: Optional[str] = None
    close_reason: Optional[str] = None

@dataclass
class Portfolio:
    """User's trading portfolio"""
    user_id: int
    cash_balance: float
    total_equity: float
    open_positions: List[SimulatedOrder]
    total_pnl: float
    total_pnl_percent: float
    margin_used: float
    free_margin: float

class PredictiveEngine:
    """Handles price prediction using multiple timeframe analysis"""
    
    def __init__(self):
        self.timeframe_weights = {
            '1min': 0.05,
            '5min': 0.10,
            '15min': 0.15,
            '1hr': 0.25,
            '4hr': 0.30,
            'daily': 0.15
        }
        
        # Time horizons for each timeframe (in hours)
        self.prediction_horizons = {
            '1min': [0.5, 1, 2],
            '5min': [1, 4, 8],
            '15min': [4, 12, 24],
            '1hr': [12, 48, 96],
            '4hr': [24, 96, 240],
            'daily': [168, 672, 1680]  # 1 week, 1 month, 10 weeks
        }
    
    def predict_timeframe_movement(self, kalman_state: Dict, timeframe: str, current_price: float) -> TimeframePrediction:
        """Predict movement for a specific timeframe"""
        velocity = kalman_state.get('velocity', 0)
        acceleration = kalman_state.get('acceleration', 0)
        volatility = max(kalman_state.get('volatility', 0.01), 0.001)
        
        # Calculate direction and magnitude
        momentum = velocity + acceleration * 0.5
        direction = 'UP' if momentum > 0.001 else 'DOWN' if momentum < -0.001 else 'SIDEWAYS'
        
        # Magnitude based on momentum and volatility
        magnitude = abs(momentum) / volatility * 100
        magnitude = min(magnitude, 50)  # Cap at 50%
        
        # Generate predictions for different time horizons
        predictions = []
        horizons = self.prediction_horizons.get(timeframe, [24, 72, 168])
        
        for i, hours in enumerate(horizons):
            # Price prediction using momentum and mean reversion
            time_factor = np.sqrt(hours / 24)  # Time decay
            mean_reversion = 0.95 ** (hours / 24)  # Mean reversion factor
            
            predicted_change = momentum * time_factor * mean_reversion
            predicted_price = current_price * (1 + predicted_change)
            
            # Probability calculation
            momentum_strength = abs(momentum) / volatility
            time_decay = np.exp(-hours / 168)  # Weekly decay
            base_probability = 0.5 + 0.3 * np.tanh(momentum_strength) * time_decay
            
            # Adjust probability based on prediction level
            probability = base_probability * (0.9 - i * 0.15)  # Decreasing confidence
            probability = max(0.1, min(0.9, probability))
            
            # Confidence interval
            ci_width = volatility * np.sqrt(hours / 24) * 2
            lower_bound = predicted_price * (1 - ci_width)
            upper_bound = predicted_price * (1 + ci_width)
            
            predictions.append(PredictionLevel(
                price=predicted_price,
                probability=probability,
                time_to_target_hours=hours,
                confidence_interval=(lower_bound, upper_bound)
            ))
        
        # Reliability score based on volatility and momentum consistency
        reliability_score = min(0.95, 0.3 + 0.7 * np.exp(-volatility * 10))
        
        return TimeframePrediction(
            timeframe=timeframe,
            direction=direction,
            magnitude=magnitude,
            predictions=predictions,
            reliability_score=reliability_score
        )
    
    def combine_timeframe_predictions(self, timeframe_predictions: Dict[str, TimeframePrediction], 
                                   current_price: float) -> TimeframePrediction:
        """Combine predictions from all timeframes"""
        
        # Weighted direction voting
        direction_votes = {'UP': 0, 'DOWN': 0, 'SIDEWAYS': 0}
        weighted_magnitude = 0
        total_weight = 0
        
        for tf, pred in timeframe_predictions.items():
            weight = self.timeframe_weights.get(tf, 0.1) * pred.reliability_score
            direction_votes[pred.direction] += weight
            weighted_magnitude += pred.magnitude * weight
            total_weight += weight
        
        # Determine combined direction
        combined_direction = max(direction_votes, key=direction_votes.get)
        combined_magnitude = weighted_magnitude / total_weight if total_weight > 0 else 0
        
        # Create combined predictions by averaging similar time horizons
        horizon_groups = {
            'short': [],    # < 24 hours
            'medium': [],   # 24-168 hours
            'long': []      # > 168 hours
        }
        
        for tf, pred in timeframe_predictions.items():
            weight = self.timeframe_weights.get(tf, 0.1) * pred.reliability_score
            for p in pred.predictions:
                if p.time_to_target_hours < 24:
                    horizon_groups['short'].append((p, weight))
                elif p.time_to_target_hours <= 168:
                    horizon_groups['medium'].append((p, weight))
                else:
                    horizon_groups['long'].append((p, weight))
        
        combined_predictions = []
        for group_name, predictions in horizon_groups.items():
            if predictions:
                # Weighted average of predictions in this group
                total_weight = sum(w for _, w in predictions)
                avg_price = sum(p.price * w for p, w in predictions) / total_weight
                avg_probability = sum(p.probability * w for p, w in predictions) / total_weight
                avg_time = sum(p.time_to_target_hours * w for p, w in predictions) / total_weight
                
                # Combined confidence interval
                lower_bounds = [p.confidence_interval[0] for p, _ in predictions]
                upper_bounds = [p.confidence_interval[1] for p, _ in predictions]
                combined_ci = (min(lower_bounds), max(upper_bounds))
                
                combined_predictions.append(PredictionLevel(
                    price=avg_price,
                    probability=avg_probability,
                    time_to_target_hours=avg_time,
                    confidence_interval=combined_ci
                ))
        
        # Overall reliability
        avg_reliability = np.mean([pred.reliability_score for pred in timeframe_predictions.values()])
        
        return TimeframePrediction(
            timeframe='combined',
            direction=combined_direction,
            magnitude=combined_magnitude,
            predictions=combined_predictions,
            reliability_score=avg_reliability
        )

class KellyCriterionCalculator:
    """Calculate optimal position sizing using Kelly Criterion"""
    
    def __init__(self, max_risk_per_trade: float = 0.02):
        self.max_risk_per_trade = max_risk_per_trade  # 2% max risk per trade
    
    def calculate_kelly_position(self, predictions: List[PredictionLevel], 
                               entry_price: float, stop_loss: float, risk_tolerance: float) -> KellyPositioning:
        """
        Calculate Kelly optimal position size, strictly enforcing risk_tolerance.
        """
        
        if not predictions or entry_price <= 0 or stop_loss <= 0:
            return KellyPositioning(0, 0, 0.5, 1, 0)
        
        # Calculate expected returns and probabilities
        total_expected_return = 0
        total_probability = 0
        risk_per_unit = abs(entry_price - stop_loss) / entry_price
        
        if risk_per_unit == 0:
             return KellyPositioning(0, 0, 0.5, 1, 0)

        win_scenarios = []
        loss_scenarios = []
        
        for pred in predictions:
            potential_return = abs(pred.price - entry_price) / entry_price
            probability = pred.probability
            
            if (pred.price > entry_price and entry_price > stop_loss) or \
               (pred.price < entry_price and entry_price < stop_loss):
                # Winning scenario
                win_scenarios.append((potential_return, probability))
                total_expected_return += potential_return * probability
                total_probability += probability
            else:
                # Losing scenario
                loss_scenarios.append((potential_return, probability))
        
        # Calculate win probability and average win/loss ratio
        win_probability = sum(prob for _, prob in win_scenarios)
        loss_probability = 1 - win_probability
        
        if win_scenarios:
            avg_win = np.mean([ret for ret, _ in win_scenarios])
        else:
            avg_win = 0.01
        
        avg_loss = risk_per_unit  # Loss is limited by stop loss
        
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1
        
        # Kelly formula: f = (bp - q) / b
        # where b = win/loss ratio, p = win probability, q = loss probability
        if win_loss_ratio > 0 and win_probability > 0:
            kelly_fraction = (win_loss_ratio * win_probability - loss_probability) / win_loss_ratio
        else:
            kelly_fraction = 0
        
        # Apply Kelly fraction but cap it for risk management
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        
        # Position size based on risk tolerance
        max_position_by_risk = risk_tolerance / risk_per_unit

        # The final position size is the minimum of the Kelly fraction and the risk-tolerance-based size.
        optimal_position = min(kelly_fraction, max_position_by_risk)
        
        return KellyPositioning(
            optimal_position_size=optimal_position,
            expected_return=total_expected_return,
            win_probability=win_probability,
            avg_win_loss_ratio=win_loss_ratio,
            max_position_size=max_position_by_risk
        )

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
        
        # IMPROVED VOLATILITY CALCULATION
        volatilities = [state.get('volatility', 0.01) for state in kalman_states.values() if 'volatility' in state]
        kalman_volatility = np.mean(volatilities) if volatilities else 0.01
        
        # Calculate price-based volatility as backup
        price_based_volatility = current_price * 0.02  # 2% of current price
        
        # Use the larger of the two volatilities, with minimum thresholds
        if current_price > 100:  # For stocks/indices
            min_volatility = current_price * 0.005  # 0.5% minimum
            avg_volatility = max(kalman_volatility, price_based_volatility, min_volatility)
        elif current_price > 1:  # For forex pairs
            min_volatility = 0.001  # 10 pips minimum for forex
            avg_volatility = max(kalman_volatility, price_based_volatility, min_volatility)
        else:  # For crypto with small values
            min_volatility = current_price * 0.01  # 1% minimum
            avg_volatility = max(kalman_volatility, price_based_volatility, min_volatility)
        
        # BETTER TP/SL CALCULATION
        if signal == 'BUY':
            # More aggressive TP levels
            tp_levels = [
                current_price + avg_volatility * 3.0,   # First TP: 3x volatility
                current_price + avg_volatility * 5.0,   # Second TP: 5x volatility  
                current_price + avg_volatility * 8.0    # Third TP: 8x volatility
            ]
            sl_level = current_price - avg_volatility * 2.5  # SL: 2.5x volatility
            
        elif signal == 'SELL':
            tp_levels = [
                current_price - avg_volatility * 3.0,
                current_price - avg_volatility * 5.0,
                current_price - avg_volatility * 8.0
            ]
            sl_level = current_price + avg_volatility * 2.5
            
        else:  # HOLD
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

class EnhancedBayesianInference:
    """Enhanced Bayesian inference with predictive capabilities"""
    
    def __init__(self, risk_tolerance: float = 0.02):
        self.risk_tolerance = risk_tolerance
        self.predictive_engine = PredictiveEngine()
        self.kelly_calculator = KellyCriterionCalculator(risk_tolerance)
        self.prior_bull = 0.5

    # --- NEWLY ADDED UTILITY FUNCTIONS ---
    def estimate_time_to_target(self, predictions: List[PredictionLevel]) -> float:
        """Weighted estimate of time to target in hours."""
        if not predictions:
            return 0.0
        
        weighted_time = sum(p.time_to_target_hours * p.probability for p in predictions)
        total_probability = sum(p.probability for p in predictions)
        
        return weighted_time / total_probability if total_probability > 0 else 0.0

    def estimate_entry_price(self, trend_data: np.array, signal: str, current_price: float) -> float:
        """Estimate optimal entry price based on recent trend movement."""
        if len(trend_data) < 10:
            return current_price

        # Use a small percentage of the price as a buffer for entry
        volatility_adjustment = current_price * 0.002  # ~0.2% entry buffer

        if signal == 'BUY':
            # Target the 25th percentile of the recent trend as a potential dip
            dip_zone = np.percentile(trend_data[-10:], 25)
            # Ensure the entry is not drastically far from the current price
            return max(dip_zone - volatility_adjustment, current_price * 0.99)
        elif signal == 'SELL':
            # Target the 75th percentile of the recent trend as a potential spike
            spike_zone = np.percentile(trend_data[-10:], 75)
            # Ensure the entry is not drastically far from the current price
            return min(spike_zone + volatility_adjustment, current_price * 1.01)
        
        return current_price
    
    # --- MODIFIED METHOD SIGNATURE ---
    def generate_enhanced_signal(self, kalman_states: Dict, current_price: float, user_risk_tolerance: float, wavelet_components: Dict) -> EnhancedTradingSignal:
        """Generate enhanced trading signal with predictions and Kelly sizing"""
        
        # Generate timeframe predictions
        timeframe_predictions = {}
        for timeframe, state in kalman_states.items():
            if timeframe != 'trend':
                pred = self.predictive_engine.predict_timeframe_movement(
                    state, timeframe, current_price
                )
                timeframe_predictions[timeframe] = pred
        
        # Combine predictions
        combined_prediction = self.predictive_engine.combine_timeframe_predictions(
            timeframe_predictions, current_price
        )
        
        # Determine signal direction
        if combined_prediction.direction == 'UP' and combined_prediction.magnitude > 2:
            signal = 'BUY'
        elif combined_prediction.direction == 'DOWN' and combined_prediction.magnitude > 2:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        # Calculate confidence
        confidence = combined_prediction.reliability_score * min(combined_prediction.magnitude * 2, 100)
        
        # Generate dynamic TP levels from predictions
        dynamic_tp_levels = []
        if signal != 'HOLD':
            sorted_predictions = sorted(combined_prediction.predictions, 
                                      key=lambda x: x.time_to_target_hours)
            
            for pred in sorted_predictions:
                if ((signal == 'BUY' and pred.price > current_price) or 
                    (signal == 'SELL' and pred.price < current_price)):
                    dynamic_tp_levels.append(pred)
        
        # Generate dynamic SL based on risk
        volatility = np.mean([state.get('volatility', 0.01) for state in kalman_states.values()])
        sl_distance = volatility * 2.5  # 2.5x volatility for SL
        
        if signal == 'BUY':
            sl_price = current_price * (1 - sl_distance)
        elif signal == 'SELL':
            sl_price = current_price * (1 + sl_distance)
        else:
            sl_price = current_price
        
        dynamic_sl_level = PredictionLevel(
            price=sl_price,
            probability=0.95,  # High probability of hitting SL if wrong
            time_to_target_hours=24,
            confidence_interval=(sl_price * 0.99, sl_price * 1.01)
        )
        
        # Calculate Kelly position sizing, ensuring risk_tolerance is enforced
        kelly_positioning = self.kelly_calculator.calculate_kelly_position(
            dynamic_tp_levels, current_price, sl_price, user_risk_tolerance
        )
        
        # Create the initial signal object
        enhanced_signal = EnhancedTradingSignal(
            signal=signal,
            confidence=confidence,
            entry_price=current_price,
            timeframe_predictions=timeframe_predictions,
            combined_prediction=combined_prediction,
            kelly_positioning=kelly_positioning,
            dynamic_tp_levels=dynamic_tp_levels,
            dynamic_sl_level=dynamic_sl_level,
            timestamp=datetime.now().isoformat()
        )
        
        # --- INJECT NEW ESTIMATIONS ---
        enhanced_signal.estimated_time_to_target = self.estimate_time_to_target(dynamic_tp_levels)
        enhanced_signal.estimated_entry_price = self.estimate_entry_price(
            trend_data=wavelet_components.get("trend", np.array([])),
            signal=signal,
            current_price=current_price
        )
        
        return enhanced_signal

class AdvancedTradingSystem:
    def __init__(self, wavelet_type=None, process_noise=None, measurement_noise=None, risk_tolerance=None):
        self.wavelet = WaveletTransform(wavelet_type)
        self.kalman_filters = {tf: KalmanFilter(process_noise, measurement_noise) for tf in self.wavelet.timeframes}
        self.bayesian = BayesianInference(risk_tolerance)
        self.enhanced_bayesian = EnhancedBayesianInference(risk_tolerance or 0.02)

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

    def analyze_enhanced(self, price_data: np.array, user_risk_tolerance: float) -> EnhancedTradingSignal:
        """Enhanced analysis returning predictive signals"""
        if len(price_data) < 50:  # Minimum data points
            raise ValueError("Need at least 50 data points for enhanced analysis")
        
        # Existing wavelet decomposition
        wavelet_components = self.wavelet.decompose(price_data)
        if not wavelet_components:
            raise ValueError("Wavelet decomposition failed")
        
        # Existing Kalman filtering
        kalman_states = {}
        for timeframe in self.wavelet.timeframes:
            if timeframe in wavelet_components:
                component_data = wavelet_components[timeframe]
                kalman_states[timeframe] = self.kalman_filters[timeframe].process_timeframe(component_data)
        
        current_price = to_scalar(price_data[-1])
        
        # --- MODIFIED METHOD CALL ---
        # Generate enhanced signal, passing wavelet_components
        enhanced_signal = self.enhanced_bayesian.generate_enhanced_signal(
            kalman_states, 
            current_price, 
            user_risk_tolerance,
            wavelet_components
        )
        
        return enhanced_signal

class TwelveDataProvider:
    def __init__(self, api_key='fef3c30aa26c4831924fdb142f87550d', cache_manager=None, rate_limiter=None):
        self.api_key = api_key
        self.base_url = 'https://api.twelvedata.com'
        self.session = requests.Session()
        self.cache = cache_manager or CacheManager()
        self.rate_limiter = rate_limiter or RateLimiter()
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
        cache_key = f"price_{symbol}"
        cached_data = self.cache.get(cache_key)
        if cached_data:
            return cached_data

        if not self.rate_limiter.is_allowed():
            logger.warning(f"Rate limit exceeded for price data. Symbol: {symbol}")
            return {'error': 'Rate limit exceeded. Please try again later.'}

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
            
            result = {
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
            self.cache.set(cache_key, result, ttl_seconds=30) # Cache for 30 seconds
            return result
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

        cache_key = f"historical_{symbol}_{days}"
        cached_data = self.cache.get(cache_key)
        if cached_data:
            return cached_data

        if not self.rate_limiter.is_allowed():
            logger.warning(f"Rate limit exceeded for historical data. Symbol: {symbol}")
            return []

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
                self.cache.set(cache_key, prices, ttl_seconds=3600) # Cache for 1 hour
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

class TradingSimulator:
    """Handles trading simulation logic including position management and P&L calculation"""
    
    def __init__(self, db_manager, data_provider):
        self.db_manager = db_manager
        self.data_provider = data_provider
        self.order_close_lock = threading.Lock()
    
    def calculate_position_value(self, order: SimulatedOrder, current_price: float) -> Dict:
        """Calculate current value and P&L for a position"""
        if order.order_type == 'BUY':
            pnl = (current_price - order.entry_price) * order.quantity
            pnl_percent = ((current_price - order.entry_price) / order.entry_price) * 100
        else:  # SELL
            pnl = (order.entry_price - current_price) * order.quantity
            pnl_percent = ((order.entry_price - current_price) / order.entry_price) * 100
        
        return {
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'current_value': abs(current_price * order.quantity),
            'unrealized_pnl': pnl
        }
    
    def check_tp_sl_conditions(self, order: SimulatedOrder, current_price: float) -> Optional[str]:
        """Check if TP or SL conditions are met"""
        if order.order_type == 'BUY':
            if order.tp_level and current_price >= order.tp_level:
                return 'TP_HIT'
            elif order.sl_level and current_price <= order.sl_level:
                return 'SL_HIT'
        else:  # SELL
            if order.tp_level and current_price <= order.tp_level:
                return 'TP_HIT'
            elif order.sl_level and current_price >= order.sl_level:
                return 'SL_HIT'
        
        return None
    
    def place_order(self, user_id: int, symbol: str, order_type: str, quantity: float, 
                   tp_level: Optional[float] = None, sl_level: Optional[float] = None) -> Dict:
        """Place a simulated trading order"""
        try:
            # Get current price
            price_data = self.data_provider.get_current_price(symbol)
            if price_data.get('error'):
                return {'success': False, 'message': f"Failed to get price for {symbol}: {price_data.get('error')}"}
            
            current_price = price_data['price']
            if current_price <= 0:
                return {'success': False, 'message': 'Invalid price data'}
            
            # Get user's current balance
            portfolio = self.get_portfolio(user_id)
            
            # Calculate required margin (for now, 1:1 leverage)
            required_margin = abs(quantity * current_price)
            
            # Check if user has sufficient balance
            if required_margin > portfolio.free_margin:
                return {
                    'success': False, 
                    'message': f'Insufficient balance. Required: ${required_margin:.2f}, Available: ${portfolio.free_margin:.2f}'
                }
            
            # Validate TP/SL levels
            if order_type == 'BUY':
                if tp_level and tp_level <= current_price:
                    return {'success': False, 'message': 'Take Profit must be above current price for BUY orders'}
                if sl_level and sl_level >= current_price:
                    return {'success': False, 'message': 'Stop Loss must be below current price for BUY orders'}
            else:  # SELL
                if tp_level and tp_level >= current_price:
                    return {'success': False, 'message': 'Take Profit must be below current price for SELL orders'}
                if sl_level and sl_level <= current_price:
                    return {'success': False, 'message': 'Stop Loss must be above current price for SELL orders'}
            
            # Create order
            order_id = str(uuid.uuid4())
            order = SimulatedOrder(
                order_id=order_id,
                user_id=user_id,
                symbol=symbol,
                order_type=order_type,
                quantity=quantity,
                entry_price=current_price,
                current_price=current_price,
                tp_level=tp_level,
                sl_level=sl_level,
                status='OPEN',
                created_at=datetime.now().isoformat()
            )
            
            # Save order to database
            success = self.db_manager.save_simulated_order(order)
            if not success:
                return {'success': False, 'message': 'Failed to save order'}
            
            # Update user's cash balance
            self.db_manager.update_user_balance(user_id, -required_margin)
            
            # Create notification
            self.db_manager.add_notification(
                user_id,
                f"Order Placed: {order_type} {symbol}",
                f"Successfully placed {order_type} order for {quantity} units of {symbol} at ${current_price:.4f}",
                'order'
            )
            
            logger.info(f"Order placed: {order_type} {quantity} {symbol} at {current_price} for user {user_id}")
            
            return {
                'success': True,
                'order_id': order_id,
                'message': f'{order_type} order placed successfully',
                'order': {
                    'order_id': order_id,
                    'symbol': symbol,
                    'order_type': order_type,
                    'quantity': quantity,
                    'entry_price': current_price,
                    'tp_level': tp_level,
                    'sl_level': sl_level,
                    'status': 'OPEN',
                    'created_at': order.created_at
                }
            }
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {'success': False, 'message': 'Failed to place order'}
    
    def get_portfolio(self, user_id: int) -> Portfolio:
        """Get user's complete portfolio"""
        try:
            # Get user's cash balance
            cash_balance = self.db_manager.get_user_cash_balance(user_id)
            
            # Get open positions
            open_orders = self.db_manager.get_user_orders(user_id, status='OPEN')
            
            total_equity = cash_balance
            total_pnl = 0.0
            margin_used = 0.0
            
            portfolio_positions = []
            
            for order in open_orders:
                # Get current price for this symbol
                current_data = self.data_provider.get_current_price(order.symbol)
                current_price = current_data.get('price', order.entry_price)
                
                # Update order's current price
                order.current_price = current_price
                
                # Calculate P&L
                position_value = self.calculate_position_value(order, current_price)
                order.pnl = position_value['pnl']
                order.pnl_percent = position_value['pnl_percent']
                
                total_pnl += position_value['pnl']
                margin_used += abs(order.quantity * order.entry_price)
                
                # Check TP/SL conditions
                close_reason = self.check_tp_sl_conditions(order, current_price)
                if close_reason:
                    # Auto-close the position
                    self.close_order(order.order_id, current_price, close_reason)
                    continue
                
                portfolio_positions.append(order)
            
            total_equity = cash_balance + total_pnl
            free_margin = cash_balance - margin_used
            total_pnl_percent = (total_pnl / cash_balance * 100) if cash_balance > 0 else 0
            
            return Portfolio(
                user_id=user_id,
                cash_balance=cash_balance,
                total_equity=total_equity,
                open_positions=portfolio_positions,
                total_pnl=total_pnl,
                total_pnl_percent=total_pnl_percent,
                margin_used=margin_used,
                free_margin=max(0, free_margin)
            )
            
        except Exception as e:
            logger.error(f"Error getting portfolio for user {user_id}: {e}")
            return Portfolio(
                user_id=user_id,
                cash_balance=10000.0,  # Default starting balance
                total_equity=10000.0,
                open_positions=[],
                total_pnl=0.0,
                total_pnl_percent=0.0,
                margin_used=0.0,
                free_margin=10000.0
            )
    
    def close_order(self, order_id: str, close_price: float, close_reason: str = 'MANUAL') -> Dict:
        """Close an open order with transactional logic."""
        with self.order_close_lock:
            try:
                # Re-check order status within the transaction
                order = self.db_manager.get_order_by_id(order_id)
                if not order or order.status != 'OPEN':
                    return {'success': False, 'message': 'Order not found or already closed'}

                # Calculate final P&L
                position_value = self.calculate_position_value(order, close_price)
                final_pnl = position_value['pnl']

                # Update order status
                order.status = 'CLOSED' if close_reason == 'MANUAL' else close_reason
                order.current_price = close_price
                order.pnl = final_pnl
                order.pnl_percent = position_value['pnl_percent']
                order.closed_at = datetime.now().isoformat()
                order.close_reason = close_reason

                # Update in database
                self.db_manager.update_order_status(order)

                # Return margin and add/subtract P&L to user's balance
                margin_return = abs(order.quantity * order.entry_price)
                balance_change = margin_return + final_pnl
                self.db_manager.update_user_balance(order.user_id, balance_change)

                # Create notification
                pnl_text = f"${final_pnl:.2f}" if final_pnl >= 0 else f"-${abs(final_pnl):.2f}"
                self.db_manager.add_notification(
                    order.user_id,
                    f"Position Closed: {order.symbol}",
                    f"Closed {order.order_type} position for {order.symbol}. P&L: {pnl_text} ({position_value['pnl_percent']:.2f}%)",
                    'position_closed'
                )

                logger.info(f"Order {order_id} closed with P&L: {final_pnl}")

                return {
                    'success': True,
                    'message': 'Position closed successfully',
                    'pnl': final_pnl,
                    'pnl_percent': position_value['pnl_percent'],
                    'close_reason': close_reason
                }

            except Exception as e:
                logger.error(f"Error closing order {order_id}: {e}")
                # Potentially roll back transaction here if using a more advanced DB
                return {'success': False, 'message': 'Failed to close position'}

class NotificationManager:
    """Manages email and in-app notifications for users"""
    
    def __init__(self, smtp_server='smtp.gmail.com', smtp_port=587):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.email_username = os.environ.get('EMAIL_USERNAME', '')
        self.email_password = os.environ.get('EMAIL_PASSWORD', '')
    
    def send_email_notification(self, to_email: str, subject: str, message: str) -> bool:
        """Send email notification to user"""
        try:
            if not self.email_username or not self.email_password:
                logger.warning("Email credentials not configured")
                return False
            
            msg = MIMEMultipart()
            msg['From'] = self.email_username
            msg['To'] = to_email
            msg['Subject'] = subject
            
            msg.attach(MIMEText(message, 'html'))
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_username, self.email_password)
            text = msg.as_string()
            server.sendmail(self.email_username, to_email, text)
            server.quit()
            
            logger.info(f"Email notification sent to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False
    
    def create_signal_email(self, signal: EnhancedTradingSignal, symbol: str) -> str:
        """Create HTML email for trading signal"""
        direction_color = "#28a745" if signal.signal == "BUY" else "#dc3545" if signal.signal == "SELL" else "#ffc107"
        
        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f8f9fa;">
            <div style="max-width: 600px; margin: 0 auto; background-color: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px 8px 0 0;">
                    <h1 style="margin: 0; font-size: 24px;">🚀 New Trading Signal</h1>
                    <p style="margin: 5px 0 0 0; opacity: 0.9;">Advanced AI Trading System</p>
                </div>
                
                <div style="padding: 30px;">
                    <div style="text-align: center; margin-bottom: 30px;">
                        <h2 style="color: #333; margin: 0 0 10px 0;">{symbol}</h2>
                        <div style="display: inline-block; padding: 10px 20px; background-color: {direction_color}; color: white; border-radius: 25px; font-weight: bold; font-size: 18px;">
                            {signal.signal}
                        </div>
                    </div>
                    
                    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                        <h3 style="color: #333; margin: 0 0 15px 0;">📊 Signal Details</h3>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                            <div>
                                <strong>Entry Price:</strong><br>
                                <span style="font-size: 18px; color: #667eea;">${signal.entry_price:.4f}</span>
                            </div>
                            <div>
                                <strong>Confidence:</strong><br>
                                <span style="font-size: 18px; color: #28a745;">{signal.confidence:.1f}%</span>
                            </div>
                        </div>
                    </div>
                    
                    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                        <h3 style="color: #333; margin: 0 0 15px 0;">🎯 Kelly Position Sizing</h3>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                            <div>
                                <strong>Optimal Size:</strong><br>
                                <span style="font-size: 16px; color: #667eea;">{signal.kelly_positioning.optimal_position_size:.2%}</span>
                            </div>
                            <div>
                                <strong>Win Probability:</strong><br>
                                <span style="font-size: 16px; color: #28a745;">{signal.kelly_positioning.win_probability:.1%}</span>
                            </div>
                        </div>
                    </div>
                    
                    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                        <h3 style="color: #333; margin: 0 0 15px 0;">📈 Prediction</h3>
                        <p><strong>Direction:</strong> {signal.combined_prediction.direction}</p>
                        <p><strong>Magnitude:</strong> {signal.combined_prediction.magnitude:.1f}%</p>
                        <p><strong>Reliability:</strong> {signal.combined_prediction.reliability_score:.1%}</p>
                    </div>
                    
                    <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                        <h4 style="color: #856404; margin: 0 0 10px 0;">⚠️ Risk Management</h4>
                        <p style="margin: 0; color: #856404; font-size: 14px;">
                            Stop Loss: ${signal.dynamic_sl_level.price:.4f}<br>
                            Always use proper position sizing and risk management.
                        </p>
                    </div>
                    
                    <div style="text-align: center; margin-top: 30px;">
                        <p style="color: #666; font-size: 12px; margin: 0;">
                            Generated at {signal.timestamp}<br>
                            This is an automated signal from AI Trading System
                        </p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        return html

class DatabaseManager:
    def __init__(self, db_path=None):
        self.db_path = db_path or config.DATABASE_PATH
        self.init_database()

    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Original tables
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
        
        # User authentication tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                first_name TEXT,
                last_name TEXT,
                email_notifications BOOLEAN DEFAULT 1,
                push_notifications BOOLEAN DEFAULT 1,
                risk_tolerance REAL DEFAULT 0.02,
                cash_balance REAL DEFAULT 10000.0,
                created_at TEXT NOT NULL,
                last_login TEXT,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_token TEXT UNIQUE NOT NULL,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(user_id, symbol),
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                notification_type TEXT NOT NULL,
                is_read BOOLEAN DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS enhanced_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                symbol TEXT NOT NULL,
                signal TEXT NOT NULL,
                confidence REAL NOT NULL,
                entry_price REAL NOT NULL,
                kelly_position_size REAL NOT NULL,
                win_probability REAL NOT NULL,
                prediction_direction TEXT NOT NULL,
                prediction_magnitude REAL NOT NULL,
                reliability_score REAL NOT NULL,
                dynamic_tp_levels TEXT NOT NULL,
                dynamic_sl_price REAL NOT NULL,
                timeframe_predictions TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # NEW TRADING SIMULATION TABLES
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS simulated_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT UNIQUE NOT NULL,
                user_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                order_type TEXT NOT NULL,
                quantity REAL NOT NULL,
                entry_price REAL NOT NULL,
                current_price REAL NOT NULL,
                tp_level REAL,
                sl_level REAL,
                status TEXT NOT NULL DEFAULT 'OPEN',
                pnl REAL DEFAULT 0.0,
                pnl_percent REAL DEFAULT 0.0,
                created_at TEXT NOT NULL,
                closed_at TEXT,
                close_reason TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS balance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                balance_change REAL NOT NULL,
                new_balance REAL NOT NULL,
                transaction_type TEXT NOT NULL,
                description TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Add indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_orders_user_id ON simulated_orders(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_orders_status ON simulated_orders(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_orders_symbol ON simulated_orders(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_balance_user_id ON balance_history(user_id)')
        
        # Check if cash_balance column exists in users table, add if not
        cursor.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in cursor.fetchall()]
        if 'cash_balance' not in columns:
            cursor.execute('ALTER TABLE users ADD COLUMN cash_balance REAL DEFAULT 10000.0')
        
        conn.commit()
        conn.close()

    def create_user(self, username: str, email: str, password: str, first_name: str = "", last_name: str = "") -> Dict:
        """Create a new user account"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if user already exists
            cursor.execute('SELECT id FROM users WHERE username = ? OR email = ?', (username, email))
            if cursor.fetchone():
                return {'success': False, 'message': 'Username or email already exists'}
            
            password_hash = generate_password_hash(password)
            created_at = datetime.now().isoformat()
            
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, first_name, last_name, cash_balance, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (username, email, password_hash, first_name, last_name, 10000.0, created_at))
            
            user_id = cursor.lastrowid
            
            # Log initial balance
            cursor.execute('''
                INSERT INTO balance_history (user_id, balance_change, new_balance, transaction_type, description, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, 10000.0, 10000.0, 'INITIAL_DEPOSIT', 'Account creation - starting balance', created_at))
            
            conn.commit()
            conn.close()
            
            logger.info(f"New user created: {username} ({email}) with starting balance $10,000")
            return {'success': True, 'user_id': user_id, 'message': 'Account created successfully'}
            
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return {'success': False, 'message': 'Failed to create account'}

    def authenticate_user(self, username: str, password: str) -> Dict:
        """Authenticate user login"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, username, email, password_hash, first_name, last_name, 
                       email_notifications, push_notifications, risk_tolerance, cash_balance, is_active
                FROM users WHERE username = ? OR email = ?
            ''', (username, username))
            
            user = cursor.fetchone()
            if not user:
                return {'success': False, 'message': 'Invalid username or password'}
            
            if not user[10]:  # is_active
                return {'success': False, 'message': 'Account is deactivated'}
            
            if check_password_hash(user[3], password):
                # Update last login
                cursor.execute('UPDATE users SET last_login = ? WHERE id = ?', 
                             (datetime.now().isoformat(), user[0]))
                conn.commit()
                conn.close()
                
                return {
                    'success': True,
                    'user': {
                        'id': user[0],
                        'username': user[1],
                        'email': user[2],
                        'first_name': user[4],
                        'last_name': user[5],
                        'email_notifications': bool(user[6]),
                        'push_notifications': bool(user[7]),
                        'risk_tolerance': user[8],
                        'cash_balance': user[9]
                    }
                }
            else:
                conn.close()
                return {'success': False, 'message': 'Invalid username or password'}
                
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            return {'success': False, 'message': 'Authentication failed'}

    def create_session(self, user_id: int) -> str:
        """Create a new session token for user"""
        try:
            session_token = secrets.token_urlsafe(32)
            expires_at = (datetime.now() + timedelta(days=30)).isoformat()
            created_at = datetime.now().isoformat()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO user_sessions (user_id, session_token, expires_at, created_at)
                VALUES (?, ?, ?, ?)
            ''', (user_id, session_token, expires_at, created_at))
            
            conn.commit()
            conn.close()
            
            return session_token
            
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return ""

    def get_user_by_session(self, session_token: str) -> Optional[Dict]:
        """Get user by session token"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT u.id, u.username, u.email, u.first_name, u.last_name,
                       u.email_notifications, u.push_notifications, u.risk_tolerance,
                       u.cash_balance, s.expires_at
                FROM users u
                JOIN user_sessions s ON u.id = s.user_id
                WHERE s.session_token = ? AND u.is_active = 1
            ''', (session_token,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                expires_at = datetime.fromisoformat(result[9])
                if expires_at > datetime.now():
                    return {
                        'id': result[0],
                        'username': result[1],
                        'email': result[2],
                        'first_name': result[3],
                        'last_name': result[4],
                        'email_notifications': bool(result[5]),
                        'push_notifications': bool(result[6]),
                        'risk_tolerance': result[7],
                        'cash_balance': result[8]
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting user by session: {e}")
            return None

    def add_to_watchlist(self, user_id: int, symbol: str) -> bool:
        """Add symbol to user's watchlist"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR IGNORE INTO user_watchlist (user_id, symbol, created_at)
                VALUES (?, ?, ?)
            ''', (user_id, symbol.upper(), datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error adding to watchlist: {e}")
            return False

    def get_user_watchlist(self, user_id: int) -> List[str]:
        """Get user's watchlist"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT symbol FROM user_watchlist WHERE user_id = ? ORDER BY created_at DESC', (user_id,))
            symbols = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            return symbols
            
        except Exception as e:
            logger.error(f"Error getting watchlist: {e}")
            return []

    def save_enhanced_signal(self, symbol: str, signal: EnhancedTradingSignal, user_id: int = None):
        """Save enhanced trading signal"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO enhanced_signals (
                    user_id, symbol, signal, confidence, entry_price,
                    kelly_position_size, win_probability, prediction_direction,
                    prediction_magnitude, reliability_score, dynamic_tp_levels,
                    dynamic_sl_price, timeframe_predictions, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                symbol,
                signal.signal,
                to_scalar(signal.confidence),
                to_scalar(signal.entry_price),
                to_scalar(signal.kelly_positioning.optimal_position_size),
                to_scalar(signal.kelly_positioning.win_probability),
                signal.combined_prediction.direction,
                to_scalar(signal.combined_prediction.magnitude),
                to_scalar(signal.combined_prediction.reliability_score),
                json.dumps([{
                    'price': to_scalar(tp.price),
                    'probability': to_scalar(tp.probability),
                    'time_to_target_hours': to_scalar(tp.time_to_target_hours)
                } for tp in signal.dynamic_tp_levels]),
                to_scalar(signal.dynamic_sl_level.price),
                json.dumps({tf: {
                    'direction': pred.direction,
                    'magnitude': to_scalar(pred.magnitude),
                    'reliability_score': to_scalar(pred.reliability_score)
                } for tf, pred in signal.timeframe_predictions.items()}),
                signal.timestamp
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving enhanced signal: {e}")

    def add_notification(self, user_id: int, title: str, message: str, notification_type: str = 'signal'):
        """Add notification for user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO user_notifications (user_id, title, message, notification_type, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, title, message, notification_type, datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error adding notification: {e}")

    def get_user_notifications(self, user_id: int, limit: int = 20) -> List[Dict]:
        """Get user's notifications"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, title, message, notification_type, is_read, created_at
                FROM user_notifications
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            ''', (user_id, limit))
            
            notifications = []
            for row in cursor.fetchall():
                notifications.append({
                    'id': row[0],
                    'title': row[1],
                    'message': row[2],
                    'type': row[3],
                    'is_read': bool(row[4]),
                    'created_at': row[5]
                })
            
            conn.close()
            return notifications
            
        except Exception as e:
            logger.error(f"Error getting notifications: {e}")
            return []

    # NEW TRADING SIMULATION METHODS
    
    def save_simulated_order(self, order: SimulatedOrder) -> bool:
        """Save a simulated trading order"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO simulated_orders (
                    order_id, user_id, symbol, order_type, quantity, entry_price,
                    current_price, tp_level, sl_level, status, pnl, pnl_percent, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                order.order_id, order.user_id, order.symbol, order.order_type,
                order.quantity, order.entry_price, order.current_price,
                order.tp_level, order.sl_level, order.status,
                order.pnl, order.pnl_percent, order.created_at
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error saving simulated order: {e}")
            return False
    
    def get_user_orders(self, user_id: int, status: str = None, limit: int = 100) -> List[SimulatedOrder]:
        """Get user's trading orders"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if status:
                cursor.execute('''
                    SELECT * FROM simulated_orders 
                    WHERE user_id = ? AND status = ?
                    ORDER BY created_at DESC LIMIT ?
                ''', (user_id, status, limit))
            else:
                cursor.execute('''
                    SELECT * FROM simulated_orders 
                    WHERE user_id = ?
                    ORDER BY created_at DESC LIMIT ?
                ''', (user_id, limit))
            
            orders = []
            for row in cursor.fetchall():
                order = SimulatedOrder(
                    order_id=row[1],
                    user_id=row[2],
                    symbol=row[3],
                    order_type=row[4],
                    quantity=row[5],
                    entry_price=row[6],
                    current_price=row[7],
                    tp_level=row[8],
                    sl_level=row[9],
                    status=row[10],
                    pnl=row[11],
                    pnl_percent=row[12],
                    created_at=row[13],
                    closed_at=row[14],
                    close_reason=row[15]
                )
                orders.append(order)
            
            conn.close()
            return orders
            
        except Exception as e:
            logger.error(f"Error getting user orders: {e}")
            return []
    
    def get_order_by_id(self, order_id: str) -> Optional[SimulatedOrder]:
        """Get order by ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM simulated_orders WHERE order_id = ?', (order_id,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return SimulatedOrder(
                    order_id=row[1],
                    user_id=row[2],
                    symbol=row[3],
                    order_type=row[4],
                    quantity=row[5],
                    entry_price=row[6],
                    current_price=row[7],
                    tp_level=row[8],
                    sl_level=row[9],
                    status=row[10],
                    pnl=row[11],
                    pnl_percent=row[12],
                    created_at=row[13],
                    closed_at=row[14],
                    close_reason=row[15]
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting order by ID: {e}")
            return None
    
    def update_order_status(self, order: SimulatedOrder) -> bool:
        """Update order status and details"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE simulated_orders SET
                    current_price = ?, status = ?, pnl = ?, pnl_percent = ?,
                    closed_at = ?, close_reason = ?
                WHERE order_id = ?
            ''', (
                order.current_price, order.status, order.pnl, order.pnl_percent,
                order.closed_at, order.close_reason, order.order_id
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error updating order status: {e}")
            return False
    
    def get_user_cash_balance(self, user_id: int) -> float:
        """Get user's current cash balance"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT cash_balance FROM users WHERE id = ?', (user_id,))
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result else 10000.0
            
        except Exception as e:
            logger.error(f"Error getting user cash balance: {e}")
            return 10000.0
    
    def update_user_balance(self, user_id: int, balance_change: float, transaction_type: str = 'TRADE', description: str = None) -> bool:
        """Update user's cash balance"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get current balance
            cursor.execute('SELECT cash_balance FROM users WHERE id = ?', (user_id,))
            current_balance = cursor.fetchone()[0]
            
            # Calculate new balance
            new_balance = current_balance + balance_change
            
            # Update user's balance
            cursor.execute('UPDATE users SET cash_balance = ? WHERE id = ?', (new_balance, user_id))
            
            # Log the transaction
            cursor.execute('''
                INSERT INTO balance_history (user_id, balance_change, new_balance, transaction_type, description, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, balance_change, new_balance, transaction_type, description, datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error updating user balance: {e}")
            return False

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

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        session_token = session.get('session_token')
        if not session_token:
            return jsonify({'error': 'Authentication required'}), 401
        
        user = db_manager.get_user_by_session(session_token)
        if not user:
            session.pop('session_token', None)
            return jsonify({'error': 'Invalid session'}), 401
        
        request.user = user
        return f(*args, **kwargs)
    return decorated_function

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-this')
CORS(app)

# Initialize managers
cache_manager = CacheManager()
rate_limiter = RateLimiter(max_calls_per_minute=20) # Increased limit
data_provider = TwelveDataProvider(cache_manager=cache_manager, rate_limiter=rate_limiter)
db_manager = DatabaseManager()
notification_manager = NotificationManager()
trading_simulator = TradingSimulator(db_manager, data_provider)
trading_systems = {}

# Authentication routes
@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.json
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        confirm_password = data.get('confirm_password', '')
        first_name = data.get('first_name', '').strip()
        last_name = data.get('last_name', '').strip()
        
        if not username or not email or not password:
            return jsonify({'error': 'Username, email, and password are required'}), 400

        if password != confirm_password:
            return jsonify({'error': 'Passwords do not match'}), 400

        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters long'}), 400
        
        result = db_manager.create_user(username, email, password, first_name, last_name)
        
        if result['success']:
            return jsonify({'message': result['message']}), 201
        else:
            return jsonify({'error': result['message']}), 400
            
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.json
        username = data.get('username', '').strip()
        password = data.get('password', '')

        if not username or not password:
            return jsonify({'error': 'Username and password are required'}), 400

        result = db_manager.authenticate_user(username, password)

        if result['success']:
            session_token = db_manager.create_session(result['user']['id'])
            session['session_token'] = session_token
            return jsonify({
                'message': 'Login successful',
                'user': result['user'],
                'session_token': session_token
            })
        else:
            return jsonify({'error': result['message']}), 401

    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500

@app.route('/api/logout', methods=['POST'])
@login_required
def logout():
    session.pop('session_token', None)
    return jsonify({'message': 'Logged out successfully'})

@app.route('/api/profile', methods=['GET'])
@login_required
def get_profile():
    return jsonify({'user': request.user})

@app.route('/api/watchlist', methods=['GET', 'POST', 'DELETE'])
@login_required
def manage_watchlist():
    if request.method == 'GET':
        watchlist = db_manager.get_user_watchlist(request.user['id'])
        return jsonify({'watchlist': watchlist})
    
    elif request.method == 'POST':
        data = request.json
        symbol = data.get('symbol', '').upper().strip()
        
        if not symbol:
            return jsonify({'error': 'Symbol is required'}), 400
        
        success = db_manager.add_to_watchlist(request.user['id'], symbol)
        if success:
            return jsonify({'message': f'{symbol} added to watchlist'})
        else:
            return jsonify({'error': 'Failed to add to watchlist'}), 500

@app.route('/api/notifications', methods=['GET'])
@login_required
def get_notifications():
    notifications = db_manager.get_user_notifications(request.user['id'])
    return jsonify({'notifications': notifications})

# NEW TRADING SIMULATION ENDPOINTS

@app.route('/api/order', methods=['POST'])
@login_required
def place_order():
    """Place a simulated buy or sell order"""
    try:
        data = request.json
        symbol = data.get('symbol', '').upper().strip()
        order_type = data.get('order_type', '').upper()
        quantity = float(data.get('quantity', 0))
        tp_level = data.get('tp_level')
        sl_level = data.get('sl_level')
        
        # Validation
        if not symbol:
            return jsonify({'error': 'Symbol is required'}), 400
        
        if order_type not in ['BUY', 'SELL']:
            return jsonify({'error': 'Order type must be BUY or SELL'}), 400
        
        if quantity <= 0:
            return jsonify({'error': 'Quantity must be greater than 0'}), 400
        
        if tp_level is not None:
            tp_level = float(tp_level)
        
        if sl_level is not None:
            sl_level = float(sl_level)
        
        # Place the order
        result = trading_simulator.place_order(
            user_id=request.user['id'],
            symbol=symbol,
            order_type=order_type,
            quantity=quantity,
            tp_level=tp_level,
            sl_level=sl_level
        )
        
        if result['success']:
            return jsonify(result), 201
        else:
            return jsonify({'error': result['message']}), 400
            
    except ValueError as e:
        return jsonify({'error': 'Invalid input data'}), 400
    except Exception as e:
        logger.error(f"Order placement error: {e}")
        return jsonify({'error': 'Failed to place order'}), 500

@app.route('/api/orders', methods=['GET'])
@login_required
def get_orders():
    """Get user's order/trade history"""
    try:
        status = request.args.get('status')  # OPEN, CLOSED, TP_HIT, SL_HIT
        limit = int(request.args.get('limit', 50))
        
        orders = db_manager.get_user_orders(request.user['id'], status, limit)
        
        # Convert orders to JSON-serializable format
        orders_data = []
        for order in orders:
            orders_data.append({
                'order_id': order.order_id,
                'symbol': order.symbol,
                'order_type': order.order_type,
                'quantity': order.quantity,
                'entry_price': order.entry_price,
                'current_price': order.current_price,
                'tp_level': order.tp_level,
                'sl_level': order.sl_level,
                'status': order.status,
                'pnl': order.pnl,
                'pnl_percent': order.pnl_percent,
                'created_at': order.created_at,
                'closed_at': order.closed_at,
                'close_reason': order.close_reason
            })
        
        return jsonify({
            'orders': orders_data,
            'total_count': len(orders_data)
        })
        
    except Exception as e:
        logger.error(f"Error getting orders: {e}")
        return jsonify({'error': 'Failed to get orders'}), 500

@app.route('/api/portfolio', methods=['GET'])
@login_required
def get_portfolio():
    """Get user's portfolio including open positions and cash balance"""
    try:
        portfolio = trading_simulator.get_portfolio(request.user['id'])
        
        # Convert to JSON-serializable format
        portfolio_data = {
            'user_id': portfolio.user_id,
            'cash_balance': portfolio.cash_balance,
            'total_equity': portfolio.total_equity,
            'total_pnl': portfolio.total_pnl,
            'total_pnl_percent': portfolio.total_pnl_percent,
            'margin_used': portfolio.margin_used,
            'free_margin': portfolio.free_margin,
            'open_positions': []
        }
        
        for position in portfolio.open_positions:
            portfolio_data['open_positions'].append({
                'order_id': position.order_id,
                'symbol': position.symbol,
                'order_type': position.order_type,
                'quantity': position.quantity,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'tp_level': position.tp_level,
                'sl_level': position.sl_level,
                'pnl': position.pnl,
                'pnl_percent': position.pnl_percent,
                'created_at': position.created_at
            })
        
        return jsonify(portfolio_data)
        
    except Exception as e:
        logger.error(f"Error getting portfolio: {e}")
        return jsonify({'error': 'Failed to get portfolio'}), 500

@app.route('/api/order/<order_id>/close', methods=['POST'])
@login_required
def close_order(order_id):
    """Manually close an open order"""
    try:
        # Verify order belongs to user
        order = db_manager.get_order_by_id(order_id)
        if not order or order.user_id != request.user['id']:
            return jsonify({'error': 'Order not found'}), 404
        
        if order.status != 'OPEN':
            return jsonify({'error': 'Order is already closed'}), 400
        
        # Get current price
        price_data = data_provider.get_current_price(order.symbol)
        if 'error' in price_data:
            return jsonify({'error': price_data['error']}), 429 # Too Many Requests
        current_price = price_data.get('price', order.entry_price)
        
        # Close the order
        result = trading_simulator.close_order(order_id, current_price, 'MANUAL')
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify({'error': result['message']}), 400
            
    except Exception as e:
        logger.error(f"Error closing order: {e}")
        return jsonify({'error': 'Failed to close order'}), 500

@app.route('/api/portfolio/summary', methods=['GET'])
@login_required
def get_portfolio_summary():
    """Get portfolio performance summary"""
    try:
        portfolio = trading_simulator.get_portfolio(request.user['id'])
        
        # Get closed orders for performance metrics
        closed_orders = db_manager.get_user_orders(request.user['id'], status='CLOSED', limit=100)
        tp_hit_orders = db_manager.get_user_orders(request.user['id'], status='TP_HIT', limit=100)
        sl_hit_orders = db_manager.get_user_orders(request.user['id'], status='SL_HIT', limit=100)
        
        all_closed = closed_orders + tp_hit_orders + sl_hit_orders
        
        # Calculate performance metrics
        total_trades = len(all_closed)
        winning_trades = len([o for o in all_closed if o.pnl > 0])
        losing_trades = len([o for o in all_closed if o.pnl < 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_profit = sum([o.pnl for o in all_closed if o.pnl > 0])
        total_loss = sum([o.pnl for o in all_closed if o.pnl < 0])
        
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
        
        avg_win = total_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = total_loss / losing_trades if losing_trades > 0 else 0
        
        summary = {
            'portfolio': {
                'cash_balance': portfolio.cash_balance,
                'total_equity': portfolio.total_equity,
                'total_pnl': portfolio.total_pnl,
                'total_pnl_percent': portfolio.total_pnl_percent,
                'free_margin': portfolio.free_margin,
                'open_positions_count': len(portfolio.open_positions)
            },
            'performance': {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_profit': total_profit,
                'total_loss': total_loss,
                'avg_win': avg_win,
                'avg_loss': avg_loss
            }
        }
        
        return jsonify(summary)
        
    except Exception as e:
        logger.error(f"Error getting portfolio summary: {e}")
        return jsonify({'error': 'Failed to get portfolio summary'}), 500

# Enhanced analysis endpoint
@app.route('/api/analyze_enhanced', methods=['POST'])
@login_required
def analyze_enhanced():
    try:
        data = request.json
        symbol = data.get('symbol', 'AAPL').upper()
        risk_tolerance = float(data.get('risk_tolerance', request.user.get('risk_tolerance', 0.02)))
        wavelet_type = data.get('wavelet_type', config.WAVELET_TYPE)
        
        # Create enhanced trading system
        system_key = f"enhanced_{wavelet_type}_{risk_tolerance}"
        if system_key not in trading_systems:
            trading_systems[system_key] = AdvancedTradingSystem(
                wavelet_type=wavelet_type,
                risk_tolerance=risk_tolerance
            )
        
        trading_system = trading_systems[system_key]
        
        # Get data
        current_data = data_provider.get_current_price(symbol)
        if 'error' in current_data:
            return jsonify({'error': current_data['error']}), 429
            
        historical_prices = data_provider.get_historical_data(symbol, 100)
        
        if len(historical_prices) < 50:
            return jsonify({'error': 'Insufficient historical data. Need at least 50 points'}), 400
        
        # Analyze with enhanced system
        price_array = np.array([to_scalar(x) for x in historical_prices])
        enhanced_signal = trading_system.analyze_enhanced(price_array, risk_tolerance)
        
        # Save signal
        db_manager.save_enhanced_signal(symbol, enhanced_signal, request.user['id'])
        
        # Send notification if high confidence signal
        if enhanced_signal.confidence > 70:
            notification_title = f"🚀 High Confidence {enhanced_signal.signal} Signal"
            notification_message = f"{symbol}: {enhanced_signal.signal} at ${enhanced_signal.entry_price:.4f} ({enhanced_signal.confidence:.1f}% confidence)"
            
            # Add in-app notification
            db_manager.add_notification(
                request.user['id'],
                notification_title,
                notification_message,
                'signal'
            )
            
            # Send email notification if enabled
            if request.user.get('email_notifications', True):
                email_html = notification_manager.create_signal_email(enhanced_signal, symbol)
                notification_manager.send_email_notification(
                    request.user['email'],
                    notification_title,
                    email_html
                )
        
        # Convert to JSON-serializable format
        def serialize_prediction_level(pred_level):
            return {
                'price': pred_level.price,
                'probability': pred_level.probability,
                'time_to_target_hours': pred_level.time_to_target_hours,
                'confidence_interval': pred_level.confidence_interval
            }
        
        def serialize_timeframe_prediction(tf_pred):
            return {
                'timeframe': tf_pred.timeframe,
                'direction': tf_pred.direction,
                'magnitude': tf_pred.magnitude,
                'predictions': [serialize_prediction_level(p) for p in tf_pred.predictions],
                'reliability_score': tf_pred.reliability_score
            }
        
        # --- MODIFIED RESPONSE DICTIONARY ---
        response = {
            'signal': enhanced_signal.signal,
            'confidence': enhanced_signal.confidence,
            'entry_price': enhanced_signal.entry_price,
            'estimated_entry_price': enhanced_signal.estimated_entry_price,
            'estimated_time_to_target': enhanced_signal.estimated_time_to_target,
            
            'kelly_positioning': {
                'optimal_position_size': enhanced_signal.kelly_positioning.optimal_position_size,
                'expected_return': enhanced_signal.kelly_positioning.expected_return,
                'win_probability': enhanced_signal.kelly_positioning.win_probability,
                'avg_win_loss_ratio': enhanced_signal.kelly_positioning.avg_win_loss_ratio,
                'max_position_size': enhanced_signal.kelly_positioning.max_position_size
            },
            
            'combined_prediction': serialize_timeframe_prediction(enhanced_signal.combined_prediction),
            
            'timeframe_predictions': {
                tf: serialize_timeframe_prediction(pred) 
                for tf, pred in enhanced_signal.timeframe_predictions.items()
            },
            
            'dynamic_tp_levels': [serialize_prediction_level(tp) for tp in enhanced_signal.dynamic_tp_levels],
            'dynamic_sl_level': serialize_prediction_level(enhanced_signal.dynamic_sl_level),
            
            'current_data': current_data,
            'timestamp': enhanced_signal.timestamp
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Enhanced analysis error: {e}")
        return jsonify({'error': str(e)}), 500

# Original analysis endpoint (now requires auth)
@app.route('/api/analyze', methods=['POST'])
@login_required
def analyze_symbol():
    try:
        data = request.json
        symbol = data.get('symbol', 'AAPL')
        timeframe = data.get('timeframe', '1hr')
        risk_tolerance = float(data.get('risk_tolerance', request.user.get('risk_tolerance', config.RISK_TOLERANCE)))
        wavelet_type = data.get('wavelet_type', config.WAVELET_TYPE)
        
        system_key = f"{wavelet_type}_{risk_tolerance}"
        if system_key not in trading_systems:
            trading_systems[system_key] = AdvancedTradingSystem(
                wavelet_type=wavelet_type,
                risk_tolerance=risk_tolerance
            )
        
        trading_system = trading_systems[system_key]
        
        current_data = data_provider.get_current_price(symbol)
        if 'error' in current_data:
            return jsonify({'error': current_data['error']}), 429
            
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

# Other endpoints (update with auth where needed)
@app.route('/api/price/<symbol>', methods=['GET'])
def get_price(symbol):
    try:
        price_data = data_provider.get_current_price(symbol)
        if 'error' in price_data:
            return jsonify(price_data), 429
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
        if not historical_data:
            return jsonify({'error': 'Rate limit exceeded or no data available'}), 429
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
@login_required
def get_signals_history(symbol):
    try:
        conn = sqlite3.connect(db_manager.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM enhanced_signals WHERE symbol = ? AND user_id = ?
            ORDER BY timestamp DESC LIMIT 10
        ''', (symbol, request.user['id']))
        signals = cursor.fetchall()
        conn.close()
        
        signal_list = []
        for signal in signals:
            signal_dict = {
                'id': signal[0],
                'symbol': signal[2],
                'signal': signal[3],
                'confidence': to_scalar(signal[4]),
                'entry_price': to_scalar(signal[5]),
                'kelly_position_size': to_scalar(signal[6]),
                'win_probability': to_scalar(signal[7]),
                'prediction_direction': signal[8],
                'prediction_magnitude': to_scalar(signal[9]),
                'reliability_score': to_scalar(signal[10]),
                'timestamp': signal[14]
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
        'version': '4.1.0', # Version incremented
        'features': ['authentication', 'notifications', 'enhanced_signals', 'kelly_sizing', 'trading_simulation', 'caching', 'rate_limiting', 'time_prediction'],
        'data_provider': 'Twelve Data'
    })

@app.route("/", methods=["GET"])
def home():
    return render_template('index.html')

def background_updater():
    """Background process to update prices and send notifications for watchlisted symbols"""
    symbols = ['EUR/USD', 'XAU/USD', 'USD/JPY', 'AAPL', 'BTC/USD']
    
    while True:
        try:
            # Update general market data
            for symbol in symbols:
                try:
                    price_data = data_provider.get_current_price(symbol)
                    if 'error' not in price_data:
                        db_manager.save_price_data(price_data)
                        logger.info(f"Updated price for {symbol}: {price_data['price']}")
                    else:
                        logger.warning(f"Could not update price for {symbol}: {price_data.get('error')}")
                    time.sleep(2)
                except Exception as e:
                    logger.error(f"Error updating {symbol}: {e}")
                    continue
            
            # Check for new signals on watchlisted symbols
            conn = sqlite3.connect(db_manager.db_path)
            cursor = conn.cursor()
            
            # Get all unique watchlisted symbols
            cursor.execute('SELECT DISTINCT symbol FROM user_watchlist')
            watchlisted_symbols = [row[0] for row in cursor.fetchall()]
            
            for symbol in watchlisted_symbols:
                try:
                    # Get users watching this symbol
                    cursor.execute('''
                        SELECT u.id, u.email, u.email_notifications, u.risk_tolerance
                        FROM users u
                        JOIN user_watchlist w ON u.id = w.user_id
                        WHERE w.symbol = ? AND u.is_active = 1
                    ''', (symbol,))
                    
                    watching_users = cursor.fetchall()
                    
                    if watching_users:
                        # Analyze the symbol
                        historical_prices = data_provider.get_historical_data(symbol, 100)
                        if len(historical_prices) >= 50:
                            # Use enhanced trading system
                            system_key = "enhanced_db4_0.02"
                            if system_key not in trading_systems:
                                trading_systems[system_key] = AdvancedTradingSystem(
                                    wavelet_type='db4',
                                    risk_tolerance=0.02
                                )
                            
                            trading_system = trading_systems[system_key]
                            price_array = np.array([to_scalar(x) for x in historical_prices])
                            # Use a default risk tolerance for background checks
                            enhanced_signal = trading_system.analyze_enhanced(price_array, 0.02)
                            
                            # If high confidence signal, notify users
                            if enhanced_signal.confidence > 75 and enhanced_signal.signal != 'HOLD':
                                for user_data in watching_users:
                                    user_id, email, email_notifications, risk_tolerance = user_data
                                    
                                    # Save signal for user
                                    db_manager.save_enhanced_signal(symbol, enhanced_signal, user_id)
                                    
                                    # Add notification
                                    notification_title = f"🎯 {enhanced_signal.signal} Signal Alert"
                                    notification_message = f"{symbol}: {enhanced_signal.signal} signal with {enhanced_signal.confidence:.1f}% confidence"
                                    
                                    db_manager.add_notification(
                                        user_id,
                                        notification_title,
                                        notification_message,
                                        'watchlist_alert'
                                    )
                                    
                                    # Send email if enabled
                                    if email_notifications:
                                        email_html = notification_manager.create_signal_email(enhanced_signal, symbol)
                                        notification_manager.send_email_notification(
                                            email,
                                            notification_title,
                                            email_html
                                        )
                                
                                logger.info(f"Sent notifications for {symbol} signal to {len(watching_users)} users")
                    
                    time.sleep(5)
                    
                except Exception as e:
                    logger.error(f"Error processing watchlisted symbol {symbol}: {e}")
                    continue
            
            # Check and auto-close positions based on TP/SL
            try:
                cursor.execute('SELECT DISTINCT user_id FROM simulated_orders WHERE status = "OPEN"')
                active_users = [row[0] for row in cursor.fetchall()]
                
                for user_id in active_users:
                    portfolio = trading_simulator.get_portfolio(user_id)
                    # The get_portfolio method already handles TP/SL checking
                    
            except Exception as e:
                logger.error(f"Error checking TP/SL conditions: {e}")
            
            conn.close()
            
            logger.info("Background update cycle completed, waiting 15 minutes...")
            time.sleep(900)  # 15 minutes
            
        except Exception as e:
            logger.error(f"Background update error: {e}")
            time.sleep(60)

if __name__ == '__main__':
    # Start background updater
    updater_thread = threading.Thread(target=background_updater, daemon=True)
    updater_thread.start()
    logger.info("Background updater started with 15-minute intervals")
    
    # Start Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
