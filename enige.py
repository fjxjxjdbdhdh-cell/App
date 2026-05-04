# gm_engine.py - GM Trading AI Engine - Technical Analysis & Market Engine
import os
import sys
import json
import time
import random
import logging
import asyncio
import hashlib
import re
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict, deque

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

logger = logging.getLogger(__name__)

# ============================================================
# TECHNICAL ANALYZER
# ============================================================

class TechnicalAnalyzer:
    def __init__(self):
        self.name = "TechnicalAnalyzer"

    def calculate_indicators(self, ohlcv: list) -> dict:
        try:
            if not ohlcv or len(ohlcv) < 20:
                return self._empty_indicators()
            if np is None or pd is None:
                return self._empty_indicators()

            df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
            closes = df['close'].values.astype(float)
            highs = df['high'].values.astype(float)
            lows = df['low'].values.astype(float)
            opens = df['open'].values.astype(float)
            volumes = df['volume'].values.astype(float)

            result = {}

            # --- EMA ---
            for period in [8, 13, 21, 34, 55, 89, 200]:
                result[f'ema_{period}'] = self._ema(closes, period)

            # --- SMA ---
            for period in [20, 50, 100, 200]:
                result[f'sma_{period}'] = self._sma(closes, period)

            # --- RSI ---
            result['rsi_14'] = self._rsi(closes, 14)
            result['rsi_7'] = self._rsi(closes, 7)
            result['rsi_21'] = self._rsi(closes, 21)

            # --- MACD ---
            macd_line, signal_line, histogram = self._macd(closes)
            result['macd'] = macd_line
            result['macd_signal'] = signal_line
            result['macd_histogram'] = histogram

            # --- Bollinger Bands ---
            bb_upper, bb_middle, bb_lower = self._bollinger_bands(closes, 20, 2)
            result['bb_upper'] = bb_upper
            result['bb_middle'] = bb_middle
            result['bb_lower'] = bb_lower
            result['bb_width'] = (bb_upper - bb_lower) / (bb_middle + 1e-10)
            result['bb_percent'] = (closes[-1] - bb_lower) / (bb_upper - bb_lower + 1e-10)

            # --- ATR ---
            result['atr_14'] = self._atr(highs, lows, closes, 14)
            result['atr_7'] = self._atr(highs, lows, closes, 7)

            # --- ADX ---
            adx, di_plus, di_minus = self._adx(highs, lows, closes, 14)
            result['adx'] = adx
            result['di_plus'] = di_plus
            result['di_minus'] = di_minus

            # --- Stochastic ---
            stoch_k, stoch_d = self._stochastic(highs, lows, closes, 14, 3)
            result['stoch_k'] = stoch_k
            result['stoch_d'] = stoch_d

            # --- Ichimoku ---
            ichimoku = self._ichimoku(highs, lows, closes)
            result.update(ichimoku)

            # --- Parabolic SAR ---
            result['parabolic_sar'] = self._parabolic_sar(highs, lows)

            # --- CCI ---
            result['cci_14'] = self._cci(highs, lows, closes, 14)
            result['cci_20'] = self._cci(highs, lows, closes, 20)

            # --- Williams %R ---
            result['williams_r'] = self._williams_r(highs, lows, closes, 14)

            # --- ROC ---
            result['roc_10'] = self._roc(closes, 10)
            result['roc_20'] = self._roc(closes, 20)

            # --- TSI ---
            result['tsi'] = self._tsi(closes)

            # --- Ultimate Oscillator ---
            result['ultimate_oscillator'] = self._ultimate_oscillator(highs, lows, closes)

            # --- MFI ---
            result['mfi_14'] = self._mfi(highs, lows, closes, volumes, 14)

            # --- OBV ---
            result['obv'] = self._obv(closes, volumes)

            # --- CMF ---
            result['cmf_20'] = self._cmf(highs, lows, closes, volumes, 20)

            # --- VWAP ---
            result['vwap'] = self._vwap(highs, lows, closes, volumes)

            # --- Keltner Channels ---
            kc_upper, kc_middle, kc_lower = self._keltner_channels(highs, lows, closes)
            result['kc_upper'] = kc_upper
            result['kc_middle'] = kc_middle
            result['kc_lower'] = kc_lower

            # --- Momentum ---
            result['momentum_10'] = self._momentum(closes, 10)

            # --- Volatility ---
            result['volatility'] = float(np.std(closes[-20:]) / np.mean(closes[-20:]) * 100) if len(closes) >= 20 else 0.0

            # --- Trend direction ---
            result['trend_direction'] = self._get_trend_direction(closes, result)

            # --- Candle patterns ---
            result['candle_patterns'] = self._detect_candle_patterns(opens, highs, lows, closes)

            # --- Price ---
            result['current_price'] = float(closes[-1])
            result['prev_close'] = float(closes[-2]) if len(closes) >= 2 else float(closes[-1])
            result['price_change_pct'] = float((closes[-1] - closes[-2]) / closes[-2] * 100) if len(closes) >= 2 else 0.0

            return result

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return self._empty_indicators()

    def _ema(self, data, period):
        try:
            if len(data) < period:
                return float(data[-1]) if len(data) > 0 else 0.0
            alpha = 2.0 / (period + 1)
            ema = float(data[0])
            for price in data[1:]:
                ema = alpha * float(price) + (1 - alpha) * ema
            return ema
        except:
            return 0.0

    def _sma(self, data, period):
        try:
            if len(data) < period:
                return float(np.mean(data)) if len(data) > 0 else 0.0
            return float(np.mean(data[-period:]))
        except:
            return 0.0

    def _rsi(self, data, period=14):
        try:
            if len(data) < period + 1:
                return 50.0
            deltas = np.diff(data.astype(float))
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            if avg_loss == 0:
                return 100.0
            rs = avg_gain / avg_loss
            return float(100 - (100 / (1 + rs)))
        except:
            return 50.0

    def _macd(self, data, fast=12, slow=26, signal=9):
        try:
            if len(data) < slow:
                return 0.0, 0.0, 0.0
            ema_fast = self._ema(data, fast)
            ema_slow = self._ema(data, slow)
            macd_line = ema_fast - ema_slow

            macd_values = []
            for i in range(signal, len(data) + 1):
                ef = self._ema(data[:i], fast)
                es = self._ema(data[:i], slow)
                macd_values.append(ef - es)

            signal_line = self._ema(np.array(macd_values), signal) if len(macd_values) >= signal else macd_line
            histogram = macd_line - signal_line
            return float(macd_line), float(signal_line), float(histogram)
        except:
            return 0.0, 0.0, 0.0

    def _bollinger_bands(self, data, period=20, std_dev=2):
        try:
            if len(data) < period:
                price = float(data[-1]) if len(data) > 0 else 0.0
                return price * 1.02, price, price * 0.98
            sma = float(np.mean(data[-period:]))
            std = float(np.std(data[-period:]))
            upper = sma + std_dev * std
            lower = sma - std_dev * std
            return upper, sma, lower
        except:
            return 0.0, 0.0, 0.0

    def _atr(self, highs, lows, closes, period=14):
        try:
            if len(highs) < 2:
                return 0.0
            trs = []
            for i in range(1, len(highs)):
                h = float(highs[i])
                l = float(lows[i])
                pc = float(closes[i-1])
                tr = max(h - l, abs(h - pc), abs(l - pc))
                trs.append(tr)
            if len(trs) < period:
                return float(np.mean(trs)) if trs else 0.0
            return float(np.mean(trs[-period:]))
        except:
            return 0.0

    def _adx(self, highs, lows, closes, period=14):
        try:
            if len(highs) < period + 1:
                return 25.0, 25.0, 25.0
            dm_plus_list = []
            dm_minus_list = []
            tr_list = []
            for i in range(1, len(highs)):
                up_move = float(highs[i]) - float(highs[i-1])
                down_move = float(lows[i-1]) - float(lows[i])
                dm_plus_list.append(up_move if up_move > down_move and up_move > 0 else 0)
                dm_minus_list.append(down_move if down_move > up_move and down_move > 0 else 0)
                h = float(highs[i])
                l = float(lows[i])
                pc = float(closes[i-1])
                tr_list.append(max(h - l, abs(h - pc), abs(l - pc)))

            atr = float(np.mean(tr_list[-period:])) if len(tr_list) >= period else 1.0
            di_plus = float(np.mean(dm_plus_list[-period:]) / atr * 100) if atr > 0 else 0.0
            di_minus = float(np.mean(dm_minus_list[-period:]) / atr * 100) if atr > 0 else 0.0

            dx_list = []
            for i in range(len(tr_list) - period, len(tr_list)):
                if i < 0:
                    continue
                atr_i = float(np.mean(tr_list[max(0, i-period):i+1])) if i >= period else atr
                dip = float(np.mean(dm_plus_list[max(0, i-period):i+1]) / atr_i * 100) if atr_i > 0 else 0
                dim = float(np.mean(dm_minus_list[max(0, i-period):i+1]) / atr_i * 100) if atr_i > 0 else 0
                dx = abs(dip - dim) / (dip + dim + 1e-10) * 100
                dx_list.append(dx)

            adx = float(np.mean(dx_list)) if dx_list else 25.0
            return adx, di_plus, di_minus
        except:
            return 25.0, 25.0, 25.0

    def _stochastic(self, highs, lows, closes, k_period=14, d_period=3):
        try:
            if len(highs) < k_period:
                return 50.0, 50.0
            high_max = float(np.max(highs[-k_period:]))
            low_min = float(np.min(lows[-k_period:]))
            if high_max == low_min:
                return 50.0, 50.0
            k = (float(closes[-1]) - low_min) / (high_max - low_min) * 100
            k_values = []
            for i in range(k_period, len(highs) + 1):
                hm = float(np.max(highs[i-k_period:i]))
                lm = float(np.min(lows[i-k_period:i]))
                if hm != lm:
                    k_values.append((float(closes[i-1]) - lm) / (hm - lm) * 100)
                else:
                    k_values.append(50.0)
            d = float(np.mean(k_values[-d_period:])) if len(k_values) >= d_period else k
            return float(k), float(d)
        except:
            return 50.0, 50.0

    def _ichimoku(self, highs, lows, closes):
        try:
            n = len(closes)
            tenkan = (float(np.max(highs[-9:])) + float(np.min(lows[-9:]))) / 2 if n >= 9 else float(closes[-1])
            kijun = (float(np.max(highs[-26:])) + float(np.min(lows[-26:]))) / 2 if n >= 26 else float(closes[-1])
            senkou_a = (tenkan + kijun) / 2
            senkou_b = (float(np.max(highs[-52:])) + float(np.min(lows[-52:]))) / 2 if n >= 52 else float(closes[-1])
            chikou = float(closes[-1])
            return {
                'ichimoku_tenkan': tenkan,
                'ichimoku_kijun': kijun,
                'ichimoku_senkou_a': senkou_a,
                'ichimoku_senkou_b': senkou_b,
                'ichimoku_chikou': chikou,
            }
        except:
            p = float(closes[-1]) if len(closes) > 0 else 0.0
            return {
                'ichimoku_tenkan': p, 'ichimoku_kijun': p,
                'ichimoku_senkou_a': p, 'ichimoku_senkou_b': p, 'ichimoku_chikou': p
            }

    def _parabolic_sar(self, highs, lows, af_start=0.02, af_max=0.2):
        try:
            if len(highs) < 2:
                return float(lows[-1]) if lows else 0.0
            bullish = True
            sar = float(lows[0])
            ep = float(highs[0])
            af = af_start
            for i in range(1, len(highs)):
                sar = sar + af * (ep - sar)
                if bullish:
                    if float(lows[i]) < sar:
                        bullish = False
                        sar = ep
                        ep = float(lows[i])
                        af = af_start
                    else:
                        if float(highs[i]) > ep:
                            ep = float(highs[i])
                            af = min(af + af_start, af_max)
                        sar = min(sar, float(lows[i-1]), float(lows[i]))
                else:
                    if float(highs[i]) > sar:
                        bullish = True
                        sar = ep
                        ep = float(highs[i])
                        af = af_start
                    else:
                        if float(lows[i]) < ep:
                            ep = float(lows[i])
                            af = min(af + af_start, af_max)
                        sar = max(sar, float(highs[i-1]), float(highs[i]))
            return float(sar)
        except:
            return float(lows[-1]) if lows else 0.0

    def _cci(self, highs, lows, closes, period=14):
        try:
            if len(closes) < period:
                return 0.0
            typical = [(float(highs[i]) + float(lows[i]) + float(closes[i])) / 3 for i in range(len(closes))]
            mean_tp = float(np.mean(typical[-period:]))
            mean_dev = float(np.mean([abs(t - mean_tp) for t in typical[-period:]]))
            if mean_dev == 0:
                return 0.0
            return float((typical[-1] - mean_tp) / (0.015 * mean_dev))
        except:
            return 0.0

    def _williams_r(self, highs, lows, closes, period=14):
        try:
            if len(closes) < period:
                return -50.0
            high_max = float(np.max(highs[-period:]))
            low_min = float(np.min(lows[-period:]))
            if high_max == low_min:
                return -50.0
            return float((high_max - closes[-1]) / (high_max - low_min) * -100)
        except:
            return -50.0

    def _roc(self, data, period=10):
        try:
            if len(data) <= period:
                return 0.0
            old = float(data[-period-1])
            if old == 0:
                return 0.0
            return float((float(data[-1]) - old) / old * 100)
        except:
            return 0.0

    def _tsi(self, data, r=25, s=13):
        try:
            if len(data) < r + s:
                return 0.0
            diffs = np.diff(data.astype(float))
            smoothed = self._double_smooth(diffs, r, s)
            abs_smoothed = self._double_smooth(np.abs(diffs), r, s)
            if abs_smoothed == 0:
                return 0.0
            return float(smoothed / abs_smoothed * 100)
        except:
            return 0.0

    def _double_smooth(self, data, r, s):
        try:
            if len(data) < r:
                return float(np.mean(data)) if len(data) > 0 else 0.0
            first = self._ema(data, r)
            arr = np.array([self._ema(data[:i], r) for i in range(1, len(data)+1)])
            return self._ema(arr, s)
        except:
            return 0.0

    def _ultimate_oscillator(self, highs, lows, closes, p1=7, p2=14, p3=28):
        try:
            if len(closes) < p3 + 1:
                return 50.0
            bp_list = []
            tr_list = []
            for i in range(1, len(closes)):
                pc = float(closes[i-1])
                h = float(highs[i])
                l = float(lows[i])
                c = float(closes[i])
                bp = c - min(l, pc)
                tr = max(h, pc) - min(l, pc)
                bp_list.append(bp)
                tr_list.append(tr)

            def avg(bp, tr, period):
                if len(bp) < period:
                    return 0.5
                s_bp = sum(bp[-period:])
                s_tr = sum(tr[-period:])
                return s_bp / s_tr if s_tr > 0 else 0.5

            a1 = avg(bp_list, tr_list, p1)
            a2 = avg(bp_list, tr_list, p2)
            a3 = avg(bp_list, tr_list, p3)
            return float((4*a1 + 2*a2 + a3) / 7 * 100)
        except:
            return 50.0

    def _mfi(self, highs, lows, closes, volumes, period=14):
        try:
            if len(closes) < period + 1:
                return 50.0
            typical = [(float(highs[i]) + float(lows[i]) + float(closes[i])) / 3 for i in range(len(closes))]
            pos_flow = 0.0
            neg_flow = 0.0
            for i in range(len(typical) - period, len(typical)):
                if i <= 0:
                    continue
                mf = typical[i] * float(volumes[i])
                if typical[i] > typical[i-1]:
                    pos_flow += mf
                else:
                    neg_flow += mf
            if neg_flow == 0:
                return 100.0
            mfr = pos_flow / neg_flow
            return float(100 - 100 / (1 + mfr))
        except:
            return 50.0

    def _obv(self, closes, volumes):
        try:
            obv = 0.0
            for i in range(1, len(closes)):
                if float(closes[i]) > float(closes[i-1]):
                    obv += float(volumes[i])
                elif float(closes[i]) < float(closes[i-1]):
                    obv -= float(volumes[i])
            return float(obv)
        except:
            return 0.0

    def _cmf(self, highs, lows, closes, volumes, period=20):
        try:
            if len(closes) < period:
                return 0.0
            mf_vol = 0.0
            vol_sum = 0.0
            for i in range(len(closes) - period, len(closes)):
                h = float(highs[i])
                l = float(lows[i])
                c = float(closes[i])
                v = float(volumes[i])
                if h != l:
                    mf_mult = ((c - l) - (h - c)) / (h - l)
                else:
                    mf_mult = 0.0
                mf_vol += mf_mult * v
                vol_sum += v
            return float(mf_vol / vol_sum) if vol_sum > 0 else 0.0
        except:
            return 0.0

    def _vwap(self, highs, lows, closes, volumes):
        try:
            typical = [(float(highs[i]) + float(lows[i]) + float(closes[i])) / 3 for i in range(len(closes))]
            cum_pv = sum(tp * float(v) for tp, v in zip(typical, volumes))
            cum_vol = sum(float(v) for v in volumes)
            return float(cum_pv / cum_vol) if cum_vol > 0 else float(closes[-1])
        except:
            return float(closes[-1]) if len(closes) > 0 else 0.0

    def _keltner_channels(self, highs, lows, closes, period=20, atr_mult=2):
        try:
            ema = self._ema(closes, period)
            atr = self._atr(highs, lows, closes, period)
            upper = ema + atr_mult * atr
            lower = ema - atr_mult * atr
            return upper, ema, lower
        except:
            p = float(closes[-1]) if len(closes) > 0 else 0.0
            return p * 1.02, p, p * 0.98

    def _momentum(self, data, period=10):
        try:
            if len(data) <= period:
                return 0.0
            return float(data[-1]) - float(data[-period-1])
        except:
            return 0.0

    def _get_trend_direction(self, closes, indicators):
        try:
            ema8 = indicators.get('ema_8', 0)
            ema21 = indicators.get('ema_21', 0)
            ema55 = indicators.get('ema_55', 0)
            price = float(closes[-1]) if len(closes) > 0 else 0.0

            if ema8 > ema21 > ema55 and price > ema8:
                return 'STRONG_UP'
            elif ema8 > ema21 and price > ema21:
                return 'UP'
            elif ema8 < ema21 < ema55 and price < ema8:
                return 'STRONG_DOWN'
            elif ema8 < ema21 and price < ema21:
                return 'DOWN'
            else:
                return 'SIDEWAYS'
        except:
            return 'UNKNOWN'

    def _detect_candle_patterns(self, opens, highs, lows, closes):
        try:
            patterns = []
            if len(closes) < 3:
                return patterns

            # Current and previous candle
            o = float(opens[-1])
            h = float(highs[-1])
            l = float(lows[-1])
            c = float(closes[-1])
            po = float(opens[-2])
            ph = float(highs[-2])
            pl = float(lows[-2])
            pc = float(closes[-2])

            body = abs(c - o)
            upper_shadow = h - max(o, c)
            lower_shadow = min(o, c) - l
            prev_body = abs(pc - po)

            # Hammer
            if (lower_shadow >= 2 * body and upper_shadow <= 0.1 * body
                    and c > o and body > 0):
                patterns.append({'name': 'HAMMER', 'direction': 'BUY', 'strength': 0.7})

            # Shooting Star
            if (upper_shadow >= 2 * body and lower_shadow <= 0.1 * body
                    and c < o and body > 0):
                patterns.append({'name': 'SHOOTING_STAR', 'direction': 'SELL', 'strength': 0.7})

            # Doji
            if body <= 0.05 * (h - l) and (h - l) > 0:
                patterns.append({'name': 'DOJI', 'direction': 'NEUTRAL', 'strength': 0.5})

            # Bullish Engulfing
            if (po > pc and c > o and c > po and o < pc and body > prev_body):
                patterns.append({'name': 'BULLISH_ENGULFING', 'direction': 'BUY', 'strength': 0.8})

            # Bearish Engulfing
            if (pc > po and o > c and o > pc and c < po and body > prev_body):
                patterns.append({'name': 'BEARISH_ENGULFING', 'direction': 'SELL', 'strength': 0.8})

            # Pin Bar BUY
            if (lower_shadow >= 3 * body and upper_shadow <= body * 0.5):
                patterns.append({'name': 'PIN_BAR_BUY', 'direction': 'BUY', 'strength': 0.75})

            # Pin Bar SELL
            if (upper_shadow >= 3 * body and lower_shadow <= body * 0.5):
                patterns.append({'name': 'PIN_BAR_SELL', 'direction': 'SELL', 'strength': 0.75})

            # Three White Soldiers
            if len(closes) >= 3:
                c1, c2, c3 = float(closes[-3]), float(closes[-2]), float(closes[-1])
                o1, o2, o3 = float(opens[-3]), float(opens[-2]), float(opens[-1])
                if c3 > c2 > c1 and o3 > o2 > o1 and c3 > o3 and c2 > o2 and c1 > o1:
                    patterns.append({'name': 'THREE_WHITE_SOLDIERS', 'direction': 'BUY', 'strength': 0.85})

            # Three Black Crows
            if len(closes) >= 3:
                c1, c2, c3 = float(closes[-3]), float(closes[-2]), float(closes[-1])
                o1, o2, o3 = float(opens[-3]), float(opens[-2]), float(opens[-1])
                if c3 < c2 < c1 and o3 < o2 < o1 and c3 < o3 and c2 < o2 and c1 < o1:
                    patterns.append({'name': 'THREE_BLACK_CROWS', 'direction': 'SELL', 'strength': 0.85})

            # Morning Star
            if len(closes) >= 3:
                if (pc < po and abs(c - o) < abs(pc - po) * 0.3 and c > o and
                        float(closes[-3]) < float(opens[-3])):
                    patterns.append({'name': 'MORNING_STAR', 'direction': 'BUY', 'strength': 0.8})

            # Evening Star
            if len(closes) >= 3:
                if (pc > po and abs(c - o) < abs(pc - po) * 0.3 and c < o and
                        float(closes[-3]) > float(opens[-3])):
                    patterns.append({'name': 'EVENING_STAR', 'direction': 'SELL', 'strength': 0.8})

            return patterns
        except Exception as e:
            logger.error(f"Pattern detection error: {e}")
            return []

    def generate_signal(self, indicators: dict, symbol: str = 'XAUUSD') -> dict:
        try:
            if not indicators or not indicators.get('current_price'):
                return {'direction': 'WAIT', 'confidence': 0.0, 'reason': 'No data'}

            bull_score = 0.0
            bear_score = 0.0
            reasons = []

            price = indicators.get('current_price', 0)

            # RSI
            rsi = indicators.get('rsi_14', 50)
            if rsi < 30:
                bull_score += 2.0
                reasons.append(f'RSI oversold ({rsi:.1f})')
            elif rsi > 70:
                bear_score += 2.0
                reasons.append(f'RSI overbought ({rsi:.1f})')
            elif rsi < 45:
                bull_score += 0.5
            elif rsi > 55:
                bear_score += 0.5

            # MACD
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            macd_hist = indicators.get('macd_histogram', 0)
            if macd > macd_signal and macd_hist > 0:
                bull_score += 1.5
                reasons.append('MACD bullish crossover')
            elif macd < macd_signal and macd_hist < 0:
                bear_score += 1.5
                reasons.append('MACD bearish crossover')

            # EMA trend
            ema8 = indicators.get('ema_8', price)
            ema21 = indicators.get('ema_21', price)
            ema55 = indicators.get('ema_55', price)
            if ema8 > ema21 > ema55:
                bull_score += 2.0
                reasons.append('EMA bullish alignment')
            elif ema8 < ema21 < ema55:
                bear_score += 2.0
                reasons.append('EMA bearish alignment')

            # ADX
            adx = indicators.get('adx', 25)
            di_plus = indicators.get('di_plus', 25)
            di_minus = indicators.get('di_minus', 25)
            if adx > 25:
                if di_plus > di_minus:
                    bull_score += 1.5
                    reasons.append(f'ADX strong uptrend ({adx:.1f})')
                else:
                    bear_score += 1.5
                    reasons.append(f'ADX strong downtrend ({adx:.1f})')

            # Bollinger
            bb_upper = indicators.get('bb_upper', price * 1.02)
            bb_lower = indicators.get('bb_lower', price * 0.98)
            bb_middle = indicators.get('bb_middle', price)
            if price <= bb_lower:
                bull_score += 1.5
                reasons.append('Price at BB lower band')
            elif price >= bb_upper:
                bear_score += 1.5
                reasons.append('Price at BB upper band')

            # Stochastic
            stoch_k = indicators.get('stoch_k', 50)
            stoch_d = indicators.get('stoch_d', 50)
            if stoch_k < 20 and stoch_d < 20:
                bull_score += 1.0
                reasons.append(f'Stoch oversold ({stoch_k:.1f})')
            elif stoch_k > 80 and stoch_d > 80:
                bear_score += 1.0
                reasons.append(f'Stoch overbought ({stoch_k:.1f})')

            # Ichimoku
            tenkan = indicators.get('ichimoku_tenkan', price)
            kijun = indicators.get('ichimoku_kijun', price)
            senkou_a = indicators.get('ichimoku_senkou_a', price)
            senkou_b = indicators.get('ichimoku_senkou_b', price)
            if price > max(senkou_a, senkou_b) and tenkan > kijun:
                bull_score += 1.5
                reasons.append('Price above Ichimoku cloud')
            elif price < min(senkou_a, senkou_b) and tenkan < kijun:
                bear_score += 1.5
                reasons.append('Price below Ichimoku cloud')

            # CCI
            cci = indicators.get('cci_14', 0)
            if cci < -100:
                bull_score += 1.0
                reasons.append(f'CCI oversold ({cci:.0f})')
            elif cci > 100:
                bear_score += 1.0
                reasons.append(f'CCI overbought ({cci:.0f})')

            # Candle patterns
            patterns = indicators.get('candle_patterns', [])
            for p in patterns:
                if p['direction'] == 'BUY':
                    bull_score += p['strength'] * 1.5
                    reasons.append(f"Pattern: {p['name']}")
                elif p['direction'] == 'SELL':
                    bear_score += p['strength'] * 1.5
                    reasons.append(f"Pattern: {p['name']}")

            # MFI
            mfi = indicators.get('mfi_14', 50)
            if mfi < 20:
                bull_score += 0.5
            elif mfi > 80:
                bear_score += 0.5

            # Williams %R
            wr = indicators.get('williams_r', -50)
            if wr < -80:
                bull_score += 0.5
            elif wr > -20:
                bear_score += 0.5

            total = bull_score + bear_score
            if total == 0:
                return {'direction': 'WAIT', 'confidence': 0.0, 'reason': 'No clear signals', 'bull_score': 0, 'bear_score': 0}

            bull_pct = bull_score / total
            bear_pct = bear_score / total

            min_confidence = 0.55
            if bull_pct >= min_confidence:
                direction = 'BUY'
                confidence = min(bull_pct, 0.99)
            elif bear_pct >= min_confidence:
                direction = 'SELL'
                confidence = min(bear_pct, 0.99)
            else:
                direction = 'WAIT'
                confidence = max(bull_pct, bear_pct)

            atr = indicators.get('atr_14', price * 0.001)
            if direction == 'BUY':
                sl = price - atr * 2
                tp = price + atr * 3
            elif direction == 'SELL':
                sl = price + atr * 2
                tp = price - atr * 3
            else:
                sl = price - atr * 2
                tp = price + atr * 3

            return {
                'direction': direction,
                'confidence': round(float(confidence), 4),
                'bull_score': round(float(bull_score), 2),
                'bear_score': round(float(bear_score), 2),
                'reason': '; '.join(reasons[:5]),
                'reasons': reasons,
                'entry': round(float(price), 5),
                'sl': round(float(sl), 5),
                'tp': round(float(tp), 5),
                'rr': round(abs(tp - price) / (abs(sl - price) + 1e-10), 2),
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return {'direction': 'WAIT', 'confidence': 0.0, 'reason': str(e)}

    def detect_market_regime(self, ohlcv: list) -> str:
        try:
            if not ohlcv or len(ohlcv) < 50 or np is None:
                return 'UNKNOWN'

            closes = np.array([float(c[4]) for c in ohlcv[-100:]])
            highs = np.array([float(c[2]) for c in ohlcv[-100:]])
            lows = np.array([float(c[3]) for c in ohlcv[-100:]])

            atr = self._atr(highs, lows, closes, 14)
            avg_price = float(np.mean(closes))
            volatility = atr / avg_price * 100 if avg_price > 0 else 0

            adx, di_plus, di_minus = self._adx(highs, lows, closes, 14)

            price_range = (float(np.max(closes[-20:])) - float(np.min(closes[-20:]))) / avg_price * 100

            if adx > 40:
                if volatility > 1.5:
                    return 'HIGH_VOLATILITY'
                return 'STRONG_TREND'
            elif adx > 25:
                return 'MODERATE_TREND'
            elif price_range < 0.5:
                return 'CONSOLIDATION'
            else:
                return 'RANGING'
        except:
            return 'UNKNOWN'

    def find_support_resistance(self, ohlcv: list, n_levels: int = 5) -> dict:
        try:
            if not ohlcv or len(ohlcv) < 30 or np is None:
                return {'support': [], 'resistance': []}

            highs = [float(c[2]) for c in ohlcv]
            lows = [float(c[3]) for c in ohlcv]
            closes = [float(c[4]) for c in ohlcv]
            current = closes[-1]

            local_highs = []
            local_lows = []
            window = 5

            for i in range(window, len(highs) - window):
                if highs[i] == max(highs[i-window:i+window+1]):
                    local_highs.append(highs[i])
                if lows[i] == min(lows[i-window:i+window+1]):
                    local_lows.append(lows[i])

            resistance_levels = self._cluster_levels([h for h in local_highs if h > current])
            support_levels = self._cluster_levels([l for l in local_lows if l < current])

            resistance_levels.sort()
            support_levels.sort(reverse=True)

            return {
                'support': [{'price': round(l, 5), 'strength': self._calculate_level_strength(l, lows)} for l in support_levels[:n_levels]],
                'resistance': [{'price': round(l, 5), 'strength': self._calculate_level_strength(l, highs)} for l in resistance_levels[:n_levels]]
            }
        except Exception as e:
            logger.error(f"S/R error: {e}")
            return {'support': [], 'resistance': []}

    def _calculate_level_strength(self, level: float, prices: list) -> float:
        try:
            tolerance = level * 0.001
            touches = sum(1 for p in prices if abs(p - level) <= tolerance)
            return min(touches / 10.0, 1.0)
        except:
            return 0.5

    def _cluster_levels(self, levels: list, tolerance_pct: float = 0.002) -> list:
        try:
            if not levels:
                return []
            sorted_levels = sorted(levels)
            clusters = []
            current_cluster = [sorted_levels[0]]
            for level in sorted_levels[1:]:
                if abs(level - current_cluster[-1]) / (current_cluster[-1] + 1e-10) <= tolerance_pct:
                    current_cluster.append(level)
                else:
                    clusters.append(float(np.mean(current_cluster)))
                    current_cluster = [level]
            clusters.append(float(np.mean(current_cluster)))
            return clusters
        except:
            return levels

    def _empty_indicators(self) -> dict:
        return {
            'current_price': 0.0, 'prev_close': 0.0, 'price_change_pct': 0.0,
            'ema_8': 0.0, 'ema_13': 0.0, 'ema_21': 0.0, 'ema_34': 0.0,
            'ema_55': 0.0, 'ema_89': 0.0, 'ema_200': 0.0,
            'sma_20': 0.0, 'sma_50': 0.0, 'sma_100': 0.0, 'sma_200': 0.0,
            'rsi_14': 50.0, 'rsi_7': 50.0, 'rsi_21': 50.0,
            'macd': 0.0, 'macd_signal': 0.0, 'macd_histogram': 0.0,
            'bb_upper': 0.0, 'bb_middle': 0.0, 'bb_lower': 0.0,
            'bb_width': 0.0, 'bb_percent': 0.5,
            'atr_14': 0.0, 'atr_7': 0.0,
            'adx': 25.0, 'di_plus': 25.0, 'di_minus': 25.0,
            'stoch_k': 50.0, 'stoch_d': 50.0,
            'ichimoku_tenkan': 0.0, 'ichimoku_kijun': 0.0,
            'ichimoku_senkou_a': 0.0, 'ichimoku_senkou_b': 0.0, 'ichimoku_chikou': 0.0,
            'parabolic_sar': 0.0,
            'cci_14': 0.0, 'cci_20': 0.0,
            'williams_r': -50.0,
            'roc_10': 0.0, 'roc_20': 0.0,
            'tsi': 0.0, 'ultimate_oscillator': 50.0,
            'mfi_14': 50.0, 'obv': 0.0, 'cmf_20': 0.0,
            'vwap': 0.0,
            'kc_upper': 0.0, 'kc_middle': 0.0, 'kc_lower': 0.0,
            'momentum_10': 0.0, 'volatility': 0.0,
            'trend_direction': 'UNKNOWN', 'candle_patterns': []
        }


# ============================================================
# STRATEGY ENGINE
# ============================================================

class StrategyEngine:
    STRATEGIES = {
        'longterm': {
            'timeframes': ['H4', 'D1'],
            'min_confidence': 0.65,
            'risk_pct': 1.0,
            'sl_atr_mult': 3.0,
            'tp_atr_mult': 6.0,
            'max_trades': 3,
            'description': 'Long-term trend following'
        },
        'scalping': {
            'timeframes': ['M1', 'M5'],
            'min_confidence': 0.6,
            'risk_pct': 0.5,
            'sl_atr_mult': 1.5,
            'tp_atr_mult': 2.0,
            'max_trades': 10,
            'description': 'Short-term scalping'
        },
        'hft': {
            'timeframes': ['M1'],
            'min_confidence': 0.55,
            'risk_pct': 0.25,
            'sl_atr_mult': 1.0,
            'tp_atr_mult': 1.5,
            'max_trades': 20,
            'description': 'High frequency trading'
        },
        'regular': {
            'timeframes': ['M15', 'H1'],
            'min_confidence': 0.62,
            'risk_pct': 0.75,
            'sl_atr_mult': 2.0,
            'tp_atr_mult': 4.0,
            'max_trades': 5,
            'description': 'Regular intraday trading'
        }
    }

    def __init__(self):
        self.analyzer = TechnicalAnalyzer()
        self.name = "StrategyEngine"

    def analyze(self, ohlcv_data: dict, strategy: str = 'regular', balance: float = 1000.0) -> dict:
        try:
            if not ohlcv_data:
                return {'error': 'No OHLCV data'}

            params = self.STRATEGIES.get(strategy, self.STRATEGIES['regular'])
            primary_tf = list(ohlcv_data.keys())[0] if ohlcv_data else 'M15'
            ohlcv = ohlcv_data.get(primary_tf, [])

            if not ohlcv:
                return {'error': 'No candle data'}

            indicators = self.analyzer.calculate_indicators(ohlcv)
            signal = self.analyzer.generate_signal(indicators)
            regime = self.analyzer.detect_market_regime(ohlcv)
            sr_levels = self.analyzer.find_support_resistance(ohlcv)

            # Multi-timeframe confirmation
            mtf = self._multi_tf_analysis(ohlcv_data, strategy)

            # Boost confidence with MTF alignment
            if mtf.get('aligned') and signal['direction'] == mtf.get('direction'):
                signal['confidence'] = min(signal['confidence'] * 1.15, 0.99)
                signal['reason'] += ' [MTF aligned]'

            # Lot size
            atr = indicators.get('atr_14', 0.001)
            price = indicators.get('current_price', 1.0)
            sl_distance = atr * params['sl_atr_mult']
            lot = self._calculate_lot_size(balance, params['risk_pct'], sl_distance, price)

            return {
                'signal': signal,
                'indicators': indicators,
                'regime': regime,
                'support_resistance': sr_levels,
                'multi_tf': mtf,
                'strategy': strategy,
                'strategy_params': params,
                'lot_size': lot,
                'balance': balance,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"StrategyEngine.analyze error: {e}")
            return {'error': str(e)}

    def _multi_tf_analysis(self, ohlcv_data: dict, strategy: str) -> dict:
        try:
            signals = {}
            for tf, ohlcv in ohlcv_data.items():
                if not ohlcv or len(ohlcv) < 20:
                    continue
                indicators = self.analyzer.calculate_indicators(ohlcv)
                signal = self.analyzer.generate_signal(indicators)
                signals[tf] = signal['direction']

            directions = [d for d in signals.values() if d != 'WAIT']
            if not directions:
                return {'aligned': False, 'direction': 'WAIT', 'signals': signals}

            buy_count = directions.count('BUY')
            sell_count = directions.count('SELL')

            if buy_count > sell_count and buy_count >= len(directions) * 0.6:
                return {'aligned': True, 'direction': 'BUY', 'signals': signals, 'strength': buy_count / len(directions)}
            elif sell_count > buy_count and sell_count >= len(directions) * 0.6:
                return {'aligned': True, 'direction': 'SELL', 'signals': signals, 'strength': sell_count / len(directions)}
            else:
                return {'aligned': False, 'direction': 'WAIT', 'signals': signals}
        except:
            return {'aligned': False, 'direction': 'WAIT', 'signals': {}}

    def _calculate_lot_size(self, balance: float, risk_pct: float, sl_distance: float, price: float) -> float:
        try:
            if sl_distance <= 0 or price <= 0:
                return 0.01
            risk_amount = balance * risk_pct / 100
            if price > 1000:  # Gold
                pip_value = 1.0
                sl_pips = sl_distance
            elif price > 100:
                pip_value = 1.0
                sl_pips = sl_distance * 100
            else:  # Forex
                pip_value = 10.0
                sl_pips = sl_distance * 10000

            lot = risk_amount / (sl_pips * pip_value)
            lot = max(0.01, min(lot, 10.0))
            return round(lot, 2)
        except:
            return 0.01


# ============================================================
# MT5 MANAGER
# ============================================================

class MT5Manager:
    def __init__(self):
        self.connected = False
        self.account_info = {}
        self._sim_prices = {
            'XAUUSD': 2650.0, 'EURUSD': 1.0850, 'GBPUSD': 1.2650,
            'USDJPY': 149.50, 'BTCUSD': 67000.0, 'ETHUSD': 3500.0,
            'USDCHF': 0.8850, 'AUDUSD': 0.6550, 'USDCAD': 1.3550,
            'NZDUSD': 0.5950, 'GBPJPY': 189.0, 'EURJPY': 162.0
        }
        self._sim_positions = []
        self._sim_balance = 10000.0
        self._price_noise = {}
        self.logger = logging.getLogger('MT5Manager')

    def connect(self, login: int = None, password: str = None,
                server: str = None, path: str = None) -> dict:
        try:
            if MT5_AVAILABLE and login and password and server:
                kwargs = {}
                if path:
                    kwargs['path'] = path
                if not mt5.initialize(**kwargs):
                    raise Exception(f"MT5 initialize failed: {mt5.last_error()}")
                authorized = mt5.login(login=login, password=password, server=server)
                if not authorized:
                    raise Exception(f"MT5 login failed: {mt5.last_error()}")
                acc = mt5.account_info()
                if acc:
                    self.connected = True
                    self.account_info = {
                        'login': acc.login, 'balance': acc.balance,
                        'equity': acc.equity, 'margin': acc.margin,
                        'free_margin': acc.margin_free, 'leverage': acc.leverage,
                        'currency': acc.currency, 'server': acc.server,
                        'company': acc.company, 'name': acc.name,
                        'connected': True, 'mode': 'live'
                    }
                    self.logger.info(f"MT5 connected: {acc.login}")
                    return {'success': True, 'account': self.account_info}
            raise Exception("MT5 not available, using simulation")
        except Exception as e:
            self.logger.info(f"MT5 simulation mode: {e}")
            self.connected = True
            self.account_info = {
                'login': login or 12345678,
                'balance': self._sim_balance,
                'equity': self._sim_balance,
                'margin': 0, 'free_margin': self._sim_balance,
                'leverage': 100, 'currency': 'USD',
                'server': server or 'Demo', 'company': 'GM Trading AI',
                'name': 'Demo Account', 'connected': True, 'mode': 'simulation'
            }
            return {'success': True, 'account': self.account_info, 'mode': 'simulation'}

    def disconnect(self):
        try:
            if MT5_AVAILABLE and self.connected:
                mt5.shutdown()
            self.connected = False
            self.logger.info("MT5 disconnected")
        except Exception as e:
            self.logger.error(f"Disconnect error: {e}")

    def get_price(self, symbol: str = 'XAUUSD') -> dict:
        try:
            if MT5_AVAILABLE and self.connected:
                try:
                    tick = mt5.symbol_info_tick(symbol)
                    if tick:
                        return {
                            'symbol': symbol, 'bid': tick.bid, 'ask': tick.ask,
                            'spread': round((tick.ask - tick.bid) * 10000, 1),
                            'time': datetime.fromtimestamp(tick.time).isoformat()
                        }
                except:
                    pass
            return self._sim_price(symbol)
        except Exception as e:
            self.logger.error(f"get_price error: {e}")
            return self._sim_price(symbol)

    def _sim_price(self, symbol: str) -> dict:
        base = self._sim_prices.get(symbol, 1.0)
        noise = self._price_noise.get(symbol, 0)
        noise += random.uniform(-0.0002, 0.0002)
        noise = max(-0.01, min(0.01, noise))
        self._price_noise[symbol] = noise
        price = base * (1 + noise)
        spread = 0.2 if 'XAU' in symbol else 0.00012
        return {
            'symbol': symbol, 'bid': round(price, 5),
            'ask': round(price + spread, 5),
            'spread': round(spread * 10000, 1),
            'time': datetime.now().isoformat()
        }

    def open_trade(self, symbol: str, direction: str, lot: float,
                   sl: float = None, tp: float = None, comment: str = '') -> dict:
        try:
            if MT5_AVAILABLE and self.connected:
                try:
                    tick = mt5.symbol_info_tick(symbol)
                    if tick:
                        order_type = mt5.ORDER_TYPE_BUY if direction == 'BUY' else mt5.ORDER_TYPE_SELL
                        price = tick.ask if direction == 'BUY' else tick.bid
                        request = {
                            'action': mt5.TRADE_ACTION_DEAL,
                            'symbol': symbol,
                            'volume': lot,
                            'type': order_type,
                            'price': price,
                            'sl': sl or 0.0,
                            'tp': tp or 0.0,
                            'comment': comment,
                            'type_time': mt5.ORDER_TIME_GTC,
                            'type_filling': mt5.ORDER_FILLING_IOC,
                        }
                        result = mt5.order_send(request)
                        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                            return {'success': True, 'ticket': result.order, 'price': result.price}
                except Exception as ex:
                    self.logger.warning(f"MT5 trade error: {ex}")

            # Simulation
            price_data = self._sim_price(symbol)
            price = price_data['ask'] if direction == 'BUY' else price_data['bid']
            ticket = random.randint(100000, 999999)
            position = {
                'ticket': ticket, 'symbol': symbol, 'direction': direction,
                'lot': lot, 'open_price': price, 'sl': sl, 'tp': tp,
                'comment': comment, 'open_time': datetime.now().isoformat(),
                'profit': 0.0, 'current_price': price
            }
            self._sim_positions.append(position)
            return {'success': True, 'ticket': ticket, 'price': price, 'mode': 'simulation'}
        except Exception as e:
            self.logger.error(f"open_trade error: {e}")
            return {'success': False, 'error': str(e)}

    def close_trade(self, ticket: int, lot: float = None) -> dict:
        try:
            if MT5_AVAILABLE and self.connected:
                try:
                    position = mt5.positions_get(ticket=ticket)
                    if position:
                        pos = position[0]
                        order_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
                        tick = mt5.symbol_info_tick(pos.symbol)
                        price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask
                        request = {
                            'action': mt5.TRADE_ACTION_DEAL,
                            'symbol': pos.symbol,
                            'volume': lot or pos.volume,
                            'type': order_type,
                            'position': ticket,
                            'price': price,
                            'type_time': mt5.ORDER_TIME_GTC,
                            'type_filling': mt5.ORDER_FILLING_IOC,
                        }
                        result = mt5.order_send(request)
                        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                            return {'success': True, 'profit': pos.profit}
                except Exception as ex:
                    self.logger.warning(f"MT5 close error: {ex}")

            # Simulation
            for pos in self._sim_positions:
                if pos['ticket'] == ticket:
                    price_data = self._sim_price(pos['symbol'])
                    close_price = price_data['bid'] if pos['direction'] == 'BUY' else price_data['ask']
                    if pos['direction
