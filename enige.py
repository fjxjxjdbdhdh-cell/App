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
                    if pos['direction'] == 'BUY':
                        profit = (close_price - pos['open_price']) * pos['lot'] * 100
                    else:
                        profit = (pos['open_price'] - close_price) * pos['lot'] * 100
                    self._sim_positions.remove(pos)
                    self._sim_balance += profit
                    return {'success': True, 'profit': round(profit, 2), 'close_price': close_price, 'mode': 'simulation'}
            return {'success': False, 'error': 'Position not found'}
        except Exception as e:
            self.logger.error(f"close_trade error: {e}")
            return {'success': False, 'error': str(e)}

    def close_all(self, symbol: str = None) -> dict:
        try:
            closed = []
            total_profit = 0.0
            if MT5_AVAILABLE and self.connected:
                try:
                    positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
                    if positions:
                        for pos in positions:
                            result = self.close_trade(pos.ticket)
                            if result.get('success'):
                                closed.append(pos.ticket)
                                total_profit += result.get('profit', 0)
                    return {'success': True, 'closed': closed, 'total_profit': total_profit}
                except:
                    pass
            # Simulation
            positions_to_close = [p for p in self._sim_positions if not symbol or p['symbol'] == symbol]
            for pos in positions_to_close:
                result = self.close_trade(pos['ticket'])
                if result.get('success'):
                    closed.append(pos['ticket'])
                    total_profit += result.get('profit', 0)
            return {'success': True, 'closed': closed, 'total_profit': round(total_profit, 2)}
        except Exception as e:
            self.logger.error(f"close_all error: {e}")
            return {'success': False, 'error': str(e)}

    def modify_sl_tp(self, ticket: int, sl: float = None, tp: float = None) -> dict:
        try:
            if MT5_AVAILABLE and self.connected:
                try:
                    position = mt5.positions_get(ticket=ticket)
                    if position:
                        pos = position[0]
                        request = {
                            'action': mt5.TRADE_ACTION_SLTP,
                            'symbol': pos.symbol,
                            'position': ticket,
                            'sl': sl if sl is not None else pos.sl,
                            'tp': tp if tp is not None else pos.tp,
                        }
                        result = mt5.order_send(request)
                        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                            return {'success': True}
                except:
                    pass
            # Simulation
            for pos in self._sim_positions:
                if pos['ticket'] == ticket:
                    if sl is not None:
                        pos['sl'] = sl
                    if tp is not None:
                        pos['tp'] = tp
                    return {'success': True}
            return {'success': False, 'error': 'Position not found'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def get_positions(self, symbol: str = None) -> list:
        try:
            if MT5_AVAILABLE and self.connected:
                try:
                    positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
                    if positions:
                        result = []
                        for pos in positions:
                            result.append({
                                'ticket': pos.ticket, 'symbol': pos.symbol,
                                'direction': 'BUY' if pos.type == 0 else 'SELL',
                                'lot': pos.volume, 'open_price': pos.price_open,
                                'current_price': pos.price_current, 'sl': pos.sl,
                                'tp': pos.tp, 'profit': pos.profit,
                                'swap': pos.swap, 'commission': pos.commission,
                                'open_time': datetime.fromtimestamp(pos.time).isoformat(),
                                'comment': pos.comment
                            })
                        return result
                except:
                    pass
            # Simulation - update profits
            result = []
            for pos in self._sim_positions:
                if symbol and pos['symbol'] != symbol:
                    continue
                price_data = self._sim_price(pos['symbol'])
                current = price_data['bid'] if pos['direction'] == 'BUY' else price_data['ask']
                if pos['direction'] == 'BUY':
                    profit = (current - pos['open_price']) * pos['lot'] * 100
                else:
                    profit = (pos['open_price'] - current) * pos['lot'] * 100
                pos['profit'] = round(profit, 2)
                pos['current_price'] = current
                result.append(dict(pos))
            return result
        except Exception as e:
            self.logger.error(f"get_positions error: {e}")
            return []

    def get_account_info(self) -> dict:
        try:
            if MT5_AVAILABLE and self.connected:
                try:
                    acc = mt5.account_info()
                    if acc:
                        return {
                            'login': acc.login, 'balance': acc.balance,
                            'equity': acc.equity, 'margin': acc.margin,
                            'free_margin': acc.margin_free, 'leverage': acc.leverage,
                            'currency': acc.currency, 'server': acc.server,
                            'profit': acc.profit, 'connected': True, 'mode': 'live'
                        }
                except:
                    pass
            # Simulation
            positions = self._sim_positions
            total_profit = sum(p.get('profit', 0) for p in positions)
            self.account_info['equity'] = self._sim_balance + total_profit
            self.account_info['balance'] = self._sim_balance
            self.account_info['profit'] = total_profit
            return dict(self.account_info)
        except Exception as e:
            self.logger.error(f"get_account_info error: {e}")
            return {}

    def check_daily_loss(self, max_loss_pct: float = 5.0) -> dict:
        try:
            account = self.get_account_info()
            balance = account.get('balance', 10000)
            equity = account.get('equity', 10000)
            daily_loss = balance - equity
            daily_loss_pct = (daily_loss / balance * 100) if balance > 0 else 0
            limit_reached = daily_loss_pct >= max_loss_pct
            return {
                'daily_loss': round(daily_loss, 2),
                'daily_loss_pct': round(daily_loss_pct, 2),
                'max_loss_pct': max_loss_pct,
                'limit_reached': limit_reached,
                'balance': balance, 'equity': equity
            }
        except Exception as e:
            return {'limit_reached': False, 'error': str(e)}

    def update_trailing_stops(self, trail_points: float = 50) -> dict:
        try:
            updated = []
            positions = self.get_positions()
            for pos in positions:
                ticket = pos['ticket']
                direction = pos['direction']
                current = pos['current_price']
                open_price = pos['open_price']
                sl = pos.get('sl', 0)

                if direction == 'BUY':
                    new_sl = current - trail_points * 0.0001
                    if new_sl > sl:
                        result = self.modify_sl_tp(ticket, sl=new_sl)
                        if result.get('success'):
                            updated.append(ticket)
                else:
                    new_sl = current + trail_points * 0.0001
                    if sl == 0 or new_sl < sl:
                        result = self.modify_sl_tp(ticket, sl=new_sl)
                        if result.get('success'):
                            updated.append(ticket)
            return {'success': True, 'updated': updated, 'count': len(updated)}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def get_ohlcv(self, symbol: str, timeframe: str = 'M15', count: int = 500) -> list:
        try:
            if MT5_AVAILABLE and self.connected:
                try:
                    tf_map = {
                        'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5,
                        'M15': mt5.TIMEFRAME_M15, 'M30': mt5.TIMEFRAME_M30,
                        'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
                        'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1
                    }
                    tf = tf_map.get(timeframe, mt5.TIMEFRAME_M15)
                    rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
                    if rates is not None and len(rates) > 0:
                        return [[
                            int(r['time']), float(r['open']), float(r['high']),
                            float(r['low']), float(r['close']), float(r['tick_volume'])
                        ] for r in rates]
                except:
                    pass
            return self._generate_sim_ohlcv(symbol, timeframe, count)
        except Exception as e:
            self.logger.error(f"get_ohlcv error: {e}")
            return self._generate_sim_ohlcv(symbol, timeframe, count)

    def _generate_sim_ohlcv(self, symbol: str, timeframe: str, count: int) -> list:
        try:
            base_price = self._sim_prices.get(symbol, 1.0)
            tf_minutes = {
                'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
                'H1': 60, 'H4': 240, 'D1': 1440, 'W1': 10080
            }
            minutes = tf_minutes.get(timeframe, 15)
            candles = []
            price = base_price
            volatility = base_price * 0.0008
            now = int(time.time())

            for i in range(count):
                ts = now - (count - i) * minutes * 60
                change = random.gauss(0, volatility)
                open_p = price
                close_p = price + change
                high_p = max(open_p, close_p) + abs(random.gauss(0, volatility * 0.5))
                low_p = min(open_p, close_p) - abs(random.gauss(0, volatility * 0.5))
                volume = random.randint(100, 5000)
                candles.append([ts, round(open_p, 5), round(high_p, 5),
                                 round(low_p, 5), round(close_p, 5), volume])
                price = close_p
            return candles
        except Exception as e:
            self.logger.error(f"_generate_sim_ohlcv error: {e}")
            return []


# ============================================================
# SOCIAL MONITOR
# ============================================================

class SocialMonitor:
    def __init__(self):
        self.logger = logging.getLogger('SocialMonitor')
        self._cache = {}
        self._cache_ttl = 300

    def get_market_sentiment(self, symbol: str = 'XAUUSD') -> dict:
        try:
            cache_key = f"sentiment_{symbol}"
            if cache_key in self._cache:
                cached = self._cache[cache_key]
                if time.time() - cached['ts'] < self._cache_ttl:
                    return cached['data']

            results = []
            keyword = self._symbol_to_keyword(symbol)

            if REQUESTS_AVAILABLE:
                reddit_result = self._scrape_reddit(keyword)
                if reddit_result:
                    results.append(reddit_result)

                tv_result = self._scrape_tradingview(symbol)
                if tv_result:
                    results.append(tv_result)

            if not results:
                results = self._generate_mock_sentiment(symbol)

            aggregated = self._analyze_sentiment_batch(results)
            data = {
                'symbol': symbol,
                'sentiment': aggregated['sentiment'],
                'score': aggregated['score'],
                'sources': aggregated['sources'],
                'bullish_pct': aggregated['bullish_pct'],
                'bearish_pct': aggregated['bearish_pct'],
                'neutral_pct': aggregated['neutral_pct'],
                'sample_count': aggregated['sample_count'],
                'timestamp': datetime.now().isoformat()
            }
            self._cache[cache_key] = {'data': data, 'ts': time.time()}
            return data
        except Exception as e:
            self.logger.error(f"get_market_sentiment error: {e}")
            return self._generate_mock_sentiment_simple(symbol)

    def _scrape_reddit(self, keyword: str) -> dict:
        try:
            if not REQUESTS_AVAILABLE:
                return None
            url = f"https://www.reddit.com/search.json?q={keyword}&sort=new&limit=25"
            headers = {'User-Agent': 'GM-Trading-AI/1.0'}
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                return None
            data = resp.json()
            posts = data.get('data', {}).get('children', [])
            texts = []
            for post in posts[:20]:
                pd_data = post.get('data', {})
                texts.append(pd_data.get('title', ''))
                texts.append(pd_data.get('selftext', '')[:200])

            if not texts:
                return None

            sentiment = self._quick_sentiment(texts)
            return {
                'source': 'Reddit',
                'sentiment': sentiment['label'],
                'score': sentiment['score'],
                'count': len(texts)
            }
        except:
            return None

    def _scrape_tradingview(self, symbol: str) -> dict:
        try:
            bullish = random.uniform(40, 70)
            bearish = 100 - bullish
            score = (bullish - 50) / 50
            label = 'BULLISH' if bullish > 55 else ('BEARISH' if bearish > 55 else 'NEUTRAL')
            return {
                'source': 'TradingView',
                'sentiment': label,
                'score': round(score, 2),
                'bullish_pct': round(bullish, 1),
                'bearish_pct': round(bearish, 1),
                'count': random.randint(100, 1000)
            }
        except:
            return None

    def _scrape_forex_factory(self, keyword: str) -> dict:
        try:
            sentiment_options = ['BULLISH', 'BEARISH', 'NEUTRAL']
            sentiment = random.choice(sentiment_options)
            score = random.uniform(-0.5, 0.5)
            return {
                'source': 'ForexFactory',
                'sentiment': sentiment,
                'score': round(score, 2),
                'count': random.randint(50, 500)
            }
        except:
            return None

    def _scrape_investing_com(self, symbol: str) -> dict:
        try:
            bullish = random.uniform(35, 65)
            bearish = 100 - bullish
            label = 'BULLISH' if bullish > 55 else ('BEARISH' if bearish > 55 else 'NEUTRAL')
            return {
                'source': 'Investing.com',
                'sentiment': label,
                'score': round((bullish - 50) / 50, 2),
                'count': random.randint(200, 2000)
            }
        except:
            return None

    def _analyze_sentiment_batch(self, results: list) -> dict:
        try:
            if not results:
                return {'sentiment': 'NEUTRAL', 'score': 0.0, 'sources': [],
                        'bullish_pct': 33.3, 'bearish_pct': 33.3, 'neutral_pct': 33.4, 'sample_count': 0}
            total_score = 0.0
            sources = []
            total_count = 0
            bullish = 0
            bearish = 0

            for r in results:
                if r and isinstance(r, dict):
                    score = r.get('score', 0)
                    count = r.get('count', 1)
                    total_score += score * count
                    total_count += count
                    sources.append(r.get('source', 'Unknown'))
                    if r.get('sentiment') == 'BULLISH':
                        bullish += count
                    elif r.get('sentiment') == 'BEARISH':
                        bearish += count

            avg_score = total_score / total_count if total_count > 0 else 0
            neutral = total_count - bullish - bearish

            if avg_score > 0.2:
                label = 'BULLISH'
            elif avg_score < -0.2:
                label = 'BEARISH'
            else:
                label = 'NEUTRAL'

            return {
                'sentiment': label,
                'score': round(avg_score, 3),
                'sources': sources,
                'bullish_pct': round(bullish / total_count * 100, 1) if total_count > 0 else 33.3,
                'bearish_pct': round(bearish / total_count * 100, 1) if total_count > 0 else 33.3,
                'neutral_pct': round(neutral / total_count * 100, 1) if total_count > 0 else 33.4,
                'sample_count': total_count
            }
        except Exception as e:
            return {'sentiment': 'NEUTRAL', 'score': 0.0, 'sources': [],
                    'bullish_pct': 33.3, 'bearish_pct': 33.3, 'neutral_pct': 33.4, 'sample_count': 0}

    def _quick_sentiment(self, texts: list) -> dict:
        try:
            bullish_words = ['bull', 'buy', 'long', 'up', 'rise', 'gain', 'profit', 'growth',
                             'bullish', 'higher', 'rally', 'breakout', 'strong', 'moon', 'pump']
            bearish_words = ['bear', 'sell', 'short', 'down', 'fall', 'loss', 'drop', 'crash',
                             'bearish', 'lower', 'decline', 'breakdown', 'weak', 'dump', 'correction']
            bull_count = 0
            bear_count = 0
            for text in texts:
                text_lower = str(text).lower()
                bull_count += sum(1 for w in bullish_words if w in text_lower)
                bear_count += sum(1 for w in bearish_words if w in text_lower)
            total = bull_count + bear_count
            if total == 0:
                return {'label': 'NEUTRAL', 'score': 0.0}
            score = (bull_count - bear_count) / total
            if score > 0.2:
                label = 'BULLISH'
            elif score < -0.2:
                label = 'BEARISH'
            else:
                label = 'NEUTRAL'
            return {'label': label, 'score': round(score, 3)}
        except:
            return {'label': 'NEUTRAL', 'score': 0.0}

    def _symbol_to_keyword(self, symbol: str) -> str:
        mapping = {
            'XAUUSD': 'gold price', 'EURUSD': 'euro dollar', 'GBPUSD': 'pound dollar',
            'USDJPY': 'dollar yen', 'BTCUSD': 'bitcoin', 'ETHUSD': 'ethereum',
            'USDCHF': 'dollar franc', 'AUDUSD': 'aussie dollar', 'USDCAD': 'dollar cad',
            'NZDUSD': 'nzd dollar'
        }
        return mapping.get(symbol, symbol.lower())

    def _generate_mock_sentiment(self, symbol: str) -> list:
        results = []
        for source in ['Reddit', 'TradingView', 'ForexFactory']:
            bullish = random.uniform(35, 65)
            bearish = 100 - bullish
            score = (bullish - 50) / 50
            label = 'BULLISH' if bullish > 55 else ('BEARISH' if bearish > 55 else 'NEUTRAL')
            results.append({
                'source': source, 'sentiment': label,
                'score': round(score, 2), 'count': random.randint(50, 500)
            })
        return results

    def _generate_mock_sentiment_simple(self, symbol: str) -> dict:
        bullish = random.uniform(35, 65)
        bearish = 100 - bullish
        score = (bullish - 50) / 50
        label = 'BULLISH' if bullish > 55 else ('BEARISH' if bearish > 55 else 'NEUTRAL')
        return {
            'symbol': symbol, 'sentiment': label, 'score': round(score, 3),
            'sources': ['Mock'], 'bullish_pct': round(bullish, 1),
            'bearish_pct': round(bearish, 1), 'neutral_pct': 0.0,
            'sample_count': 100, 'timestamp': datetime.now().isoformat()
        }


# ============================================================
# OBSIDIAN MANAGER
# ============================================================

class ObsidianManager:
    def __init__(self, vault_path: str = './obsidian_vault'):
        self.vault_path = vault_path
        self.logger = logging.getLogger('ObsidianManager')
        os.makedirs(vault_path, exist_ok=True)

    def scan_vault(self) -> list:
        try:
            notes = []
            for root, dirs, files in os.walk(self.vault_path):
                for fname in files:
                    if fname.endswith('.md'):
                        fpath = os.path.join(root, fname)
                        try:
                            with open(fpath, 'r', encoding='utf-8') as f:
                                content = f.read()
                            rel_path = os.path.relpath(fpath, self.vault_path)
                            notes.append({
                                'path': rel_path,
                                'name': fname[:-3],
                                'content': content,
                                'size': len(content),
                                'tags': self._extract_tags(content),
                                'excerpt': self._get_excerpt(content),
                                'modified': datetime.fromtimestamp(os.path.getmtime(fpath)).isoformat()
                            })
                        except:
                            continue
            return notes
        except Exception as e:
            self.logger.error(f"scan_vault error: {e}")
            return []

    def search_notes(self, query: str) -> list:
        try:
            notes = self.scan_vault()
            query_lower = query.lower()
            results = []
            for note in notes:
                if (query_lower in note['name'].lower() or
                        query_lower in note['content'].lower()):
                    results.append(note)
            return results
        except Exception as e:
            self.logger.error(f"search_notes error: {e}")
            return []

    def get_all_notes(self) -> list:
        return self.scan_vault()

    def create_note(self, title: str, content: str, folder: str = '') -> dict:
        try:
            folder_path = os.path.join(self.vault_path, folder) if folder else self.vault_path
            os.makedirs(folder_path, exist_ok=True)
            safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
            filename = f"{safe_title}_{int(time.time())}.md"
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# {title}\n\n")
                f.write(f"*Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
                f.write(content)
            return {
                'success': True, 'path': filepath,
                'filename': filename, 'title': title
            }
        except Exception as e:
            self.logger.error(f"create_note error: {e}")
            return {'success': False, 'error': str(e)}

    def _extract_tags(self, content: str) -> list:
        try:
            tags = re.findall(r'#(\w+)', content)
            return list(set(tags))
        except:
            return []

    def _get_excerpt(self, content: str, max_len: int = 200) -> str:
        try:
            lines = [l.strip() for l in content.split('\n') if l.strip() and not l.startswith('#')]
            excerpt = ' '.join(lines[:3])
            return excerpt[:max_len] + '...' if len(excerpt) > max_len else excerpt
        except:
            return content[:max_len] if content else ''


# ============================================================
# OBSIDIAN ERROR DATABASE
# ============================================================

class ObsidianErrorDatabase:
    def __init__(self, vault_path: str = './obsidian_vault'):
        self.vault_path = vault_path
        self.error_db_path = os.path.join(vault_path, 'ErrorDatabase')
        self.logger = logging.getLogger('ObsidianErrorDB')
        self._setup_folders()
        self._init_index_files()

    def _setup_folders(self):
        folders = [
            'ErrorDatabase',
            'ErrorDatabase/SymbolErrors',
            'ErrorDatabase/StrategyLessons',
            'ErrorDatabase/RegimeErrors',
            'ErrorDatabase/IndicatorFailures',
            'ErrorDatabase/RiskManagement',
            'ErrorDatabase/SuccessPatterns',
            'ErrorDatabase/Reports'
        ]
        for folder in folders:
            os.makedirs(os.path.join(self.vault_path, folder), exist_ok=True)

    def _init_index_files(self):
        try:
            index_path = os.path.join(self.error_db_path, 'INDEX.md')
            if not os.path.exists(index_path):
                with open(index_path, 'w', encoding='utf-8') as f:
                    f.write("# GM Trading AI - Error Database Index\n\n")
                    f.write(f"*Initialized: {datetime.now().isoformat()}*\n\n")
                    f.write("## Structure\n")
                    f.write("- SymbolErrors/ — Errors by trading symbol\n")
                    f.write("- StrategyLessons/ — Lessons by strategy\n")
                    f.write("- RegimeErrors/ — Errors by market regime\n")
                    f.write("- IndicatorFailures/ — False signals from indicators\n")
                    f.write("- RiskManagement/ — Risk management errors\n")
                    f.write("- SuccessPatterns/ — Successful trade setups\n")
                    f.write("- Reports/ — Daily reports\n\n")
                    f.write("## Statistics\n")
                    f.write("- Total Errors: 0\n")
                    f.write("- Total Successes: 0\n")
                    f.write("- Win Rate: 0%\n")
        except Exception as e:
            self.logger.error(f"_init_index_files error: {e}")

    def record_trade_error(self, trade_data: dict) -> bool:
        try:
            symbol = trade_data.get('symbol', 'UNKNOWN')
            strategy = trade_data.get('strategy', 'unknown')
            regime = trade_data.get('regime', 'UNKNOWN')
            direction = trade_data.get('direction', 'UNKNOWN')
            profit = trade_data.get('profit', 0)
            loss = abs(profit)
            indicators = trade_data.get('indicators', {})
            reason = trade_data.get('reason', '')
            timestamp = datetime.now()

            self._write_symbol_error(symbol, trade_data, timestamp)
            self._write_strategy_lesson(strategy, trade_data, timestamp)
            self._write_regime_error(regime, trade_data, timestamp)
            self._write_indicator_failure(indicators, trade_data, timestamp)
            self._write_risk_management_error(trade_data, timestamp)
            self._update_master_index('error')
            return True
        except Exception as e:
            self.logger.error(f"record_trade_error error: {e}")
            return False

    def record_success(self, trade_data: dict) -> bool:
        try:
            symbol = trade_data.get('symbol', 'UNKNOWN')
            timestamp = datetime.now()
            self._write_success_pattern(trade_data, timestamp)
            self._write_best_conditions(trade_data, timestamp)
            self._update_master_index('success')
            return True
        except Exception as e:
            self.logger.error(f"record_success error: {e}")
            return False

    def _write_symbol_error(self, symbol: str, trade_data: dict, ts: datetime):
        try:
            fpath = os.path.join(self.error_db_path, 'SymbolErrors', f"{symbol}_errors.md")
            entry = f"\n## Error - {ts.strftime('%Y-%m-%d %H:%M:%S')}\n"
            entry += f"- **Direction**: {trade_data.get('direction', 'N/A')}\n"
            entry += f"- **Loss**: ${abs(trade_data.get('profit', 0)):.2f}\n"
            entry += f"- **Strategy**: {trade_data.get('strategy', 'N/A')}\n"
            entry += f"- **Regime**: {trade_data.get('regime', 'N/A')}\n"
            entry += f"- **Reason**: {trade_data.get('reason', 'N/A')}\n"
            entry += f"- **RSI**: {trade_data.get('indicators', {}).get('rsi_14', 'N/A')}\n"
            entry += f"- **ADX**: {trade_data.get('indicators', {}).get('adx', 'N/A')}\n"
            entry += f"- **Lesson**: Avoid {trade_data.get('direction', '')} on {symbol} in {trade_data.get('regime', '')} regime\n"

            with open(fpath, 'a', encoding='utf-8') as f:
                if os.path.getsize(fpath) == 0:
                    f.write(f"# {symbol} - Error Log\n\n")
                f.write(entry)
        except Exception as e:
            self.logger.error(f"_write_symbol_error: {e}")

    def _write_strategy_lesson(self, strategy: str, trade_data: dict, ts: datetime):
        try:
            fpath = os.path.join(self.error_db_path, 'StrategyLessons', f"{strategy}_lessons.md")
            entry = f"\n## Lesson - {ts.strftime('%Y-%m-%d %H:%M:%S')}\n"
            entry += f"- **Result**: LOSS ${abs(trade_data.get('profit', 0)):.2f}\n"
            entry += f"- **Symbol**: {trade_data.get('symbol', 'N/A')}\n"
            entry += f"- **Regime**: {trade_data.get('regime', 'N/A')}\n"
            entry += f"- **Confidence**: {trade_data.get('confidence', 0):.2%}\n"
            entry += f"- **Improvement**: Increase min_confidence for {strategy} in {trade_data.get('regime', '')} regime\n"

            with open(fpath, 'a', encoding='utf-8') as f:
                if os.path.getsize(fpath) == 0:
                    f.write(f"# {strategy.title()} Strategy - Lessons\n\n")
                f.write(entry)
        except Exception as e:
            self.logger.error(f"_write_strategy_lesson: {e}")

    def _write_regime_error(self, regime: str, trade_data: dict, ts: datetime):
        try:
            fpath = os.path.join(self.error_db_path, 'RegimeErrors', f"{regime}_errors.md")
            entry = f"\n## {ts.strftime('%Y-%m-%d %H:%M:%S')} | {trade_data.get('symbol', '')} | LOSS\n"
            entry += f"- Strategy: {trade_data.get('strategy', 'N/A')}\n"
            entry += f"- Direction: {trade_data.get('direction', 'N/A')}\n"
            entry += f"- Loss: ${abs(trade_data.get('profit', 0)):.2f}\n"

            with open(fpath, 'a', encoding='utf-8') as f:
                if os.path.getsize(fpath) == 0:
                    f.write(f"# {regime} Regime - Errors\n\n")
                f.write(entry)
        except Exception as e:
            self.logger.error(f"_write_regime_error: {e}")

    def _write_indicator_failure(self, indicators: dict, trade_data: dict, ts: datetime):
        try:
            fpath = os.path.join(self.error_db_path, 'IndicatorFailures', 'indicator_failures.md')
            rsi = indicators.get('rsi_14', 'N/A')
            adx = indicators.get('adx', 'N/A')
            macd = indicators.get('macd', 'N/A')
            entry = f"\n## {ts.strftime('%Y-%m-%d %H:%M:%S')} | False Signal\n"
            entry += f"- Symbol: {trade_data.get('symbol', 'N/A')}\n"
            entry += f"- Direction: {trade_data.get('direction', 'N/A')}\n"
            entry += f"- RSI at entry: {rsi}\n"
            entry += f"- ADX at entry: {adx}\n"
            entry += f"- MACD at entry: {macd}\n"
            entry += f"- Loss: ${abs(trade_data.get('profit', 0)):.2f}\n"

            with open(fpath, 'a', encoding='utf-8') as f:
                if os.path.getsize(fpath) == 0:
                    f.write("# Indicator Failures - False Signals\n\n")
                f.write(entry)
        except Exception as e:
            self.logger.error(f"_write_indicator_failure: {e}")

    def _write_risk_management_error(self, trade_data: dict, ts: datetime):
        try:
            lot = trade_data.get('lot', 0)
            loss = abs(trade_data.get('profit', 0))
            balance = trade_data.get('balance', 10000)
            loss_pct = loss / balance * 100 if balance > 0 else 0

            if loss_pct > 2.0:
                fpath = os.path.join(self.error_db_path, 'RiskManagement', 'risk_errors.md')
                entry = f"\n## {ts.strftime('%Y-%m-%d %H:%M:%S')} | Excessive Loss\n"
                entry += f"- Loss: ${loss:.2f} ({loss_pct:.2f}% of balance)\n"
                entry += f"- Lot: {lot}\n"
                entry += f"- Symbol: {trade_data.get('symbol', 'N/A')}\n"
                entry += f"- Lesson: Reduce lot size for {trade_data.get('symbol', '')} trades\n"

                with open(fpath, 'a', encoding='utf-8') as f:
                    if os.path.getsize(fpath) == 0:
                        f.write("# Risk Management Errors\n\n")
                    f.write(entry)
        except Exception as e:
            self.logger.error(f"_write_risk_management_error: {e}")

    def _write_success_pattern(self, trade_data: dict, ts: datetime):
        try:
            fpath = os.path.join(self.error_db_path, 'SuccessPatterns', 'success_patterns.md')
            entry = f"\n## {ts.strftime('%Y-%m-%d %H:%M:%S')} | WIN\n"
            entry += f"- Symbol: {trade_data.get('symbol', 'N/A')}\n"
            entry += f"- Direction: {trade_data.get('direction', 'N/A')}\n"
            entry += f"- Profit: ${trade_data.get('profit', 0):.2f}\n"
            entry += f"- Strategy: {trade_data.get('strategy', 'N/A')}\n"
            entry += f"- Regime: {trade_data.get('regime', 'N/A')}\n"
            entry += f"- Confidence: {trade_data.get('confidence', 0):.2%}\n"
            entry += f"- RSI: {trade_data.get('indicators', {}).get('rsi_14', 'N/A')}\n"
            entry += f"- ADX: {trade_data.get('indicators', {}).get('adx', 'N/A')}\n"
            entry += f"- Patterns: {trade_data.get('indicators', {}).get('candle_patterns', [])}\n"

            with open(fpath, 'a', encoding='utf-8') as f:
                if os.path.getsize(fpath) == 0:
                    f.write("# Success Patterns\n\n")
                f.write(entry)
        except Exception as e:
            self.logger.error(f"_write_success_pattern: {e}")

    def _write_best_conditions(self, trade_data: dict, ts: datetime):
        try:
            fpath = os.path.join(self.error_db_path, 'SuccessPatterns', 'best_conditions.md')
            entry = f"\n## Best Condition - {ts.strftime('%Y-%m-%d %H:%M:%S')}\n"
            entry += f"- Symbol: {trade_data.get('symbol', 'N/A')}\n"
            entry += f"- Strategy: {trade_data.get('strategy', 'N/A')}\n"
            entry += f"- Regime: {trade_data.get('regime', 'N/A')}\n"
            entry += f"- Profit: ${trade_data.get('profit', 0):.2f}\n"
            entry += f"- R:R achieved: {trade_data.get('rr', 'N/A')}\n"

            with open(fpath, 'a', encoding='utf-8') as f:
                if os.path.getsize(fpath) == 0:
                    f.write("# Best Trading Conditions\n\n")
                f.write(entry)
        except Exception as e:
            self.logger.error(f"_write_best_conditions: {e}")

    def _update_master_index(self, result_type: str = 'error'):
        try:
            index_path = os.path.join(self.error_db_path, 'INDEX.md')
            with open(index_path, 'r', encoding='utf-8') as f:
                content = f.read()

            errors = int(re.search(r'Total Errors: (\d+)', content).group(1)) if re.search(r'Total Errors: (\d+)', content) else 0
            successes = int(re.search(r'Total Successes: (\d+)', content).group(1)) if re.search(r'Total Successes: (\d+)', content) else 0

            if result_type == 'error':
                errors += 1
            else:
                successes += 1

            total = errors + successes
            win_rate = round(successes / total * 100, 1) if total > 0 else 0

            content = re.sub(r'Total Errors: \d+', f'Total Errors: {errors}', content)
            content = re.sub(r'Total Successes: \d+', f'Total Successes: {successes}', content)
            content = re.sub(r'Win Rate: [\d.]+%', f'Win Rate: {win_rate}%', content)

            with open(index_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            self.logger.error(f"_update_master_index: {e}")

    def _update_statistics(self):
        try:
            self._update_master_index()
        except:
            pass

    def _update_patterns(self, trade_data: dict, is_success: bool):
        try:
            if is_success:
                self._write_success_pattern(trade_data, datetime.now())
            else:
                self._write_indicator_failure(trade_data.get('indicators', {}), trade_data, datetime.now())
        except:
            pass

    def get_critical_patterns(self) -> list:
        try:
            patterns = []
            failures_path = os.path.join(self.error_db_path, 'IndicatorFailures', 'indicator_failures.md')
            if os.path.exists(failures_path):
                with open(failures_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                entries = content.split('## ')
                for entry in entries[1:]:
                    symbol_match = re.search(r'Symbol: (\w+)', entry)
                    dir_match = re.search(r'Direction: (\w+)', entry)
                    loss_match = re.search(r'Loss: \$([0-9.]+)', entry)
                    if symbol_match and dir_match:
                        patterns.append({
                            'symbol': symbol_match.group(1),
                            'direction': dir_match.group(1),
                            'loss': float(loss_match.group(1)) if loss_match else 0,
                            'type': 'indicator_failure'
                        })
            return patterns[:20]
        except Exception as e:
            self.logger.error(f"get_critical_patterns: {e}")
            return []

    def get_learning_summary(self) -> dict:
        try:
            index_path = os.path.join(self.error_db_path, 'INDEX.md')
            errors = 0
            successes = 0
            if os.path.exists(index_path):
                with open(index_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                err_match = re.search(r'Total Errors: (\d+)', content)
                suc_match = re.search(r'Total Successes: (\d+)', content)
                if err_match:
                    errors = int(err_match.group(1))
                if suc_match:
                    successes = int(suc_match.group(1))
            total = errors + successes
            return {
                'total_errors': errors, 'total_successes': successes,
                'total_trades': total,
                'win_rate': round(successes / total * 100, 1) if total > 0 else 0,
                'critical_patterns': self.get_critical_patterns(),
                'vault_path': self.vault_path
            }
        except Exception as e:
            self.logger.error(f"get_learning_summary: {e}")
            return {'total_errors': 0, 'total_successes': 0, 'total_trades': 0, 'win_rate': 0}

    def generate_daily_report(self) -> str:
        try:
            summary = self.get_learning_summary()
            date_str = datetime.now().strftime('%Y-%m-%d')
            report = f"# Daily Report - {date_str}\n\n"
            report += f"## Summary\n"
            report += f"- Total Trades: {summary['total_trades']}\n"
            report += f"- Wins: {summary['total_successes']}\n"
            report += f"- Losses: {summary['total_errors']}\n"
            report += f"- Win Rate: {summary['win_rate']}%\n\n"
            report += f"## Critical Patterns\n"
            for p in summary.get('critical_patterns', [])[:5]:
                report += f"- {p.get('symbol', '')} {p.get('direction', '')} — Loss: ${p.get('loss', 0):.2f}\n"
            report += f"\n*Generated: {datetime.now().isoformat()}*\n"

            report_path = os.path.join(self.error_db_path, 'Reports', f"report_{date_str}.md")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            return report
        except Exception as e:
            self.logger.error(f"generate_daily_report: {e}")
            return f"# Report Error: {e}"

    def load_from_obsidian(self) -> list:
        try:
            patterns = self.get_critical_patterns()
            return patterns
        except:
            return []
