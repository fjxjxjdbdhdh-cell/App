# ============================================================
# GM TRADER — АВТОМАТИЧЕСКИЙ ТРЕЙДЕР
# AutoTrader | RiskManager | ModelDriftMonitor
# AITester (A/B) | RegimeMemory
# ============================================================

import os
import json
import time
import math
import random
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger("GM_TRADER")

try:
    import numpy as np
    NUMPY_OK = True
except ImportError:
    NUMPY_OK = False
    np = None

try:
    from sqlalchemy.orm import Session
    SQLALCHEMY_OK = True
except ImportError:
    SQLALCHEMY_OK = False


# ============================================================
# RISK MANAGER
# ============================================================

class RiskManager:
    def __init__(self):
        self.max_daily_loss_pct = 0.05
        self.max_daily_trades = 20
        self.max_open_positions = 5
        self.max_lot = 5.0
        self.min_lot = 0.01
        self.max_risk_per_trade = 0.02
        self.kelly_fraction = 0.25
        self.daily_trades = []
        self.daily_pnl = 0.0
        self.last_reset = datetime.utcnow().date()
        self.slippage_threshold = 0.0003
        self.drawdown_halt = 0.15

    def _reset_if_new_day(self):
        today = datetime.utcnow().date()
        if today != self.last_reset:
            self.daily_trades = []
            self.daily_pnl = 0.0
            self.last_reset = today

    def calculate_lot(
        self,
        balance: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        sl_distance: float,
        price: float,
        drawdown_pct: float = 0.0
    ) -> float:
        """Расчёт лота по формуле Келли с учётом просадки"""
        try:
            self._reset_if_new_day()

            if balance <= 0 or sl_distance <= 0 or price <= 0:
                return self.min_lot

            # Формула Келли
            if avg_loss > 0 and win_rate > 0:
                b = avg_win / (avg_loss + 1e-10)
                p = win_rate
                q = 1 - p
                kelly = (b * p - q) / (b + 1e-10)
                kelly = max(0.01, min(kelly, 0.5))
                kelly_risk = kelly * self.kelly_fraction
            else:
                kelly_risk = self.max_risk_per_trade

            # Уменьшение при просадке
            if drawdown_pct > 0.05:
                drawdown_factor = 1.0 - (drawdown_pct - 0.05) * 3
                drawdown_factor = max(0.2, drawdown_factor)
                kelly_risk *= drawdown_factor

            # Расчёт лота
            risk_amount = balance * kelly_risk
            pip_value = 10.0 if price < 10 else (1.0 if price < 100 else 0.1)
            sl_pips = sl_distance / (price * 0.0001 + 1e-10)
            lot = risk_amount / (sl_pips * pip_value + 1e-10)
            lot = max(self.min_lot, min(round(lot, 2), self.max_lot))
            return lot

        except Exception as e:
            logger.error(f"calculate_lot error: {e}")
            return self.min_lot

    def check_daily_limit(self, balance: float) -> dict:
        """Проверка дневного лимита"""
        try:
            self._reset_if_new_day()

            issues = []
            allowed = True

            # Дневной убыток
            if balance > 0 and self.daily_pnl < 0:
                loss_pct = abs(self.daily_pnl) / balance
                if loss_pct >= self.max_daily_loss_pct:
                    issues.append(
                        f"Дневной лимит убытка: {loss_pct*100:.1f}% >= {self.max_daily_loss_pct*100:.1f}%"
                    )
                    allowed = False

            # Количество сделок
            if len(self.daily_trades) >= self.max_daily_trades:
                issues.append(f"Лимит сделок за день: {len(self.daily_trades)}")
                allowed = False

            return {
                "allowed": allowed,
                "issues": issues,
                "daily_pnl": round(self.daily_pnl, 2),
                "daily_trades": len(self.daily_trades),
                "loss_pct": round(abs(self.daily_pnl) / max(balance, 1) * 100, 2)
            }
        except Exception as e:
            logger.error(f"check_daily_limit error: {e}")
            return {"allowed": True, "issues": [], "daily_pnl": 0}

    def check_slippage(self, expected_price: float, actual_price: float) -> dict:
        """Проверка проскальзывания"""
        try:
            if expected_price <= 0:
                return {"acceptable": True, "slippage": 0.0}
            slippage = abs(actual_price - expected_price) / (expected_price + 1e-10)
            acceptable = slippage <= self.slippage_threshold
            return {
                "acceptable": acceptable,
                "slippage": round(slippage, 6),
                "slippage_pips": round(slippage * expected_price * 10000, 2),
                "threshold": self.slippage_threshold
            }
        except Exception as e:
            logger.error(f"check_slippage error: {e}")
            return {"acceptable": True, "slippage": 0.0}

    def calculate_risk_reward(
        self,
        entry: float,
        sl: float,
        tp: float,
        direction: str
    ) -> dict:
        """Расчёт Risk/Reward"""
        try:
            if direction == "BUY":
                risk = entry - sl
                reward = tp - entry
            else:
                risk = sl - entry
                reward = entry - tp

            if risk <= 0:
                return {"rr_ratio": 0.0, "acceptable": False}

            rr = reward / (risk + 1e-10)
            acceptable = rr >= 1.5

            return {
                "rr_ratio": round(rr, 2),
                "risk_pips": round(risk * 10000, 1),
                "reward_pips": round(reward * 10000, 1),
                "acceptable": acceptable
            }
        except Exception as e:
            logger.error(f"calculate_risk_reward error: {e}")
            return {"rr_ratio": 0.0, "acceptable": False}

    def record_trade(self, profit: float):
        """Запись результата сделки"""
        try:
            self._reset_if_new_day()
            self.daily_pnl += profit
            self.daily_trades.append({
                "profit": profit,
                "time": datetime.utcnow().isoformat()
            })
        except Exception as e:
            logger.error(f"record_trade error: {e}")

    def get_stats(self) -> dict:
        """Статистика риска"""
        try:
            self._reset_if_new_day()
            wins = [t for t in self.daily_trades if t["profit"] > 0]
            return {
                "daily_trades": len(self.daily_trades),
                "daily_pnl": round(self.daily_pnl, 2),
                "daily_wins": len(wins),
                "daily_win_rate": round(len(wins) / max(len(self.daily_trades), 1) * 100, 1),
                "max_daily_loss_pct": self.max_daily_loss_pct * 100,
                "max_daily_trades": self.max_daily_trades
            }
        except Exception as e:
            logger.error(f"RiskManager.get_stats error: {e}")
            return {}


# ============================================================
# MODEL DRIFT MONITOR
# ============================================================

class ModelDriftMonitor:
    def __init__(self, window: int = 50):
        self.window = window
        self.baseline_win_rate = None
        self.baseline_avg_pnl = None
        self.baseline_confidence = None
        self.recent_results = []
        self.drift_detected = False
        self.drift_count = 0
        self.last_check = datetime.utcnow()
        self.psi_threshold = 0.2
        self.retrain_threshold = 3

    def set_baseline(self, win_rate: float, avg_pnl: float, avg_confidence: float = 0.6):
        """Установка бейзлайна"""
        try:
            self.baseline_win_rate = win_rate
            self.baseline_avg_pnl = avg_pnl
            self.baseline_confidence = avg_confidence
            self.drift_detected = False
            self.drift_count = 0
            logger.info(
                f"Baseline установлен: WR={win_rate:.2f}, "
                f"AvgPnL={avg_pnl:.2f}, Conf={avg_confidence:.2f}"
            )
        except Exception as e:
            logger.error(f"set_baseline error: {e}")

    def calculate_psi(self, expected: list, actual: list, bins: int = 10) -> float:
        """Расчёт Population Stability Index"""
        try:
            if not expected or not actual or not NUMPY_OK:
                return 0.0

            expected_arr = np.array(expected, dtype=float)
            actual_arr = np.array(actual, dtype=float)

            all_vals = np.concatenate([expected_arr, actual_arr])
            min_val = float(np.min(all_vals))
            max_val = float(np.max(all_vals))

            if min_val == max_val:
                return 0.0

            exp_counts, bin_edges = np.histogram(expected_arr, bins=bins, range=(min_val, max_val))
            act_counts, _ = np.histogram(actual_arr, bins=bins, range=(min_val, max_val))

            exp_pct = (exp_counts + 1) / (len(expected_arr) + bins)
            act_pct = (act_counts + 1) / (len(actual_arr) + bins)

            psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / (exp_pct + 1e-10) + 1e-10)))
            return round(abs(psi), 4)

        except Exception as e:
            logger.error(f"calculate_psi error: {e}")
            return 0.0

    def check_drift(self, recent_win_rate: float, recent_avg_pnl: float) -> dict:
        """Проверка дрейфа модели"""
        try:
            if self.baseline_win_rate is None:
                return {"drift": False, "psi": 0.0, "message": "Baseline not set"}

            wr_diff = abs(recent_win_rate - self.baseline_win_rate)
            pnl_diff = abs(recent_avg_pnl - self.baseline_avg_pnl)

            wr_drift = wr_diff > 0.15
            pnl_drift = pnl_diff > abs(self.baseline_avg_pnl) * 0.5 + 5

            psi_value = 0.0
            if NUMPY_OK:
                exp_data = [self.baseline_win_rate] * self.window
                act_data = [recent_win_rate] * len(self.recent_results)
                if act_data:
                    psi_value = self.calculate_psi(exp_data, act_data)

            drift_detected = wr_drift or pnl_drift or psi_value > self.psi_threshold

            if drift_detected:
                self.drift_count += 1
                self.drift_detected = True
                logger.warning(
                    f"⚠️ Дрейф модели: WR {self.baseline_win_rate:.2f}→{recent_win_rate:.2f}, "
                    f"PSI={psi_value:.3f}"
                )
            else:
                self.drift_detected = False

            return {
                "drift": drift_detected,
                "wr_drift": wr_drift,
                "pnl_drift": pnl_drift,
                "psi": psi_value,
                "drift_count": self.drift_count,
                "message": "Drift detected" if drift_detected else "Model stable"
            }

        except Exception as e:
            logger.error(f"check_drift error: {e}")
            return {"drift": False, "psi": 0.0, "message": str(e)}

    def should_retrain(self) -> bool:
        """Нужно ли переобучение"""
        try:
            return self.drift_count >= self.retrain_threshold
        except Exception:
            return False

    def add_result(self, profit: float, confidence: float):
        """Добавить результат"""
        try:
            self.recent_results.append({
                "profit": profit,
                "confidence": confidence,
                "time": datetime.utcnow().isoformat()
            })
            if len(self.recent_results) > self.window * 2:
                self.recent_results = self.recent_results[-self.window:]
        except Exception as e:
            logger.error(f"add_result error: {e}")


# ============================================================
# A/B ТЕСТИРОВАНИЕ ИИ
# ============================================================

class AITester:
    def __init__(self):
        self.version_a = {"name": "v_current", "weights": {}, "trades": 0, "wins": 0, "pnl": 0.0}
        self.version_b = {"name": "v_experimental", "weights": {}, "trades": 0, "wins": 0, "pnl": 0.0}
        self.active_version = "a"
        self.initialized = False
        self.results_a = []
        self.results_b = []
        self.switch_threshold = 20

    def initialize(self, weights_a: dict, weights_b: dict = None):
        """Инициализация двух версий"""
        try:
            self.version_a["weights"] = weights_a.copy()
            if weights_b:
                self.version_b["weights"] = weights_b.copy()
            else:
                # Мутация весов для версии B
                mutated = {}
                for k, v in weights_a.items():
                    mutation = random.gauss(0, 0.1)
                    mutated[k] = max(0.1, min(5.0, v + mutation))
                self.version_b["weights"] = mutated

            self.initialized = True
            logger.info("A/B тестирование инициализировано")
        except Exception as e:
            logger.error(f"AITester.initialize error: {e}")

    def compare(self, signal: dict) -> dict:
        """Сравнение решений двух версий"""
        try:
            if not self.initialized:
                return signal

            # Версия A
            score_a = signal.get("score", 0)
            weights_a = self.version_a["weights"]
            boost_a = sum(weights_a.values()) / max(len(weights_a), 1)

            # Версия B
            weights_b = self.version_b["weights"]
            boost_b = sum(weights_b.values()) / max(len(weights_b), 1)

            if self.active_version == "a":
                final_confidence = min(0.99, signal.get("confidence", 0.5) * boost_a / 10)
            else:
                final_confidence = min(0.99, signal.get("confidence", 0.5) * boost_b / 10)

            signal["confidence"] = final_confidence
            signal["ab_version"] = self.active_version
            signal["boost_a"] = round(boost_a, 3)
            signal["boost_b"] = round(boost_b, 3)
            return signal

        except Exception as e:
            logger.error(f"AITester.compare error: {e}")
            return signal

    def learn_from_result(self, profit: float, version: str = None):
        """Обучение на результате"""
        try:
            v = version or self.active_version
            is_win = profit > 0

            if v == "a":
                self.version_a["trades"] += 1
                self.version_a["pnl"] += profit
                if is_win:
                    self.version_a["wins"] += 1
                self.results_a.append(profit)
                if len(self.results_a) > 100:
                    self.results_a = self.results_a[-100:]
            else:
                self.version_b["trades"] += 1
                self.version_b["pnl"] += profit
                if is_win:
                    self.version_b["wins"] += 1
                self.results_b.append(profit)
                if len(self.results_b) > 100:
                    self.results_b = self.results_b[-100:]

            # Автопереключение
            total = self.version_a["trades"] + self.version_b["trades"]
            if total > 0 and total % self.switch_threshold == 0:
                self._auto_switch()

        except Exception as e:
            logger.error(f"AITester.learn_from_result error: {e}")

    def _auto_switch(self):
        """Автоматическое переключение на лучшую версию"""
        try:
            wr_a = self.version_a["wins"] / max(self.version_a["trades"], 1)
            wr_b = self.version_b["wins"] / max(self.version_b["trades"], 1)
            pnl_a = self.version_a["pnl"]
            pnl_b = self.version_b["pnl"]

            score_a = wr_a * 0.6 + (1 if pnl_a > 0 else 0) * 0.4
            score_b = wr_b * 0.6 + (1 if pnl_b > 0 else 0) * 0.4

            if score_b > score_a + 0.05:
                self.active_version = "b"
                logger.info(f"A/B: переключение на B (score_b={score_b:.3f} > score_a={score_a:.3f})")
            else:
                self.active_version = "a"
                logger.info(f"A/B: остаётся A (score_a={score_a:.3f})")

        except Exception as e:
            logger.error(f"_auto_switch error: {e}")

    def switch_version(self, version: str):
        """Ручное переключение версии"""
        if version in ["a", "b"]:
            self.active_version = version
            logger.info(f"A/B: переключено на версию {version}")

    def get_stats(self) -> dict:
        """Статистика A/B тестирования"""
        try:
            wr_a = self.version_a["wins"] / max(self.version_a["trades"], 1) * 100
            wr_b = self.version_b["wins"] / max(self.version_b["trades"], 1) * 100
            return {
                "active_version": self.active_version,
                "version_a": {
                    "name": self.version_a["name"],
                    "trades": self.version_a["trades"],
                    "wins": self.version_a["wins"],
                    "win_rate": round(wr_a, 2),
                    "pnl": round(self.version_a["pnl"], 2)
                },
                "version_b": {
                    "name": self.version_b["name"],
                    "trades": self.version_b["trades"],
                    "wins": self.version_b["wins"],
                    "win_rate": round(wr_b, 2),
                    "pnl": round(self.version_b["pnl"], 2)
                }
            }
        except Exception as e:
            logger.error(f"AITester.get_stats error: {e}")
            return {}


# ============================================================
# ПАМЯТЬ РЕЖИМОВ
# ============================================================

class RegimeMemory:
    def __init__(self):
        self.memory = {
            "STRONG_TREND": {},
            "MODERATE_TREND": {},
            "CONSOLIDATION": {},
            "HIGH_VOLATILITY": {},
            "RANGING": {}
        }
        self.history = []

    def remember(self, regime: str, strategy: str, profit: float):
        """Запомнить результат стратегии в режиме"""
        try:
            if regime not in self.memory:
                self.memory[regime] = {}
            if strategy not in self.memory[regime]:
                self.memory[regime][strategy] = {
                    "trades": 0, "wins": 0,
                    "pnl": 0.0, "win_rate": 0.5
                }
            m = self.memory[regime][strategy]
            m["trades"] += 1
            m["pnl"] += profit
            if profit > 0:
                m["wins"] += 1
            m["win_rate"] = m["wins"] / m["trades"]

            self.history.append({
                "regime": regime,
                "strategy": strategy,
                "profit": profit,
                "time": datetime.utcnow().isoformat()
            })
            if len(self.history) > 1000:
                self.history = self.history[-1000:]

        except Exception as e:
            logger.error(f"RegimeMemory.remember error: {e}")

    def recommend(self, regime: str) -> dict:
        """Рекомендовать стратегию для режима"""
        try:
            if regime not in self.memory or not self.memory[regime]:
                defaults = {
                    "STRONG_TREND": "longterm",
                    "MODERATE_TREND": "regular",
                    "CONSOLIDATION": "scalping",
                    "HIGH_VOLATILITY": "hft",
                    "RANGING": "scalping"
                }
                return {
                    "strategy": defaults.get(regime, "regular"),
                    "confidence": 0.5,
                    "based_on": "default"
                }

            best_strat = self.get_best_strategy(regime)
            strat_data = self.memory[regime].get(best_strat, {})
            return {
                "strategy": best_strat,
                "confidence": strat_data.get("win_rate", 0.5),
                "trades": strat_data.get("trades", 0),
                "pnl": round(strat_data.get("pnl", 0), 2),
                "based_on": "historical"
            }
        except Exception as e:
            logger.error(f"RegimeMemory.recommend error: {e}")
            return {"strategy": "regular", "confidence": 0.5}

    def get_best_strategy(self, regime: str) -> str:
        """Лучшая стратегия для режима"""
        try:
            if regime not in self.memory or not self.memory[regime]:
                return "regular"

            best = None
            best_score = -999.0
            for strategy, data in self.memory[regime].items():
                if data["trades"] < 3:
                    continue
                score = data["win_rate"] * 0.6 + min(data["pnl"] / 100, 1.0) * 0.4
                if score > best_score:
                    best_score = score
                    best = strategy

            return best or "regular"
        except Exception as e:
            logger.error(f"get_best_strategy error: {e}")
            return "regular"

    def get_all_recommendations(self) -> dict:
        """Рекомендации для всех режимов"""
        try:
            result = {}
            for regime in self.memory:
                result[regime] = self.recommend(regime)
            return result
        except Exception as e:
            logger.error(f"get_all_recommendations error: {e}")
            return {}


# ============================================================
# ГЛАВНЫЙ АВТОТРЕЙДЕР
# ============================================================

class AutoTrader:
    def __init__(self, user_id: int, db_factory, brain=None):
        self.user_id = user_id
        self.db_factory = db_factory
        self.brain = brain

        self._running = False
        self._thread = None
        self._loop = None
        self._stop_event = threading.Event()

        # Настройки
        self.symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
        self.strategy = "regular"
        self.timeframe = "H1"
        self.risk_percent = 0.01
        self.max_trades = 3
        self.min_confidence = 0.65
        self.use_ai = True
        self.interval_seconds = 60

        # Компоненты
        self.risk_manager = RiskManager()
        self.drift_monitor = ModelDriftMonitor()
        self.ab_tester = AITester()
        self.regime_memory = RegimeMemory()

        # Кэш OHLCV
        self._ohlcv_cache = {}
        self._ohlcv_cache_time = {}
        self._ohlcv_ttl = 300

        # Статистика
        self.cycle_count = 0
        self.trades_opened = 0
        self.trades_closed = 0
        self.total_pnl = 0.0
        self.start_time = None
        self.last_cycle_time = None
        self.last_error = None
        self.trade_log = []

        # Состояние
        self.state = {
            "status": "stopped",
            "current_symbol": None,
            "last_signal": None,
            "positions_count": 0,
            "balance": 0.0,
            "equity": 0.0
        }

        logger.info(f"AutoTrader инициализирован для user_id={user_id}")

    async def start(self):
        """Запуск автотрейдера"""
        try:
            if self._running:
                logger.warning("AutoTrader уже запущен")
                return

            self._running = True
            self._stop_event.clear()
            self.start_time = datetime.utcnow()
            self.state["status"] = "running"

            # Загрузка настроек из БД
            self._load_settings()

            # Инициализация A/B тестирования
            if self.brain:
                self.ab_tester.initialize(self.brain.weights)

            # Запуск в фоновом потоке
            self._thread = threading.Thread(
                target=self._run_sync_loop,
                daemon=True,
                name=f"AutoTrader_user{self.user_id}"
            )
            self._thread.start()

            logger.info(
                f"✅ AutoTrader запущен: symbols={self.symbols}, "
                f"strategy={self.strategy}, TF={self.timeframe}"
            )
        except Exception as e:
            logger.error(f"AutoTrader.start error: {e}")
            self._running = False

    def stop(self):
        """Остановка автотрейдера"""
        try:
            self._running = False
            self._stop_event.set()
            self.state["status"] = "stopped"

            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=5)

            logger.info("AutoTrader остановлен")
        except Exception as e:
            logger.error(f"AutoTrader.stop error: {e}")

    def is_running(self) -> bool:
        """Проверка состояния"""
        return self._running

    def _run_sync_loop(self):
        """Синхронный торговый цикл"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._trading_loop())
        except Exception as e:
            logger.error(f"_run_sync_loop error: {e}")
        finally:
            self._running = False
            self.state["status"] = "stopped"

    async def _trading_loop(self):
        """Главный торговый цикл"""
        logger.info(f"🔄 Торговый цикл запущен (interval={self.interval_seconds}s)")
        while self._running and not self._stop_event.is_set():
            try:
                self.cycle_count += 1
                self.last_cycle_time = datetime.utcnow()

                await self._analyze_and_trade()
                self._update_state()

                # Проверка дрейфа модели
                if self.cycle_count % 10 == 0:
                    self._check_model_drift()

                # Пауза
                for _ in range(self.interval_seconds):
                    if not self._running or self._stop_event.is_set():
                        break
                    await asyncio.sleep(1)

            except Exception as e:
                self.last_error = str(e)
                logger.error(f"Trading loop cycle error: {e}")
                await asyncio.sleep(10)

    async def _analyze_and_trade(self):
        """Анализ и открытие сделки"""
        try:
            balance = self._get_balance()
            if balance <= 0:
                return

            # Проверка дневного лимита
            daily_check = self.risk_manager.check_daily_limit(balance)
            if not daily_check["allowed"]:
                logger.warning(f"Дневной лимит: {daily_check['issues']}")
                self._notify_user("⚠️ Дневной лимит достигнут", "warning")
                return

            # Текущие позиции
            open_trades = self._get_open_trades_count()
            if open_trades >= self.max_trades:
                logger.debug(f"Макс позиции: {open_trades}/{self.max_trades}")
                return

            # Анализ символов
            for symbol in self.symbols:
                if not self._running:
                    break
                try:
                    await self._process_symbol(symbol, balance, open_trades)
                    await asyncio.sleep(0.5)
                except Exception as e:
                    logger.error(f"_process_symbol {symbol}: {e}")

        except Exception as e:
            logger.error(f"_analyze_and_trade error: {e}")

    async def _process_symbol(self, symbol: str, balance: float, open_trades: int):
        """Обработка одного символа"""
        try:
            self.state["current_symbol"] = symbol

            # Получение OHLCV
            ohlcv = self._get_ohlcv_cached(symbol, self.timeframe)
            if not ohlcv or len(ohlcv) < 50:
                return

            # Анализ
            try:
                from gm_engine import TechnicalAnalyzer, StrategyEngine
                analyzer = TechnicalAnalyzer()
                strategy_engine = StrategyEngine()
                indicators = analyzer.calculate_indicators(ohlcv)
                regime = analyzer.detect_market_regime(ohlcv)
                signal = strategy_engine.analyze(symbol, ohlcv, self.strategy, balance)
            except ImportError:
                signal = self._simulate_signal(symbol)
                regime = random.choice(["RANGING", "MODERATE_TREND", "CONSOLIDATION"])
                indicators = {}

            direction = signal.get("direction", "WAIT")
            confidence = signal.get("confidence", 0.0)

            # AI буст
            if self.use_ai and self.brain:
                try:
                    boost = self.brain.get_knowledge_boost()
                    confidence = min(0.99, confidence * boost)
                except Exception:
                    pass

            # Рекомендация режима
            regime_rec = self.regime_memory.recommend(regime)
            rec_strategy = regime_rec.get("strategy", self.strategy)

            self.state["last_signal"] = {
                "symbol": symbol,
                "direction": direction,
                "confidence": confidence,
                "regime": regime,
                "recommended_strategy": rec_strategy
            }

            # Проверка минимальной уверенности
            if direction == "WAIT" or confidence < self.min_confidence:
                return

            # Risk/Reward проверка
            entry = signal.get("entry", 0) or signal.get("indicators", {}).get("current_price", 0)
            sl = signal.get("sl", 0)
            tp = signal.get("tp", 0)

            if entry > 0 and sl > 0 and tp > 0:
                rr = self.risk_manager.calculate_risk_reward(entry, sl, tp, direction)
                if not rr.get("acceptable", True):
                    logger.debug(f"R:R неприемлемый для {symbol}: {rr['rr_ratio']:.2f}")
                    return

            # Расчёт лота
            sl_dist = abs(entry - sl) if entry > 0 and sl > 0 else entry * 0.002
            win_rate = self.brain.winning_trades / max(self.brain.total_trades, 1) if self.brain else 0.5
            lot = self.risk_manager.calculate_lot(
                balance=balance,
                win_rate=win_rate,
                avg_win=15.0,
                avg_loss=10.0,
                sl_distance=sl_dist,
                price=entry
            )

            # Открытие сделки
            result = self._open_trade_in_db(
                symbol=symbol,
                direction=direction,
                lot=lot,
                entry=entry,
                sl=sl,
                tp=tp,
                strategy=rec_strategy,
                confidence=confidence,
                regime=regime
            )

            if result:
                self.trades_opened += 1
                self.risk_manager.daily_trades.append({"profit": 0, "time": datetime.utcnow().isoformat()})
                self._log_event(
                    f"OPENED {direction} {symbol} lot={lot} @ {entry:.5f} "
                    f"conf={confidence:.2f} regime={regime}",
                    "success"
                )
                self._notify_user(
                    f"📈 Открыта {direction} {symbol} | conf={confidence:.0%}",
                    "info"
                )

        except Exception as e:
            logger.error(f"_process_symbol error for {symbol}: {e}")

    def _get_ohlcv_all_timeframes(self, symbol: str) -> dict:
        """Получение данных со всех таймфреймов с кэшем"""
        result = {}
        timeframes = ["M15", "H1", "H4"]
        for tf in timeframes:
            result[tf] = self._get_ohlcv_cached(symbol, tf)
        return result

    def _get_ohlcv_cached(self, symbol: str, timeframe: str) -> list:
        """OHLCV с кэшем"""
        try:
            cache_key = f"{symbol}_{timeframe}"
            now = time.time()
            if (cache_key in self._ohlcv_cache and
                    now - self._ohlcv_cache_time.get(cache_key, 0) < self._ohlcv_ttl):
                return self._ohlcv_cache[cache_key]

            try:
                from gm_engine import MT5Manager
                manager = MT5Manager()
                data = manager.get_ohlcv(symbol, timeframe, 200)
            except Exception:
                data = self._generate_sim_ohlcv(symbol, 200)

            if data:
                self._ohlcv_cache[cache_key] = data
                self._ohlcv_cache_time[cache_key] = now

            return data or []
        except Exception as e:
            logger.error(f"_get_ohlcv_cached error: {e}")
            return []

    def _generate_sim_ohlcv(self, symbol: str, count: int = 200) -> list:
        """Генерация симулированных OHLCV данных"""
        try:
            base_prices = {
                "EURUSD": 1.0850, "GBPUSD": 1.2650, "USDJPY": 149.50,
                "XAUUSD": 2050.0, "BTCUSD": 43000.0, "ETHUSD": 2500.0
            }
            base = base_prices.get(symbol, 1.0)
            now = int(time.time())
            result = []
            price = base
            for i in range(count):
                t = now - (count - i) * 3600
                change = random.gauss(0, base * 0.001)
                o = price
                c = max(o + change, 0.0001)
                h = max(o, c) + abs(random.gauss(0, base * 0.0005))
                l = min(o, c) - abs(random.gauss(0, base * 0.0005))
                v = random.randint(100, 5000)
                result.append([t, round(o, 5), round(h, 5), round(l, 5), round(c, 5), v])
                price = c
            return result
        except Exception:
            return []

    def _get_balance(self) -> float:
        """Получение баланса из БД"""
        try:
            db = self.db_factory()
            try:
                from sqlalchemy import text
                result = db.execute(
                    text("SELECT SUM(profit) FROM trades WHERE user_id=:uid AND status='closed'"),
                    {"uid": self.user_id}
                ).fetchone()
                pnl = float(result[0] or 0)
                balance = 10000.0 + pnl
                self.state["balance"] = balance
                return balance
            except Exception as e:
                logger.debug(f"_get_balance db error: {e}")
                return 10000.0
            finally:
                db.close()
        except Exception as e:
            logger.error(f"_get_balance error: {e}")
            return 10000.0

    def _get_open_trades_count(self) -> int:
        """Количество открытых сделок"""
        try:
            db = self.db_factory()
            try:
                from sqlalchemy import text
                result = db.execute(
                    text("SELECT COUNT(*) FROM trades WHERE user_id=:uid AND status='open'"),
                    {"uid": self.user_id}
                ).fetchone()
                count = int(result[0] or 0)
                self.state["positions_count"] = count
                return count
            except Exception:
                return 0
            finally:
                db.close()
        except Exception:
            return 0

    def _open_trade_in_db(
        self,
        symbol: str,
        direction: str,
        lot: float,
        entry: float,
        sl: float,
        tp: float,
        strategy: str,
        confidence: float,
        regime: str
    ) -> bool:
        """Открытие сделки через БД"""
        try:
            import uuid
            db = self.db_factory()
            try:
                from sqlalchemy import text
                ticket = str(uuid.uuid4())[:8].upper()
                db.execute(text("""
                    INSERT INTO trades
                    (user_id, ticket, symbol, direction, volume, open_price,
                     sl, tp, strategy, status, open_time, signal_confidence,
                     market_regime, ai_decision)
                    VALUES
                    (:uid, :ticket, :sym, :dir, :vol, :price,
                     :sl, :tp, :strat, 'open', :ot, :conf, :regime, :ai)
                """), {
                    "uid": self.user_id,
                    "ticket": ticket,
                    "sym": symbol,
                    "dir": direction,
                    "vol": lot,
                    "price": entry,
                    "sl": sl,
                    "tp": tp,
                    "strat": strategy,
                    "ot": datetime.utcnow(),
                    "conf": confidence,
                    "regime": regime,
                    "ai": json.dumps({
                        "auto_trader": True,
                        "boost": self.brain.get_knowledge_boost() if self.brain else 1.0,
                        "regime": regime
                    })
                })
                db.commit()
                return True
            except Exception as e:
                logger.error(f"_open_trade_in_db error: {e}")
                db.rollback()
                return False
            finally:
                db.close()
        except Exception as e:
            logger.error(f"_open_trade_in_db outer error: {e}")
            return False

    def _simulate_signal(self, symbol: str) -> dict:
        """Симулированный сигн
