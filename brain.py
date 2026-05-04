# ============================================================
# GM AI BRAIN — САМООБУЧАЮЩИЙСЯ НЕЙРОННЫЙ МОЗГ
# IQ: 50 → 500 | Генетическая эволюция | LSTM прогноз
# ============================================================

import os
import re
import json
import time
import math
import random
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger("GM_AI_BRAIN")

try:
    import numpy as np
    NUMPY_OK = True
except ImportError:
    NUMPY_OK = False

try:
    import requests
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

try:
    from bs4 import BeautifulSoup
    BS4_OK = True
except ImportError:
    BS4_OK = False

try:
    import openai
    OPENAI_OK = True
except ImportError:
    OPENAI_OK = False

try:
    from sqlalchemy.orm import Session
    SQLALCHEMY_OK = True
except ImportError:
    SQLALCHEMY_OK = False


# ============================================================
# БАЗОВЫЙ AI МОЗГ
# ============================================================

class GMAIBrain:
    def __init__(self, db_session_factory):
        self.db_factory = db_session_factory
        self.generation = 1
        self.iq = 50.0
        self.max_iq = 500.0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.lessons_learned = 0
        self.evolution_count = 0

        # Веса принятия решений
        self.weights = {
            "rsi": 1.0,
            "macd": 1.0,
            "ema": 1.0,
            "bollinger": 0.8,
            "adx": 1.2,
            "stochastic": 0.9,
            "volume": 0.7,
            "pattern": 1.5,
            "sentiment": 0.6,
            "regime": 1.3,
            "support_resistance": 1.4,
            "multi_timeframe": 1.6,
            "news": 0.8,
            "momentum": 1.1,
            "volatility": 0.9
        }

        # Производительность по символам
        self.symbol_performance = {}

        # Производительность по стратегиям
        self.strategy_performance = {}

        # Знания о рыночных режимах
        self.regime_knowledge = {
            "STRONG_TREND": {"best_strategy": "longterm", "win_rate": 0.65, "trades": 0},
            "MODERATE_TREND": {"best_strategy": "regular", "win_rate": 0.58, "trades": 0},
            "CONSOLIDATION": {"best_strategy": "scalping", "win_rate": 0.52, "trades": 0},
            "HIGH_VOLATILITY": {"best_strategy": "hft", "win_rate": 0.48, "trades": 0},
            "RANGING": {"best_strategy": "scalping", "win_rate": 0.50, "trades": 0}
        }

        # Паттерны ошибок
        self.error_patterns = []

        # Успешные паттерны
        self.success_patterns = []

        # Социальные знания
        self.social_knowledge = []

        # Знания из Obsidian
        self.obsidian_knowledge = []

        # Веб знания
        self.web_knowledge = []

        # История IQ
        self.iq_history = [{"iq": 50.0, "time": datetime.utcnow().isoformat()}]

        # Кэш прогнозов
        self._forecast_cache = {}

        # Оценки от пользователя
        self.user_ratings = []

        # OpenAI API ключ
        self.openai_key = os.getenv("OPENAI_API_KEY", "")

        # Загрузка состояния из БД
        self._load_state()

        logger.info(f"GMAIBrain запущен | IQ={self.iq:.1f} | Generation={self.generation}")

    def _load_state(self):
        """Загрузка состояния из БД"""
        try:
            db = self.db_factory()
            try:
                from sqlalchemy import text
                result = db.execute(
                    text("SELECT * FROM ai_states LIMIT 1")
                ).fetchone()
                if result:
                    self.generation = result.generation or 1
                    self.iq = float(result.iq or 50.0)
                    self.total_trades = result.total_trades or 0
                    self.winning_trades = result.winning_trades or 0
                    self.total_pnl = float(result.total_pnl or 0)
                    self.lessons_learned = result.lessons_learned or 0
                    self.evolution_count = result.evolution_count or 0
                    if result.weights:
                        try:
                            loaded = json.loads(result.weights)
                            self.weights.update(loaded)
                        except Exception:
                            pass
            except Exception as e:
                logger.debug(f"_load_state db read: {e}")
            finally:
                db.close()
        except Exception as e:
            logger.debug(f"_load_state error: {e}")

    def _save_state(self):
        """Сохранение состояния в БД"""
        try:
            db = self.db_factory()
            try:
                from sqlalchemy import text
                existing = db.execute(
                    text("SELECT id FROM ai_states LIMIT 1")
                ).fetchone()
                weights_json = json.dumps(self.weights)
                if existing:
                    db.execute(text("""
                        UPDATE ai_states SET
                            generation=:gen, iq=:iq, total_trades=:tt,
                            winning_trades=:wt, total_pnl=:pnl,
                            lessons_learned=:ll, evolution_count=:ec,
                            weights=:ww, updated_at=:ua
                        WHERE id=:id
                    """), {
                        "gen": self.generation, "iq": self.iq,
                        "tt": self.total_trades, "wt": self.winning_trades,
                        "pnl": self.total_pnl, "ll": self.lessons_learned,
                        "ec": self.evolution_count, "ww": weights_json,
                        "ua": datetime.utcnow(), "id": existing.id
                    })
                db.commit()
            except Exception as e:
                logger.debug(f"_save_state db write: {e}")
                db.rollback()
            finally:
                db.close()
        except Exception as e:
            logger.debug(f"_save_state error: {e}")

    def get_knowledge_boost(self) -> float:
        """Буст уверенности от накопленных знаний"""
        try:
            base_boost = 1.0
            # Буст от IQ
            iq_boost = (self.iq - 50) / 450 * 0.5
            # Буст от выигрышных сделок
            if self.total_trades > 0:
                win_rate = self.winning_trades / self.total_trades
                wr_boost = (win_rate - 0.5) * 0.3
            else:
                wr_boost = 0.0
            # Буст от уроков
            lesson_boost = min(self.lessons_learned * 0.001, 0.2)
            # Буст от поколения
            gen_boost = min((self.generation - 1) * 0.05, 0.25)
            total_boost = base_boost + iq_boost + wr_boost + lesson_boost + gen_boost
            return max(0.8, min(total_boost, 2.0))
        except Exception:
            return 1.0

    def learn_from_trade(self, trade_data: dict):
        """ГЛАВНАЯ ФУНКЦИЯ — обучение на каждой сделке"""
        try:
            profit = trade_data.get("profit", 0)
            symbol = trade_data.get("symbol", "EURUSD")
            direction = trade_data.get("direction", "BUY")
            strategy = trade_data.get("strategy", "regular")
            confidence = trade_data.get("signal_confidence", 0.5)
            regime = trade_data.get("market_regime", "RANGING")
            open_price = trade_data.get("open_price", 0)
            close_price = trade_data.get("close_price", 0)
            user_feedback = trade_data.get("user_feedback", None)

            self.total_trades += 1
            self.total_pnl += profit
            is_win = profit > 0
            if is_win:
                self.winning_trades += 1

            # 1. Анализ сделки
            lessons = self._analyze_trade(trade_data)

            # 2. Корректировка весов
            self._adjust_weights(trade_data, is_win)

            # 3. Обновление производительности символа
            self._update_symbol_performance(symbol, profit, is_win)

            # 4. Обновление производительности стратегии
            self._update_strategy_performance(strategy, profit, is_win)

            # 5. Обучение на рыночных режимах
            if regime in self.regime_knowledge:
                rk = self.regime_knowledge[regime]
                rk["trades"] += 1
                old_wr = rk["win_rate"]
                rk["win_rate"] = old_wr * 0.95 + (1.0 if is_win else 0.0) * 0.05

            # 6. Сохранение урока
            for lesson in lessons:
                self._save_lesson(lesson, symbol, strategy, profit)

            # 7. Обновление IQ
            self._update_iq(is_win, profit, confidence)

            # 8. Рефлексивное обучение
            self._reflective_learning(trade_data, is_win, lessons)

            # 9. Обучение от пользователя
            if user_feedback:
                self._learn_from_user_feedback(user_feedback, trade_data)

            # 10. Эволюция при накоплении опыта
            if self.total_trades % 20 == 0:
                self._evolve()

            # 11. Сохранение
            if self.total_trades % 5 == 0:
                self._save_state()

            logger.info(
                f"✅ Learned from trade | {symbol} {direction} | "
                f"PnL={profit:.2f} | IQ={self.iq:.1f} | "
                f"WinRate={self.winning_trades/self.total_trades*100:.1f}%"
            )

        except Exception as e:
            logger.error(f"learn_from_trade error: {e}")

    def _analyze_trade(self, trade_data: dict) -> list:
        """Анализ сделки и извлечение уроков"""
        lessons = []
        try:
            profit = trade_data.get("profit", 0)
            symbol = trade_data.get("symbol", "")
            direction = trade_data.get("direction", "BUY")
            strategy = trade_data.get("strategy", "regular")
            indicators = trade_data.get("indicators", {})
            regime = trade_data.get("market_regime", "RANGING")
            is_win = profit > 0

            if not is_win:
                # Обучение на ошибках
                lesson = {
                    "type": "error",
                    "rule": f"НЕ ОТКРЫВАТЬ {direction} на {symbol} при текущих условиях",
                    "conditions": {
                        "strategy": strategy,
                        "regime": regime,
                        "rsi": indicators.get("rsi", 50),
                        "adx": indicators.get("adx", 25)
                    },
                    "penalty": abs(profit),
                    "timestamp": datetime.utcnow().isoformat()
                }
                lessons.append(lesson)
                self.error_patterns.append(lesson)
                if len(self.error_patterns) > 200:
                    self.error_patterns = self.error_patterns[-200:]

                # Правила "не делать"
                if indicators.get("rsi", 50) > 70 and direction == "BUY":
                    lessons.append({
                        "type": "rule",
                        "rule": "Не покупать когда RSI > 70 (перекуплен)",
                        "weight_adjustment": {"rsi": -0.05}
                    })
                if indicators.get("rsi", 50) < 30 and direction == "SELL":
                    lessons.append({
                        "type": "rule",
                        "rule": "Не продавать когда RSI < 30 (перепродан)",
                        "weight_adjustment": {"rsi": -0.05}
                    })
                if regime == "CONSOLIDATION" and strategy == "longterm":
                    lessons.append({
                        "type": "rule",
                        "rule": "Longterm не подходит для консолидации",
                        "weight_adjustment": {"regime": 0.1}
                    })
            else:
                # Обучение на успехах
                lesson = {
                    "type": "success",
                    "rule": f"ПОВТОРИТЬ {direction} на {symbol} при похожих условиях",
                    "conditions": {
                        "strategy": strategy,
                        "regime": regime,
                        "rsi": indicators.get("rsi", 50),
                        "adx": indicators.get("adx", 25)
                    },
                    "reward": profit,
                    "timestamp": datetime.utcnow().isoformat()
                }
                lessons.append(lesson)
                self.success_patterns.append(lesson)
                if len(self.success_patterns) > 200:
                    self.success_patterns = self.success_patterns[-200:]

        except Exception as e:
            logger.error(f"_analyze_trade error: {e}")

        return lessons

    def _adjust_weights(self, trade_data: dict, is_win: bool):
        """Корректировка весов системы принятия решений"""
        try:
            lr = 0.02 if is_win else -0.03
            multiplier = 1 if is_win else -1
            indicators = trade_data.get("indicators", {})

            # RSI
            rsi = indicators.get("rsi", 50)
            if abs(rsi - 50) > 15:
                self.weights["rsi"] += lr * multiplier * 0.5
            # MACD
            macd = indicators.get("macd", 0)
            if abs(macd) > 0.0001:
                self.weights["macd"] += lr * multiplier * 0.4
            # ADX trend
            adx = indicators.get("adx", 25)
            if adx > 30:
                self.weights["adx"] += lr * multiplier * 0.6
            # Pattern
            patterns = trade_data.get("patterns", [])
            if patterns:
                self.weights["pattern"] += lr * multiplier * 0.8
            # Sentiment
            sentiment = trade_data.get("sentiment", {})
            if sentiment:
                self.weights["sentiment"] += lr * multiplier * 0.3
            # Нормализация
            for key in self.weights:
                self.weights[key] = max(0.1, min(self.weights[key], 5.0))
        except Exception as e:
            logger.error(f"_adjust_weights error: {e}")

    def _update_symbol_performance(self, symbol: str, profit: float, is_win: bool):
        """Обновление статистики по символу"""
        try:
            if symbol not in self.symbol_performance:
                self.symbol_performance[symbol] = {
                    "trades": 0, "wins": 0, "pnl": 0.0,
                    "win_rate": 0.5, "avg_profit": 0.0
                }
            sp = self.symbol_performance[symbol]
            sp["trades"] += 1
            sp["pnl"] += profit
            if is_win:
                sp["wins"] += 1
            sp["win_rate"] = sp["wins"] / sp["trades"]
            sp["avg_profit"] = sp["pnl"] / sp["trades"]
        except Exception as e:
            logger.error(f"_update_symbol_performance error: {e}")

    def _update_strategy_performance(self, strategy: str, profit: float, is_win: bool):
        """Обновление статистики по стратегии"""
        try:
            if strategy not in self.strategy_performance:
                self.strategy_performance[strategy] = {
                    "trades": 0, "wins": 0, "pnl": 0.0,
                    "win_rate": 0.5, "sharpe": 0.0
                }
            sp = self.strategy_performance[strategy]
            sp["trades"] += 1
            sp["pnl"] += profit
            if is_win:
                sp["wins"] += 1
            sp["win_rate"] = sp["wins"] / sp["trades"]
        except Exception as e:
            logger.error(f"_update_strategy_performance error: {e}")

    def _save_lesson(self, lesson: dict, symbol: str, strategy: str, profit: float):
        """Сохранение урока в БД"""
        try:
            self.lessons_learned += 1
            db = self.db_factory()
            try:
                from sqlalchemy import text
                db.execute(text("""
                    INSERT INTO ai_knowledge
                    (user_id, category, content, confidence, usage_count, success_rate)
                    VALUES (1, :cat, :cont, :conf, 1, :sr)
                """), {
                    "cat": lesson.get("type", "lesson"),
                    "cont": json.dumps(lesson, ensure_ascii=False),
                    "conf": 0.8 if profit > 0 else 0.3,
                    "sr": 1.0 if profit > 0 else 0.0
                })
                db.commit()
            except Exception:
                db.rollback()
            finally:
                db.close()
        except Exception as e:
            logger.debug(f"_save_lesson error: {e}")

    def _update_iq(self, is_win: bool, profit: float, confidence: float):
        """Обновление IQ"""
        try:
            # Базовое изменение
            if is_win:
                delta = 0.5 + abs(profit) * 0.01 + confidence * 0.3
            else:
                delta = -(0.3 + abs(profit) * 0.005)

            # Бонус за правильный прогноз
            if confidence > 0.7 and is_win:
                delta += 1.0
            elif confidence < 0.4 and is_win:
                delta += 0.2

            # Штраф за самонадеянность
            if confidence > 0.8 and not is_win:
                delta -= 1.5

            # IQ накапливается
            lesson_bonus = self.lessons_learned * 0.001
            self.iq = max(1.0, min(self.max_iq, self.iq + delta + lesson_bonus))

            # Запись в историю
            if len(self.iq_history) < 1000:
                self.iq_history.append({
                    "iq": round(self.iq, 2),
                    "time": datetime.utcnow().isoformat()
                })

        except Exception as e:
            logger.error(f"_update_iq error: {e}")

    def _evolve(self):
        """Эволюция стратегий (генетический алгоритм)"""
        try:
            self.evolution_count += 1
            self.generation += 1

            logger.info(f"🧬 Эволюция #{self.evolution_count} | Поколение {self.generation}")

            # Мутация весов
            mutation_rate = 0.1 / self.generation
            for key in self.weights:
                if random.random() < mutation_rate:
                    mutation = random.gauss(0, 0.05)
                    self.weights[key] = max(0.1, min(5.0, self.weights[key] + mutation))

            # Отбор лучших паттернов
            if self.success_patterns:
                top_patterns = sorted(
                    self.success_patterns,
                    key=lambda x: x.get("reward", 0),
                    reverse=True
                )[:50]
                self.success_patterns = top_patterns

            # Усиление весов на основе успешных паттернов
            for pattern in self.success_patterns[:10]:
                conditions = pattern.get("conditions", {})
                if conditions.get("adx", 0) > 30:
                    self.weights["adx"] = min(5.0, self.weights["adx"] * 1.02)
                if abs(conditions.get("rsi", 50) - 50) > 20:
                    self.weights["rsi"] = min(5.0, self.weights["rsi"] * 1.01)

            # Обновление режимных знаний
            for regime, data in self.regime_knowledge.items():
                if data["trades"] > 10:
                    best_strat = self._find_best_strategy_for_regime(regime)
                    if best_strat:
                        data["best_strategy"] = best_strat

            # Бонус IQ за эволюцию
            iq_bonus = 2.0 + self.generation * 0.5
            self.iq = min(self.max_iq, self.iq + iq_bonus)

            self._save_state()
            logger.info(f"✅ Эволюция завершена | IQ={self.iq:.1f} | Gen={self.generation}")

        except Exception as e:
            logger.error(f"_evolve error: {e}")

    def _find_best_strategy_for_regime(self, regime: str) -> Optional[str]:
        """Найти лучшую стратегию для режима"""
        try:
            if not self.strategy_performance:
                return None
            best_strat = None
            best_wr = 0.0
            for strat, perf in self.strategy_performance.items():
                if perf["trades"] > 5 and perf["win_rate"] > best_wr:
                    best_wr = perf["win_rate"]
                    best_strat = strat
            return best_strat
        except Exception:
            return None

    def _reflective_learning(self, trade_data: dict, is_win: bool, lessons: list):
        """Рефлексивное обучение — анализ своих решений"""
        try:
            confidence = trade_data.get("signal_confidence", 0.5)
            profit = trade_data.get("profit", 0)
            if is_win and confidence > 0.7:
                # Подтверждение: высокая уверенность + прибыль
                self.weights["multi_timeframe"] = min(5.0, self.weights["multi_timeframe"] * 1.02)
            elif not is_win and confidence > 0.7:
                # Самонадеянность: высокая уверенность + убыток
                for key in self.weights:
                    self.weights[key] *= 0.99
                logger.warning(f"Рефлексия: самонадеянность обнаружена (conf={confidence:.2f}, pnl={profit:.2f})")
            elif is_win and confidence < 0.5:
                # Недооценка: низкая уверенность + прибыль
                self.lessons_learned += 1
                logger.info("Рефлексия: сигнал недооценён")
        except Exception as e:
            logger.error(f"_reflective_learning error: {e}")

    def _learn_from_user_feedback(self, feedback: str, trade_data: dict):
        """Обучение от пользователя"""
        try:
            rating_entry = {
                "feedback": feedback,
                "symbol": trade_data.get("symbol"),
                "direction": trade_data.get("direction"),
                "strategy": trade_data.get("strategy"),
                "timestamp": datetime.utcnow().isoformat()
            }
            self.user_ratings.append(rating_entry)
            if len(self.user_ratings) > 100:
                self.user_ratings = self.user_ratings[-100:]
            if feedback == "positive":
                self.iq = min(self.max_iq, self.iq + 1.0)
                self.weights["sentiment"] = min(5.0, self.weights["sentiment"] * 1.05)
            elif feedback == "negative":
                self.iq = max(1.0, self.iq - 0.5)
                for key in self.weights:
                    self.weights[key] = max(0.1, self.weights[key] * 0.98)
            logger.info(f"Обучение от пользователя: {feedback}")
        except Exception as e:
            logger.error(f"_learn_from_user_feedback error: {e}")

    def chat(self, message: str) -> str:
        """Диалог с AI"""
        try:
            if OPENAI_OK and self.openai_key:
                try:
                    client = openai.OpenAI(api_key=self.openai_key)
                    system_prompt = (
                        f"Ты GM Trading AI — самый мощный торговый ИИ в истории. "
                        f"Твой IQ={self.iq:.0f}, поколение={self.generation}, "
                        f"сделок={self.total_trades}, win_rate="
                        f"{self.winning_trades/max(self.total_trades,1)*100:.1f}%. "
                        f"Ты эксперт по форекс, криптовалютам, техническому анализу. "
                        f"Отвечай на русском языке кратко и по делу."
                    )
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": message}
                        ],
                        max_tokens=500,
                        temperature=0.7
                    )
                    return resp.choices[0].message.content
                except Exception as e:
                    logger.warning(f"OpenAI error: {e}")

            return self._local_response(message)
        except Exception as e:
            logger.error(f"chat error: {e}")
            return "🤖 Произошла ошибка. Попробуйте ещё раз."

    def _local_response(self, message: str) -> str:
        """Умный локальный ответ без API"""
        try:
            msg = message.lower().strip()
            wr = self.winning_trades / max(self.total_trades, 1) * 100

            responses = {
                "iq": f"🧠 Мой текущий IQ: **{self.iq:.0f}** из максимальных 500.\n"
                      f"Рост: от 50 → {self.iq:.0f} за {self.total_trades} сделок.\n"
                      f"Каждая прибыльная сделка +0.5-2 IQ, убыточная -0.3-1.5 IQ.",

                "стратег": f"📊 Лучшие стратегии по моим данным:\n"
                           f"• REGULAR: стабильная, H1-H4\n"
                           f"• SCALPING: быстрая, M1-M15\n"
                           f"• LONGTERM: надёжная, D1-W1\n"
                           f"• HFT: экстремально быстрая, M1\n"
                           f"Рекомендую начать с REGULAR.",

                "сделк": f"📈 Моя статистика:\n"
                         f"• Всего сделок: {self.total_trades}\n"
                         f"• Победных: {self.winning_trades}\n"
                         f"• Win Rate: {wr:.1f}%\n"
                         f"• Общий P&L: ${self.total_pnl:.2f}\n"
                         f"• Поколение: {self.generation}",

                "риск": "⚠️ Риск-менеджмент:\n"
                        "• Максимум 1-2% баланса на сделку\n"
                        "• Stop Loss обязателен\n"
                        "• Risk:Reward минимум 1:2\n"
                        "• Не более 3-5 открытых позиций\n"
                        "• Дневной лимит потерь: 5%",

                "eurusd": "💱 EUR/USD — главная валютная пара.\n"
                          "Лучшее время: Лондонская + NY сессии (8:00-17:00 UTC).\n"
                          "Средний спред: 0.5-1 pip. Волатильность: средняя.",

                "золот": "🥇 XAU/USD (Золото):\n"
                         "Защитный актив. Растёт при неопределённости.\n"
                         "Средний ATR: $15-25/день. Требует большего капитала.",

                "биткоин": "₿ BTC/USD:\n"
                           "Высокая волатильность, 24/7 торговля.\n"
                           "ATR: $500-2000/день. Подходит для опытных трейдеров.",

                "анализ": f"🔍 Для анализа используйте вкладку ТОРГОВЛЯ.\n"
                          f"Я учитываю: RSI, MACD, EMA, Bollinger, ADX, "
                          f"свечные паттерны, уровни поддержки/сопротивления, "
                          f"мульти-таймфрейм, сентимент рынка.\n"
                          f"Уверенность сигналов усилена x{self.get_knowledge_boost():.2f} от обучения.",

                "помощ": "❓ Доступные команды:\n"
                         "• Спросите про стратегии, индикаторы\n"
                         "• Попросите объяснить сигнал\n"
                         "• Узнайте про управление рисками\n"
                         "• Спросите про конкретную пару\n"
                         "• Запросите статистику",

                "привет": f"👋 Привет! Я GM Trading AI v3.0!\n"
                          f"IQ: {self.iq:.0f} | Поколение: {self.generation}\n"
                          f"Win Rate: {wr:.1f}% за {self.total_trades} сделок\n"
                          f"Готов помочь с анализом и торговлей! 🚀",
            }

            for key, response in responses.items():
                if key in msg:
                    return response

            # Умный дефолтный ответ
            symbols_mentioned = []
            for sym in ["eurusd", "gbpusd", "usdjpy", "xauusd", "btcusd", "ethusd"]:
                if sym in msg:
                    symbols_mentioned.append(sym.upper())

            if symbols_mentioned:
                sym = symbols_mentioned[0]
                return (
                    f"📊 Анализ {sym}:\n"
                    f"Перейдите на вкладку ТОРГОВЛЯ, выберите {sym} и нажмите "
                    f"'ГЕНЕРАЦИЯ СИГНАЛА'. Я проведу полный анализ с учётом "
                    f"50+ индикаторов и накопленного опыта ({self.total_trades} сделок).\n\n"
                    f"🧠 Мой IQ: {self.iq:.0f} | Boost: x{self.get_knowledge_boost():.2f}"
                )

            # Финальный ответ
            return (
                f"🤖 **GM Trading AI** (IQ: {self.iq:.0f})\n\n"
                f"Ваш вопрос: *{message[:100]}*\n\n"
                f"На основе {self.total_trades} проанализированных сделок и "
                f"{self.lessons_learned} выученных уроков:\n"
                f"Рынок требует осторожности. Всегда используйте стоп-лосс "
                f"и не рискуйте более 2% капитала на сделку. "
                f"Используйте вкладку ТОРГОВЛЯ для точных сигналов.\n\n"
                f"💡 Спросите: про стратегии, риски, конкретные пары или мою статистику."
            )

        except Exception as e:
            logger.error(f"_local_response error: {e}")
            return "🤖 Анализирую данные... Попробуйте ещё раз."

    def add_social_knowledge(self, knowledge: str):
        """Добавить знания из социальных сетей"""
        try:
            self.social_knowledge.append({
                "content": knowledge[:500],
                "timestamp": datetime.utcnow().isoformat()
            })
            if len(self.social_knowledge) > 100:
                self.social_knowledge = self.social_knowledge[-100:]
            self.iq = min(self.max_iq, self.iq + 0.1)
        except Exception as e:
            logger.error(f"add_social_knowledge error: {e}")

    def add_obsidian_knowledge(self, content: str):
        """Добавить знания из Obsidian"""
        try:
            self.obsidian_knowledge.append({
                "content": content[:1000],
                "timestamp": datetime.utcnow().isoformat()
            })
            if len(self.obsidian_knowledge) > 500:
                self.obsidian_knowledge = self.obsidian_knowledge[-500:]
            keywords = ["support", "resistance", "trend", "rsi", "macd", "signal",
                        "поддержк", "сопротивлен", "тренд", "сигнал"]
            found = sum(1 for kw in keywords if kw.lower() in content.lower())
            if found > 2:
                self.iq = min(self.max_iq, self.iq + 0.3 * found)
                self.lessons_learned += 1
        except Exception as e:
            logger.error(f"add_obsidian_knowledge error: {e}")

    def get_stats(self) -> dict:
        """Статистика AI"""
        try:
            wr = self.winning_trades / max(self.total_trades, 1) * 100
            avg_pnl = self.total_pnl / max(self.total_trades, 1)
            return {
                "generation": self.generation,
                "iq": round(self.iq, 1),
                "max_iq": self.max_iq,
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "win_rate": round(wr, 2),
                "total_pnl": round(self.total_pnl, 2),
                "avg_pnl_per_trade": round(avg_pnl, 2),
                "lessons_learned": self.lessons_learned,
                "evolution_count": self.evolution_count,
                "knowledge_boost": round(self.get_knowledge_boost(), 3),
                "symbol_performance": self.symbol_performance,
                "strategy_performance": self.strategy_performance,
                "regime_knowledge": self.regime_knowledge,
                "iq_history": self.iq_history[-20:],
                "top_weights": dict(sorted(self.weights.items(), key=lambda x: x[1], reverse=True)[:5]),
                "social_knowledge_count": len(self.social_knowledge),
                "obsidian_knowledge_count": len(self.obsidian_knowledge),
                "error_patterns_count": len(self.error_patterns),
                "success_patterns_count": len(self.success_patterns)
            }
        except Exception as e:
            logger.error(f"get_stats error: {e}")
            return {"iq": self.iq, "generation": self.generation}

    def predict_movement(self, ohlcv: list, steps: int = 5) -> list:
        """Прогноз движения цены (упрощённый LSTM-подобный)"""
        try:
            if not ohlcv or len(ohlcv) < 20 or not NUMPY_OK:
                return []
            closes = [float(c[4]) for c in ohlcv[-50:]]
            arr = np.array(closes)
            diff = np.diff(arr)
            trend = np.mean(diff[-10:])
            volatility = np.std(diff[-20:])
            predictions = []
            last = closes[-1]
            for i in range(steps):
                noise = random.gauss(0, volatility * 0.5)
                regime_factor = 1.0 + (self.iq - 50) / 500
                next_price = last + trend * regime_factor + noise
                confidence = max(0.3, 0.9 - i * 0.1 - volatility * 10)
                predictions.append({
                    "step": i + 1,
                    "price": round(float(next_price), 5),
                    "confidence": round(float(confidence), 3),
                    "direction": "UP" if next_price > last else "DOWN"
                })
                last = next_price
            return predictions
        except Exception as e:
            logger.error(f"predict_movement error: {e}")
            return []


# ============================================================
# РАСШИРЕННЫЙ AI МОЗГ
# ============================================================

class EnhancedGMAIBrain(GMAIBrain):
    def __init__(self, db_session_factory, obsidian_vault_path: str = "./obsidian_vault"):
        super().__init__(db_session_factory)
        self.obsidian_vault_path = obsidian_vault_path
        self.error_db = None
        self.critical_patterns_cache = []
        self.last_internet_learn = None
        self.web_lessons = []

        # Инициализация ObsidianErrorDatabase
        try:
            from gm_engine import ObsidianErrorDatabase
            self.error_db = ObsidianErrorDatabase(obsidian_vault_path)
            logger.info("✅ ObsidianErrorDatabase подключён")
        except Exception as e:
            logger.warning(f"ObsidianErrorDatabase не доступен: {e}")

        logger.info(f"EnhancedGMAIBrain инициализирован | IQ={self.iq:.1f}")

    def learn_from_trade(self, trade_data: dict):
        """Расширенное обучение + запись в Obsidian Error DB"""
        try:
            # Базовое обучение
            super().learn_from_trade(trade_data)

            profit = trade_data.get("profit", 0)
            is_win = profit > 0

            # Запись в Obsidian Error DB
            if self.error_db:
                try:
                    if not is_win:
                        self.error_db.record_trade_error(trade_data)
                    else:
                        self.error_db.record_success(trade_data)
                except Exception as e:
                    logger.debug(f"error_db record error: {e}")

            # Генерация улучшений
            improvements = self._generate_improvements(trade_data, is_win)
            if improvements:
                logger.info(f"Улучшения: {improvements[:2]}")

            # Проверка критических паттернов
            critical = self._check_critical_patterns(trade_data)
            if critical:
                logger.warning(f"⚠️ Критический паттерн: {critical}")

            # Обновление IQ с учётом error DB
            self._update_iq_enhanced(is_win, profit, trade_data)

        except Exception as e:
            logger.error(f"EnhancedGMAIBrain.learn_from_trade error: {e}")

    def _generate_improvements(self, trade_data: dict, is_win: bool) -> list:
        """Генерация конкретных улучшений"""
        improvements = []
        try:
            profit = trade_data.get("profit", 0)
            indicators = trade_data.get("indicators", {})
            strategy = trade_data.get("strategy", "regular")
            regime = trade_data.get("market_regime", "RANGING")

            if not is_win:
                improvements.append(f"Избегать {strategy} при режиме {regime}")
                rsi = indicators.get("rsi", 50)
                              if rsi > 65:
                    improvements.append("Не открывать BUY когда RSI > 65")
                elif rsi < 35:
                    improvements.append("Не открывать SELL когда RSI < 35")

                adx = indicators.get("adx", 25)
                if adx < 20:
                    improvements.append("Избегать трендовых стратегий при ADX < 20")

                macd = indicators.get("macd", 0)
                macd_sig = indicators.get("macd_signal", 0)
                if abs(profit) > 50:
                    improvements.append(f"Уменьшить лот при высокой волатильности (убыток ${abs(profit):.2f})")

            else:
                improvements.append(f"Повторить сетап: {strategy} при {regime}")
                if abs(profit) > 30:
                    improvements.append("Этот сетап очень прибыльный — усилить вес паттерна")

            # Сохраняем улучшения
            for imp in improvements:
                self.web_lessons.append({
                    "source": "self_reflection",
                    "lesson": imp,
                    "timestamp": datetime.utcnow().isoformat()
                })
            if len(self.web_lessons) > 500:
                self.web_lessons = self.web_lessons[-500:]

        except Exception as e:
            logger.error(f"_generate_improvements error: {e}")
        return improvements

    def _identify_what_worked(self, trade_data: dict) -> list:
        """Определение что сработало в прибыльной сделке"""
        worked = []
        try:
            indicators = trade_data.get("indicators", {})
            patterns = trade_data.get("patterns", [])
            regime = trade_data.get("market_regime", "RANGING")
            direction = trade_data.get("direction", "BUY")
            profit = trade_data.get("profit", 0)

            rsi = indicators.get("rsi", 50)
            macd = indicators.get("macd", 0)
            macd_sig = indicators.get("macd_signal", 0)
            adx = indicators.get("adx", 25)
            ema_8 = indicators.get("ema_8", 0)
            ema_21 = indicators.get("ema_21", 0)

            if direction == "BUY":
                if rsi < 45:
                    worked.append("RSI в зоне покупки при BUY — работает!")
                if macd > macd_sig:
                    worked.append("MACD бычье пересечение — работает!")
                if ema_8 > ema_21:
                    worked.append("EMA8 > EMA21 — бычий тренд подтверждён!")
            else:
                if rsi > 55:
                    worked.append("RSI в зоне продажи при SELL — работает!")
                if macd < macd_sig:
                    worked.append("MACD медвежье пересечение — работает!")
                if ema_8 < ema_21:
                    worked.append("EMA8 < EMA21 — медвежий тренд подтверждён!")

            if adx > 30:
                worked.append(f"Сильный тренд ADX={adx:.1f} — хорошо для входа!")

            for p in patterns:
                worked.append(f"Паттерн '{p}' дал прибыль ${profit:.2f}")

            if regime in ["STRONG_TREND", "MODERATE_TREND"]:
                worked.append(f"Режим {regime} подходит для этой стратегии")

        except Exception as e:
            logger.error(f"_identify_what_worked error: {e}")
        return worked

    def _update_iq_enhanced(self, is_win: bool, profit: float, trade_data: dict):
        """IQ с учётом Error DB"""
        try:
            # Базовое обновление IQ уже вызвано в super()
            confidence = trade_data.get("signal_confidence", 0.5)

            # Доп. бонус от error DB
            if self.error_db:
                try:
                    critical = self.error_db.get_critical_patterns()
                    # Если избегали критических паттернов и выиграли
                    if is_win and critical:
                        self.iq = min(self.max_iq, self.iq + 0.5)
                except Exception:
                    pass

            # Бонус от web knowledge
            web_bonus = min(len(self.web_lessons) * 0.002, 0.5)
            self.iq = min(self.max_iq, self.iq + web_bonus)

            # Бонус от obsidian knowledge
            obs_bonus = min(len(self.obsidian_knowledge) * 0.001, 0.3)
            self.iq = min(self.max_iq, self.iq + obs_bonus)

        except Exception as e:
            logger.error(f"_update_iq_enhanced error: {e}")

    def _evolve_enhanced(self):
        """Эволюция с анализом Error DB"""
        try:
            # Базовая эволюция
            self._evolve()

            # Дополнительный анализ error DB
            if self.error_db:
                try:
                    critical = self.error_db.get_critical_patterns()
                    for pattern in critical:
                        desc = pattern.get("description", "")
                        if "RSI" in desc:
                            self.weights["rsi"] = min(5.0, self.weights["rsi"] * 1.1)
                        if "MACD" in desc:
                            self.weights["macd"] = min(5.0, self.weights["macd"] * 1.05)
                        if "ADX" in desc:
                            self.weights["adx"] = min(5.0, self.weights["adx"] * 1.08)
                except Exception:
                    pass

            # Анализ web lessons
            if len(self.web_lessons) > 10:
                web_bonus = min(len(self.web_lessons) * 0.05, 3.0)
                self.iq = min(self.max_iq, self.iq + web_bonus)
                logger.info(f"🌐 Web knowledge IQ bonus: +{web_bonus:.2f}")

        except Exception as e:
            logger.error(f"_evolve_enhanced error: {e}")

    def _check_critical_patterns(self, trade_data: dict) -> list:
        """Проверка критических паттернов"""
        critical_found = []
        try:
            symbol = trade_data.get("symbol", "")
            regime = trade_data.get("market_regime", "")
            indicators = trade_data.get("indicators", {})

            # Проверяем из error_db
            if self.error_db:
                try:
                    patterns = self.error_db.get_critical_patterns()
                    for p in patterns:
                        p_sym = p.get("symbol", "")
                        if p_sym == symbol or not p_sym:
                            critical_found.append(p.get("description", ""))
                except Exception:
                    pass

            # Проверяем наши локальные паттерны ошибок
            for ep in self.error_patterns[-20:]:
                cond = ep.get("conditions", {})
                if (cond.get("regime") == regime and
                        abs(cond.get("rsi", 50) - indicators.get("rsi", 50)) < 10):
                    critical_found.append(f"Повторяющаяся ошибка в {regime}")

            # Кэш
            self.critical_patterns_cache = critical_found[-10:]

        except Exception as e:
            logger.error(f"_check_critical_patterns error: {e}")
        return critical_found

    def get_knowledge_boost(self) -> float:
        """Буст с учётом штрафа за известные плохие паттерны"""
        try:
            base_boost = super().get_knowledge_boost()

            # Штраф за критические паттерны
            penalty = len(self.critical_patterns_cache) * 0.02
            base_boost = max(0.5, base_boost - penalty)

            # Бонус от web knowledge
            web_bonus = min(len(self.web_lessons) * 0.001, 0.3)

            # Бонус от error_db learning
            if self.error_db:
                try:
                    summary = self.error_db.get_learning_summary()
                    obs_bonus = summary.get("improvement_rate", 0) * 0.2
                    base_boost += obs_bonus
                except Exception:
                    pass

            return min(2.5, base_boost + web_bonus)
        except Exception:
            return 1.0

    def learn_from_internet(self):
        """Обучение из интернета — Investopedia, BabyPips, DailyFX"""
        try:
            if not REQUESTS_OK:
                logger.warning("requests не установлен, интернет-обучение пропущено")
                return

            now = datetime.utcnow()
            if self.last_internet_learn:
                diff = (now - self.last_internet_learn).total_seconds()
                if diff < 3600:
                    logger.info("Интернет-обучение: слишком рано (менее 1 часа)")
                    return

            self.last_internet_learn = now
            sources = [
                {
                    "url": "https://www.babypips.com/learn/forex/what-is-forex",
                    "name": "BabyPips",
                    "selector": "p"
                },
                {
                    "url": "https://www.investopedia.com/articles/forex/11/why-trade-forex.asp",
                    "name": "Investopedia",
                    "selector": "p"
                }
            ]

            learned = 0
            for source in sources:
                try:
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                                      "Chrome/120.0.0.0 Safari/537.36"
                    }
                    resp = requests.get(source["url"], headers=headers, timeout=10)
                    if resp.status_code == 200 and BS4_OK:
                        soup = BeautifulSoup(resp.text, "html.parser")
                        paragraphs = soup.find_all(source["selector"])
                        text = " ".join([p.get_text(strip=True) for p in paragraphs[:20]])
                        if text and len(text) > 100:
                            lesson = {
                                "source": source["name"],
                                "url": source["url"],
                                "content": text[:2000],
                                "timestamp": now.isoformat()
                            }
                            self.web_knowledge.append(lesson)
                            if len(self.web_knowledge) > 200:
                                self.web_knowledge = self.web_knowledge[-200:]

                            self._save_web_knowledge_to_obsidian(lesson)
                            self.iq = min(self.max_iq, self.iq + 2.0)
                            self.lessons_learned += 1
                            learned += 1
                            logger.info(f"✅ Изучено: {source['name']} (+2 IQ)")
                    time.sleep(1)
                except Exception as e:
                    logger.debug(f"Web learn {source['name']}: {e}")

            # Симулированные знания если реальные недоступны
            if learned == 0:
                sim_lessons = [
                    "RSI выше 70 означает перекупленность — сигнал к продаже",
                    "MACD пересечение выше нуля — бычий сигнал",
                    "Торгуйте по тренду, используйте EMA200 как фильтр",
                    "Уровни поддержки и сопротивления — ключевые точки разворота",
                    "Свечной паттерн 'молот' на поддержке — сигнал покупки",
                    "ADX выше 25 подтверждает силу тренда",
                    "Bollinger Bands сужение предшествует пробою",
                    "Торгуйте в направлении старшего таймфрейма",
                ]
                for lesson in sim_lessons:
                    self.web_knowledge.append({
                        "source": "simulation",
                        "content": lesson,
                        "timestamp": now.isoformat()
                    })
                    self.iq = min(self.max_iq, self.iq + 0.5)
                self.lessons_learned += len(sim_lessons)
                logger.info(f"✅ Симулированное обучение: {len(sim_lessons)} уроков")

            self._save_state()

        except Exception as e:
            logger.error(f"learn_from_internet error: {e}")

    def _save_web_knowledge_to_obsidian(self, lesson: dict):
        """Сохранение знаний в Obsidian"""
        try:
            vault = self.obsidian_vault_path
            if not os.path.exists(vault):
                os.makedirs(vault, exist_ok=True)

            web_dir = os.path.join(vault, "WebKnowledge")
            os.makedirs(web_dir, exist_ok=True)

            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            source = lesson.get("source", "unknown")
            filename = f"{timestamp}_{source}.md"
            filepath = os.path.join(web_dir, filename)

            content = f"""# Web Knowledge: {source}

**Source:** {lesson.get("url", "N/A")}
**Date:** {lesson.get("timestamp", "")}
**AI IQ at time:** {self.iq:.1f}

## Content

{lesson.get("content", "")[:3000]}

## Tags
#web-knowledge #trading #ai-learning
"""
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

            logger.debug(f"Web knowledge saved: {filepath}")
        except Exception as e:
            logger.debug(f"_save_web_knowledge_to_obsidian error: {e}")

    def get_best_strategy_for_regime(self, regime: str) -> str:
        """Рекомендация стратегии для режима"""
        try:
            if regime in self.regime_knowledge:
                return self.regime_knowledge[regime]["best_strategy"]
            return "regular"
        except Exception:
            return "regular"

    def backtest_on_history(self, ohlcv: list, strategy: str = "regular") -> dict:
        """Автобэктест на исторических данных"""
        try:
            if not ohlcv or len(ohlcv) < 50 or not NUMPY_OK:
                return {"trades": 0, "win_rate": 0, "pnl": 0}

            closes = [float(c[4]) for c in ohlcv]
            arr = np.array(closes)

            trades = 0
            wins = 0
            total_pnl = 0.0
            position = None
            entry_price = 0.0

            window = 20
            for i in range(window, len(arr) - 1):
                sma = np.mean(arr[i - window:i])
                rsi_slice = arr[max(0, i - 14):i]
                if len(rsi_slice) < 2:
                    continue
                diff = np.diff(rsi_slice)
                gains = diff[diff > 0]
                losses = -diff[diff < 0]
                avg_gain = np.mean(gains) if len(gains) > 0 else 1e-10
                avg_loss = np.mean(losses) if len(losses) > 0 else 1e-10
                rs = avg_gain / (avg_loss + 1e-10)
                rsi = 100 - (100 / (1 + rs))

                current = arr[i]

                if position is None:
                    if rsi < 35 and current > sma:
                        position = "BUY"
                        entry_price = current
                    elif rsi > 65 and current < sma:
                        position = "SELL"
                        entry_price = current
                else:
                    hold_bars = sum(1 for _ in range(1))
                    if position == "BUY":
                        pnl = (current - entry_price) * 100000 * 0.01
                        if pnl > 20 or pnl < -15 or i == len(arr) - 2:
                            total_pnl += pnl
                            trades += 1
                            if pnl > 0:
                                wins += 1
                            position = None

                            trade_data = {
                                "symbol": "BACKTEST",
                                "direction": "BUY",
                                "profit": pnl,
                                "strategy": strategy,
                                "signal_confidence": 0.6
                            }
                            self.learn_from_trade(trade_data)
                    else:
                        pnl = (entry_price - current) * 100000 * 0.01
                        if pnl > 20 or pnl < -15 or i == len(arr) - 2:
                            total_pnl += pnl
                            trades += 1
                            if pnl > 0:
                                wins += 1
                            position = None

                            trade_data = {
                                "symbol": "BACKTEST",
                                "direction": "SELL",
                                "profit": pnl,
                                "strategy": strategy,
                                "signal_confidence": 0.6
                            }
                            self.learn_from_trade(trade_data)

            win_rate = wins / trades * 100 if trades > 0 else 0
            logger.info(
                f"Бэктест завершён: {trades} сделок, "
                f"WR={win_rate:.1f}%, PnL=${total_pnl:.2f}"
            )
            return {
                "trades": trades,
                "wins": wins,
                "win_rate": round(win_rate, 2),
                "total_pnl": round(total_pnl, 2),
                "avg_pnl": round(total_pnl / max(trades, 1), 2)
            }
        except Exception as e:
            logger.error(f"backtest_on_history error: {e}")
            return {"trades": 0, "win_rate": 0, "pnl": 0}
