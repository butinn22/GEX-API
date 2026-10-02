"""Tests: notify-форматтеры Telegram принимают chat_id; finagent lang RU/EN + сводка."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from gex.adapters.notifications.telegram_sender import (
    notify_ta_analysis,
    notify_ta_timeframe,
    notify_trendline_analysis,
    notify_macd_trend_analysis,
    notify_gex_analysis,
    notify_signal_analysis,
    notify_scan_record,
)


@pytest.fixture(autouse=True)
def fake_send(monkeypatch):
    """Перехватываем реальную отправку и запоминаем вызовы."""
    calls = []

    def _fake(text, *, parse_mode=None, chat_id=None):
        calls.append({"text": text, "chat_id": chat_id, "parse_mode": parse_mode})
        return {"success": True, "batches_sent": 1, "errors": None, "raw_responses": [{"ok": True}]}

    monkeypatch.setattr("gex.adapters.notifications.telegram_sender.send_telegram_message", _fake)
    return calls


class _Obj:
    """Стаб: заданные поля + любые отсутствующие → None (не падаем на getattr)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getattr__(self, name):
        return None


class TestNotifyFormatters:
    """notify_* не должны падать с chat_id (раньше TypeError в форматтерах)."""

    def _ta(self):
        return _Obj(
            symbol="SPY", spot=500.0, consensus_trend="BULLISH",
            consensus_p_reversal=0.2, generated_at="2026-08-15T00:00:00Z",
            timeframes=[], summarize=_Obj(telegram_html_message="<b>SPY</b> summary"),
        )

    def _tl(self):
        return _Obj(
            symbol="SPY", timeframe="1d", last_close=500.0,
            trend_direction="BULLISH", trend_strength=70,
            support_lines=[], resistance_lines=[],
            strongest_support=_Obj(price=490.0, strength=80.0),
            strongest_resistance=_Obj(price=510.0, strength=75.0),
            line_angle_deg=5.0, fractal_trend="BULLISH", fractal_strength=60.0,
            combined_trend="BULLISH", combined_strength=65.0, atr=3.0, n_bars=300,
            consensus_trend="BULLISH", asset_type="stock",
            timeframes=[],
        )

    def test_ta_analysis(self, fake_send):
        res = notify_ta_analysis(self._ta(), chat_id="111")
        assert res["success"] is True
        assert fake_send[0]["chat_id"] == "111"

    def test_ta_timeframe(self, fake_send):
        tf = _Obj(timeframe="4h", trend=_Obj(direction="BEARISH", strength=55.0),
                  last_close=100.0,
                  indicators=_Obj(rsi=45.0, ema20=100.5, ema50=101.0, ema200=99.0,
                                   ema_bull_stack=False, ema_bear_stack=True,
                                   macd_hist=-0.5, macd_bull_cross=False, macd_bear_cross=True),
                  reversal=_Obj(p_reversal=0.3, signal="sell"),
                  summarize=_Obj(telegram_html_message="tf summary"))
        res = notify_ta_timeframe(tf, "BTC", chat_id="222")
        assert res["success"] is True
        assert fake_send[0]["chat_id"] == "222"

    def test_trendlines(self, fake_send):
        res = notify_trendline_analysis(self._tl(), chat_id="333")
        assert res["success"] is True
        assert fake_send[0]["chat_id"] == "333"

    def test_macd(self, fake_send):
        a = _Obj(symbol="SPY", spot=500.0, asset_type="stock",
                 consensus_trend="BULLISH", timeframes=[],
                 summarize=_Obj(telegram_html_message="macd"))
        res = notify_macd_trend_analysis(a, chat_id="444")
        assert res["success"] is True
        assert fake_send[0]["chat_id"] == "444"

    def test_gex(self, fake_send):
        a = _Obj(symbol="SPY", spot=500.0, days=30, direction="BULLISH",
                 confidence=70, p_up=0.6, p_down=0.4, regime="POSITIVE",
                 summarize=_Obj(telegram_html_message="gex"))
        res = notify_gex_analysis(a, chat_id="555")
        assert res["success"] is True
        assert fake_send[0]["chat_id"] == "555"

    def test_signal(self, fake_send):
        from datetime import datetime, timezone
        a = _Obj(symbol="SPY", timeframe="4h", spot=500.0,
                 generated_at=datetime.now(timezone.utc),
                 current_signal=_Obj(action="buy", order_type="entry_long", reason="trend",
                                      confidence_class="high", entry_score=0.7),
                 gex_context=None)
        res = notify_signal_analysis(a, chat_id="666")
        assert res["success"] is True
        assert fake_send[0]["chat_id"] == "666"

    def test_scan_record(self, fake_send):
        rec = _Obj(ticker="SPY", status="ok", error=None, ta=None, gex=None)
        res = notify_scan_record(rec, chat_id="777")
        assert res["success"] is True
        assert fake_send[0]["chat_id"] == "777"


class TestFinagentLang:
    """Язык ответа ИИ: lang=en добавляет инструкцию, ключ кэша включает lang."""

    def test_prompt_en_has_english_instruction(self):
        from finagent.agents.harness import get_harness
        bundle = {
            "symbol": "SPY", "price": 500.0, "date": "2026-08-15",
            "timeframes": ["4h", "1d"],
            "smc": {}, "trendlines": {}, "regression": {}, "gex": None,
            "horizon_days": 20, "lang": "en",
        }
        prompt = get_harness().build_predictor_prompt(bundle)
        assert "OUTPUT LANGUAGE" in prompt
        assert "ENGLISH" in prompt

    def test_prompt_ru_has_no_english_instruction(self):
        from finagent.agents.harness import get_harness
        bundle = {
            "symbol": "SPY", "price": 500.0, "date": "2026-08-15",
            "timeframes": ["4h", "1d"],
            "smc": {}, "trendlines": {}, "regression": {}, "gex": None,
            "horizon_days": 20, "lang": "ru",
        }
        prompt = get_harness().build_predictor_prompt(bundle)
        assert "OUTPUT LANGUAGE" not in prompt

    def test_llm_cache_key_includes_lang(self):
        from finagent.router import _llm_cache_key
        assert _llm_cache_key("SPY", 20, "deepseek-chat", "en").endswith(":en:deepseek-chat")
        assert _llm_cache_key("SPY", 20, "deepseek-chat", "ru").endswith(":ru:deepseek-chat")
        assert _llm_cache_key("SPY", 20, "deepseek-chat") != _llm_cache_key("SPY", 20, "deepseek-chat", "en")

    def test_telegram_html_ru_labels(self):
        from finagent.router import _format_telegram_html
        result = {
            "data": {"symbol": "SPY", "price": 500.0, "date": "2026-08-15"},
            "signal": {"direction": "bullish", "confidence": 0.75, "entry_price": 495.0,
                       "stop_loss": 485.0, "take_profit": 520.0},
            "gex": {"regime": "POSITIVE", "call_wall": 510.0, "put_wall": 490.0, "gamma_flip": 500.0},
            "prediction": {"direction": "bullish", "confidence": 0.8, "target_price": 530.0,
                           "key_support": 490.0, "key_resistance": 515.0,
                           "confidence_range": {"lower": 505.0, "upper": 525.0},
                           "rationale": "Test rationale"},
        }
        html = _format_telegram_html("SPY", result, "ru")
        assert "Прогноз ИИ" in html
        assert "Сигнал" in html
        assert "Поддержка" in html
        assert "Сопротивление" in html
        assert "Test rationale" in html

    def test_telegram_html_en_labels(self):
        from finagent.router import _format_telegram_html
        result = {
            "data": {"symbol": "SPY", "price": 500.0, "date": "2026-08-15"},
            "signal": {"direction": "bearish", "confidence": 0.6},
            "gex": {},
            "prediction": {"direction": "bearish", "confidence": 0.7, "rationale": "Bearish setup"},
        }
        html = _format_telegram_html("SPY", result, "en")
        assert "AI forecast" in html
        assert "Signal" in html
        assert "bearish" in html
        assert "Bearish setup" in html

    def test_telegram_html_no_prediction(self):
        from finagent.router import _format_telegram_html
        result = {
            "data": {"symbol": "BTC", "price": 60000.0, "date": "2026-08-15"},
            "signal": {"direction": "neutral", "confidence": 0.5},
            "gex": {},
            "prediction": {"error": "LLM down"},
        }
        html = _format_telegram_html("BTC", result, "en")
        assert "No AI response" in html
