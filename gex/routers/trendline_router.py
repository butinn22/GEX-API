"""Trendline routes: /trendlines/{ticker}."""
from fastapi import APIRouter, BackgroundTasks, Depends, Query

from gex.auth.dependencies import get_optional_user
from gex.auth.models import User, subscription_is_active
from gex.deps import provide_trendline_service
from gex.schemas import TrendlineAnalysisOut
from gex.adapters.cache.result_cache import cache_key, result_cache
from gex.adapters.notifications.telegram_sender import notify_trendline_analysis
from ._helpers import handle

from fastapi import Depends
from gex.auth.dependencies import require_subscription
router = APIRouter(dependencies=[Depends(require_subscription("BASIC"))], tags=["trendlines"])

#: Кэш финального результата авто-трендовых линий. Расчёт CPU-bound (пивоты, построение
#: линий по истории), входные бары обновляются на порядок реже, чем пользователь
#: открывает дашборд: без кэша каждая навигация по SPA пересчитывает линии всех карточек
#: заново у каждого пользователя. Ключ общий (Redis, SWR + single-flight): результат
#: одного расчёта мгновенно переиспользуется всеми — это и есть «подтянуть состояние
#: от опроса другого пользователя»; устаревшая запись отдаётся сразу и обновляется в фоне.
#:
#: TTL намеренно длинный (30 минут): трендовые линии — медленный индикатор, новые бары
#: приходят редко (1d — раз в день, 4h — раз в 4 часа), а пересчёт стоит CPU. Чаще,
#: чем раз в интервал, сервер не пересчитывает; клиент до этого порога не запрашивает
#: вовсе. Ручное «Обновить» на дашборде шлёт ``fresh=true`` и обходит кэш.
_TRENDLINES_TTL_S = 1800

@router.get("/trendlines/{ticker}", response_model=TrendlineAnalysisOut)
def get_trendlines(ticker: str, timeframe: str = Query("all"),
    resolution: int = Query(6, ge=2, le=50), history_bars: int = Query(300, ge=50, le=2000),
    max_lines: int = Query(5, ge=1, le=20), pivot_left: int = Query(5, ge=1, le=30),
    pivot_right: int = Query(5, ge=1, le=30), notify: bool = Query(False),
    fresh: bool = Query(False),
    background_tasks: BackgroundTasks = None,
    user: User | None = Depends(get_optional_user),
    svc=Depends(provide_trendline_service)) -> TrendlineAnalysisOut:
    """Анализ трендовых линий; при notify=true результат уходит в Telegram
    конкретного пользователя (chat_id из аккаунта).

    ``fresh=true`` — принудительный пересчёт (ручное «Обновить» на дашборде):
    запись кэша инвалидируется, свежий результат занимает её место.
    """
    # Уведомление в Telegram — только при активной подписке.
    chat_id = user.telegram_chat_id if (user and subscription_is_active(user)) else None
    def _do():
        if timeframe.strip().lower() == "all":
            return svc.analyze(ticker, resolution=resolution, history_bars=history_bars,
                max_support_lines=max_lines, max_resistance_lines=max_lines,
                pivot_left=pivot_left, pivot_right=pivot_right)
        return svc.analyze_timeframe(ticker, timeframe, resolution=resolution,
            history_bars=history_bars, max_support_lines=max_lines,
            max_resistance_lines=max_lines, pivot_left=pivot_left, pivot_right=pivot_right)
    notify_kwargs = dict(notify_fn=notify_trendline_analysis, notify=notify,
        background_tasks=background_tasks, chat_id=chat_id, error_src="yfinance/Bybit")
    if notify:
        # Персональная доставка: считаем на «живом» результате, минуя кэш, — уведомление
        # не должно подтягиваться из чужого запроса (и наоборот).
        return handle(_do, **notify_kwargs)
    _key = cache_key("res", "trendlines", ticker, timeframe.strip().lower(),
        resolution, history_bars, max_lines, pivot_left, pivot_right)
    if fresh:
        result_cache.invalidate(_key)
    return result_cache.get(_key, _TRENDLINES_TTL_S, lambda: handle(_do, **notify_kwargs))
