"""
MiniMax Token Plan 查询余量 + 反应式等待
"""
import os
import re
import time

import requests

MINIMAX_QUOTA_URL = "https://www.minimaxi.com/v1/token_plan/remains"

MINIMAX_ERR_QUOTA_INTERVAL = "2056"   # 超出 Token Plan 资源限制（等区间重置可恢复）
MINIMAX_ERR_BALANCE = "1008"          # 余额不足（等区间重置无效）

# 错误码解析不到时的兜底关键字（刻意不含 "exceeded"/"rate limit" 等宽泛词，
# 避免 "context length exceeded" 被误判而白睡数小时）
_QUOTA_HINTS = (
    "quota", "insufficient", "balance", "token plan", "额度", "余量",
)
_ERR_CODE_RE = re.compile(r"\b(\d{4})\b")


class LLMQuotaExceeded(Exception):
    """配额耗尽且无法靠等待区间重置恢复（周额度耗尽 / 余额不足）。"""


def query_minimax_quota(api_key=None, timeout=15):
    """查询 MiniMax 区间余量；成功返回 dict，失败返回 None（调用方应放行不阻塞）。

    返回: {'remaining_percent', 'status', 'remains_ms', 'weekly_remaining_percent'}
    """
    api_key = api_key or os.environ.get("LLM_API_KEY", "")
    if not api_key:
        return None
    try:
        r = requests.get(
            MINIMAX_QUOTA_URL,
            headers={"Authorization": "Bearer " + api_key,
                    "Content-Type": "application/json"},
            timeout=timeout,
            proxies={"http": "", "https": ""},   # 与项目一致：强制直连，绕过系统代理
        )
        d = r.json()
        if d.get("base_resp", {}).get("status_code") != 0:
            return None
        for m in d.get("model_remains", []):
            if m.get("model_name") == "general":     # 对话模型额度
                return {
                    "remaining_percent": m.get("current_interval_remaining_percent"),
                    "status": m.get("current_interval_status"),
                    "remains_ms": m.get("remains_time"),
                    "weekly_remaining_percent": m.get("current_weekly_remaining_percent"),
                }
    except Exception:
        return None
    return None


def classify_quota_error(err, api_key=None, min_percent=2, query_fn=None):
    """判定异常性质。

    返回:
    'interval' —— 配额耗尽但可等区间重置恢复（2056 等）
    'balance'  —— 余额不足（1008），等待无意义
    None       —— 非配额问题

    优先按错误码判定；解析不到码时退化为"关键字 / 配额 API 余量"判断。
    """
    s = str(err)
    codes = set(_ERR_CODE_RE.findall(s))
    if MINIMAX_ERR_BALANCE in codes:
        return "balance"
    if MINIMAX_ERR_QUOTA_INTERVAL in codes:
        return "interval"
    if any(h in s.lower() for h in _QUOTA_HINTS):
        return "interval"
    q = (query_fn or query_minimax_quota)(api_key)
    if q is not None and (q.get("remaining_percent") is not None
                        and q["remaining_percent"] <= min_percent):
        return "interval"
    return None


def is_quota_error(err, api_key=None, min_percent=2, query_fn=None):
    """便捷包装：任何配额类问题（可恢复或不可恢复）均返回 True。"""
    return classify_quota_error(err, api_key=api_key, min_percent=min_percent,
                                query_fn=query_fn) is not None


def wait_for_quota_reset(api_key=None, min_percent=2, min_wait=60, max_wait=6 * 3600,
                        buffer_s=30, sleep_fn=None, query_fn=None):
    """休眠到区间重置后返回 True（可重试）。

    - 周额度也耗尽 → 抛 LLMQuotaExceeded（等待区间重置无效）
    - 查不到配额信息 → 直接返回（不阻塞）
    - 睡眠时长夹在 [min_wait, max_wait]，防止 remains_ms 异常导致死循环/睡过头
    """
    q = (query_fn or query_minimax_quota)(api_key)
    if q is None:
        return True
    weekly = q.get("weekly_remaining_percent")
    if weekly is not None and weekly <= min_percent:
        raise LLMQuotaExceeded(
            f"MiniMax 周额度仅剩 {weekly}%（区间剩余 {q.get('remaining_percent')}%），"
            f"等待区间重置无效；请等周窗口刷新后用 --epoch 续跑本批次。"
        )
    remains_ms = q.get("remains_ms") or 0
    wait_s = max(min_wait, min(remains_ms / 1000.0 + buffer_s, max_wait))
    print(f"[quota] 区间剩余 {q.get('remaining_percent')}% 不足，"
        f"休眠 {wait_s / 60:.1f} 分钟后重试（等待区间重置）")
    (sleep_fn or time.sleep)(wait_s)
    return True


def handle_api_error(err, api_key=None, query_fn=None, sleep_fn=None):
    """LLM 调用异常的统一处理入口（供 prompting 层的 except 调用）。

    - 'interval'：休眠到区间重置后返回（调用方 retry 即继续）
    - 'balance' ：抛 LLMQuotaExceeded（需充值，等待无意义）
    - None      ：打印后返回（调用方照常重试）
    """
    kind = classify_quota_error(err, api_key=api_key, query_fn=query_fn)
    if kind == "balance":
        raise LLMQuotaExceeded(
            f"MiniMax 余额不足({MINIMAX_ERR_BALANCE})，等待区间重置无效；"
            f"请充值后重跑本批次。原始错误: {err!r}"
        )
    if kind == "interval":
        print("[quota] 检测到配额不足，等待区间重置…")
        wait_for_quota_reset(api_key=api_key, query_fn=query_fn, sleep_fn=sleep_fn)
        print("[quota] 区间已重置，继续重试本次调用")
        return
    print("API error, try again:", repr(err))
