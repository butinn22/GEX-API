"""Email service: SMTP email sending with HTML templates.

Uses Python's built-in smtplib — no external dependencies.
Graceful degradation: if SMTP is not configured, logs to console.
"""
from __future__ import annotations

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from gex.auth.config import settings

logger = logging.getLogger(__name__)


def _smtp_configured() -> bool:
    """Check if SMTP is actually configured (runtime config or .env)."""
    cfg = _effective_config()
    return bool(
        cfg["smtp_host"]
        and cfg["smtp_host"] != "localhost"
        and cfg["smtp_user"]
        and cfg["smtp_pass"]
    )


def _effective_config() -> dict:
    """Почтовая конфигурация: runtime_config поверх settings/.env."""
    try:
        from gex.auth.runtime_config import get_email_config
        return get_email_config()
    except Exception:  # noqa: BLE001
        return {
            "smtp_host": settings.SMTP_HOST,
            "smtp_port": settings.SMTP_PORT,
            "smtp_user": settings.SMTP_USER,
            "smtp_pass": settings.SMTP_PASS,
            "from_email": settings.FROM_EMAIL,
        }


def _send_email(
    to_email: str,
    subject: str,
    html_body: str,
    text_body: Optional[str] = None,
) -> bool:
    """Send an email via SMTP. Returns True on success, False on failure."""
    cfg = _effective_config()
    smtp_host = cfg["smtp_host"]
    smtp_port = int(cfg["smtp_port"] or 0)
    smtp_user = cfg["smtp_user"]
    smtp_pass = cfg["smtp_pass"]
    from_email = cfg["from_email"] or settings.FROM_EMAIL

    if not (smtp_host and smtp_host != "localhost" and smtp_user and smtp_pass):
        logger.info("SMTP not configured — email logged instead of sent")
        _log_email(to_email, subject)
        return False

    msg = MIMEMultipart("alternative")
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject

    if text_body:
        msg.attach(MIMEText(text_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        if smtp_port == 465:
            # SSL
            with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=15) as server:
                if smtp_user:
                    server.login(smtp_user, smtp_pass)
                server.send_message(msg)
        else:
            # STARTTLS (port 587 or 25)
            with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
                server.ehlo()
                if smtp_port == 587:
                    server.starttls()
                    server.ehlo()
                if smtp_user:
                    server.login(smtp_user, smtp_pass)
                server.send_message(msg)

        logger.info("Email sent to %s: %s", to_email, subject)
        return True

    except Exception as exc:
        logger.error("Failed to send email to %s: %s", to_email, exc)
        _log_email(to_email, subject)
        return False


def _log_email(to_email: str, subject: str) -> None:
    """Логировать факт письма (фолбэк без SMTP).

    ВАЖНО (аудит 2026-09-04): НЕ логируем тело письма — верификационные
    письма содержат одноразовые токены; попадание токена в логи = утечка.
    Для локальной разработки ссылка и так возвращается в ответе API.
    """
    logger.info("📧 EMAIL (console fallback): To=%s Subject=%s", to_email, subject)


VERIFICATION_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin:0;padding:0;background:#080b10;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#080b10;padding:40px 20px;">
  <tr>
    <td align="center">
      <table width="100%" cellpadding="0" cellspacing="0" style="max-width:520px;background:#0e1219;border:1px solid #1a2030;border-radius:14px;overflow:hidden;">

        <!-- Header -->
        <tr>
          <td style="padding:32px 32px 20px;text-align:center;">
            <div style="display:inline-block;width:48px;height:48px;border-radius:10px;background:linear-gradient(135deg,#4b8bf5,#6366f1);font-size:22px;font-weight:800;color:#fff;line-height:48px;text-align:center;">Γ</div>
            <h1 style="margin:16px 0 6px;font-size:22px;font-weight:700;color:#fff;letter-spacing:-0.02em;">GEX Analytics</h1>
            <p style="margin:0;font-size:14px;color:#949eae;">Подтверждение email-адреса</p>
          </td>
        </tr>

        <!-- Body -->
        <tr>
          <td style="padding:8px 32px 32px;">
            <p style="margin:0 0 16px;font-size:15px;color:#e8edf3;line-height:1.6;">
              Вы зарегистрировались в <strong>GEX Analytics</strong> — терминале микроструктурного анализа опционных рынков.
            </p>
            <p style="margin:0 0 20px;font-size:14px;color:#949eae;line-height:1.6;">
              Для завершения регистрации подтвердите ваш email-адрес, нажав кнопку ниже:
            </p>

            <!-- CTA Button -->
            <table width="100%" cellpadding="0" cellspacing="0">
              <tr>
                <td align="center" style="padding:8px 0 24px;">
                  <a href="{verify_link}" style="display:inline-block;padding:14px 40px;background:linear-gradient(135deg,#4b8bf5,#3b71d9);border-radius:8px;color:#fff;font-size:15px;font-weight:700;text-decoration:none;letter-spacing:-0.01em;box-shadow:0 4px 16px rgba(75,139,245,0.4);">
                    ✅ Подтвердить Email
                  </a>
                </td>
              </tr>
            </table>

            <!-- Fallback link -->
            <p style="margin:0 0 16px;font-size:12px;color:#5d6878;line-height:1.5;">
              Если кнопка не работает, скопируйте эту ссылку в браузер:
            </p>
            <p style="margin:0 0 24px;padding:12px 16px;background:#0c1017;border:1px solid #1a2030;border-radius:8px;font-family:'JetBrains Mono',Consolas,monospace;font-size:12px;color:#4b8bf5;word-break:break-all;">
              {verify_link}
            </p>

            <!-- Divider -->
            <div style="height:1px;background:#1a2030;margin:24px 0;"></div>

            <p style="margin:0;font-size:11px;color:#5d6878;line-height:1.6;">
              Если вы не регистрировались в GEX Analytics, просто проигнорируйте это письмо.
            </p>
          </td>
        </tr>

        <!-- Footer -->
        <tr>
          <td style="padding:16px 32px;background:#0c1017;border-top:1px solid #1a2030;text-align:center;">
            <p style="margin:0;font-size:11px;color:#3a4352;">
              GEX Analytics · Trading Terminal v0.3.0
            </p>
          </td>
        </tr>

      </table>
    </td>
  </tr>
</table>
</body>
</html>"""


def send_verification_email(email: str, token: str, base_url: str) -> bool:
    """Send verification email with token link.

    Args:
        email: Recipient email address
        token: Verification token (64-char hex)
        base_url: Base URL for the verification link

    Returns:
        True if email was sent via SMTP, False if logged to console.
    """
    verify_link = f"{base_url}/auth/verify-email?token={token}"
    html_body = VERIFICATION_HTML_TEMPLATE.format(verify_link=verify_link)
    text_body = (
        f"GEX Analytics — Подтверждение Email\n\n"
        f"Перейдите по ссылке для подтверждения:\n{verify_link}\n\n"
        f"Если вы не регистрировались, проигнорируйте это письмо."
    )

    return _send_email(
        to_email=email,
        subject="GEX Analytics — Подтвердите ваш email",
        html_body=html_body,
        text_body=text_body,
    )
