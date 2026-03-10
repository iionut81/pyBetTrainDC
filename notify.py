"""Pipeline notification module.

Always writes to a daily log file under logs/.
Optionally sends email on failure if configured in config.yaml.
"""
from __future__ import annotations

import datetime as dt
import os
import smtplib
from email.mime.text import MIMEText
from pathlib import Path
from typing import List, Tuple

from config import CFG

_ROOT = Path(__file__).resolve().parent
_NOTIF = CFG.get("notifications", {})
_LOG_DIR = _ROOT / _NOTIF.get("log_dir", "logs")


def _log_path(date: dt.date | None = None) -> Path:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    d = date or dt.date.today()
    return _LOG_DIR / f"pipeline_{d.isoformat()}.log"


def log_run(
    pipeline: str,
    results: List[Tuple[str, bool]],
    date: dt.date | None = None,
) -> Path:
    """Append a run summary to the daily log file. Returns the log path."""
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_ok = sum(1 for _, ok in results if ok)
    n_fail = sum(1 for _, ok in results if not ok)
    status = "ALL OK" if n_fail == 0 else f"{n_fail} FAILED"

    lines = [
        f"[{now}] {pipeline} — {status} ({n_ok} ok, {n_fail} failed)",
    ]
    for label, ok in results:
        icon = "  OK " if ok else "  FAIL"
        lines.append(f"  {icon}  {label}")
    lines.append("")

    path = _log_path(date)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def send_failure_email(
    pipeline: str,
    results: List[Tuple[str, bool]],
) -> bool:
    """Send an email alert for failed steps. Returns True if sent."""
    email_cfg = _NOTIF.get("email", {})
    if not email_cfg.get("enabled", False):
        return False

    failed = [label for label, ok in results if not ok]
    if not failed:
        return False

    host = email_cfg.get("smtp_host", "")
    port = int(email_cfg.get("smtp_port", 587))
    use_tls = email_cfg.get("use_tls", True)
    username = email_cfg.get("username", "") or os.environ.get("SMTP_USERNAME", "")
    password = email_cfg.get("password", "") or os.environ.get("SMTP_PASSWORD", "")
    from_addr = email_cfg.get("from_addr", "")
    to_addrs = email_cfg.get("to_addrs", [])

    if not host or not from_addr or not to_addrs:
        print("[WARN] Email notification enabled but SMTP not fully configured.")
        return False

    subject = f"[Pred] {pipeline} — {len(failed)} step(s) failed ({dt.date.today().isoformat()})"
    body_lines = [f"Pipeline: {pipeline}", f"Date: {dt.date.today().isoformat()}", ""]
    for label, ok in results:
        icon = "OK" if ok else "FAILED"
        body_lines.append(f"  [{icon}] {label}")
    body = "\n".join(body_lines)

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = ", ".join(to_addrs)

    try:
        with smtplib.SMTP(host, port, timeout=30) as smtp:
            if use_tls:
                smtp.starttls()
            if username and password:
                smtp.login(username, password)
            smtp.sendmail(from_addr, to_addrs, msg.as_string())
        print(f"  Failure email sent to {', '.join(to_addrs)}")
        return True
    except Exception as exc:
        print(f"[WARN] Could not send failure email: {exc}")
        return False


def notify(
    pipeline: str,
    results: List[Tuple[str, bool]],
    date: dt.date | None = None,
) -> None:
    """Log the run and send email if there are failures."""
    log_path = log_run(pipeline, results, date)
    n_fail = sum(1 for _, ok in results if not ok)
    if n_fail > 0:
        print(f"  Log: {log_path}")
        send_failure_email(pipeline, results)