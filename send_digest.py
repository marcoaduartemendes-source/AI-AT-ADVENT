#!/usr/bin/env python3
"""Daily AI news digest — fetches headlines, sends email. No external APIs needed."""
import os, sys, smtplib, ssl, socket, requests, xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# ── Config (all hardcoded except the password) ───────────────────────────────
EMAIL    = "Marcoaduartemendes@gmail.com"
PASSWORD = os.getenv("SMTP_PASSWORD", "")

FEEDS = [
    ("Anthropic & Claude",   "https://news.google.com/rss/search?q=Anthropic+Claude+AI&hl=en-US&gl=US&ceid=US:en"),
    ("OpenAI & ChatGPT",     "https://news.google.com/rss/search?q=OpenAI+ChatGPT&hl=en-US&gl=US&ceid=US:en"),
    ("Google Gemini",        "https://news.google.com/rss/search?q=Google+Gemini+DeepMind&hl=en-US&gl=US&ceid=US:en"),
    ("Meta AI",              "https://news.google.com/rss/search?q=Meta+AI+Llama&hl=en-US&gl=US&ceid=US:en"),
    ("AI Industry News",     "https://news.google.com/rss/search?q=artificial+intelligence&hl=en-US&gl=US&ceid=US:en"),
]

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
CUTOFF  = datetime.now(timezone.utc) - timedelta(hours=48)

# ── Fetch headlines ──────────────────────────────────────────────────────────
def fetch(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        items = root.findall(".//item")
        results = []
        for item in items[:6]:
            title = (item.findtext("title") or "").strip()
            link  = (item.findtext("link")  or "").strip()
            pub   = item.findtext("pubDate") or ""
            results.append((title, link, pub))
        return results
    except Exception as e:
        print(f"  Warning: {e}")
        return []

# ── Build HTML email ─────────────────────────────────────────────────────────
def build_html(sections):
    today = datetime.now().strftime("%A, %B %d, %Y")
    rows  = ""
    for category, items in sections:
        if not items:
            continue
        rows += f'<tr><td colspan="2" style="padding:16px 0 6px;font-size:18px;font-weight:700;color:#1a202c;border-top:2px solid #667eea">{category}</td></tr>'
        for title, link, pub in items:
            rows += f'<tr><td style="padding:6px 0"><a href="{link}" style="color:#667eea;text-decoration:none;font-size:15px">{title}</a><br><span style="font-size:12px;color:#999">{pub[:22]}</span></td></tr>'

    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8"></head>
<body style="margin:0;padding:24px;background:#f0f4f8;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif">
<div style="max-width:680px;margin:0 auto;background:#fff;border-radius:16px;padding:40px;box-shadow:0 4px 24px rgba(0,0,0,0.08)">
  <h1 style="margin:0 0 4px;font-size:26px;color:#1a202c">TODAY's MAJOR AI NEWS</h1>
  <p style="margin:0 0 24px;color:#718096;font-size:14px">{today}</p>
  <table width="100%" cellpadding="0" cellspacing="0">{rows}</table>
  <p style="margin:32px 0 0;text-align:center;font-size:12px;color:#aaa">Daily AI Digest · Delivered automatically</p>
</div></body></html>"""

# ── Send email ───────────────────────────────────────────────────────────────
def send(html, plain):
    if not PASSWORD:
        print("ERROR: SMTP_PASSWORD not set"); sys.exit(1)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "TODAY's MAJOR AI NEWS"
    msg["From"]    = EMAIL
    msg["To"]      = EMAIL
    msg.attach(MIMEText(plain, "plain", "utf-8"))
    msg.attach(MIMEText(html,  "html",  "utf-8"))

    # Force IPv4
    _orig = socket.getaddrinfo
    socket.getaddrinfo = lambda h,p,f=0,*a,**k: _orig(h,p,socket.AF_INET,*a,**k)

    errors = []
    for port, use_ssl in [(587, False), (465, True)]:
        try:
            if use_ssl:
                ctx = ssl.create_default_context()
                s = smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx, timeout=20)
            else:
                s = smtplib.SMTP("smtp.gmail.com", 587, timeout=20)
                s.starttls()
            s.login(EMAIL, PASSWORD)
            s.sendmail(EMAIL, [EMAIL], msg.as_string())
            s.quit()
            print(f"✓ Email sent via port {port}")
            return
        except Exception as e:
            errors.append(f"port {port}: {e}")

    print("ERROR sending email:"); [print(" ", e) for e in errors]
    sys.exit(2)

# ── Main ─────────────────────────────────────────────────────────────────────
print("Fetching AI news...")
sections = []
for category, url in FEEDS:
    print(f"  {category}...")
    items = fetch(url)
    sections.append((category, items))
    total = sum(len(i) for _,i in sections)

print(f"Got {total} headlines. Building email...")
html  = build_html(sections)
plain = "\n\n".join(
    f"=== {cat} ===\n" + "\n".join(f"• {t}\n  {l}" for t,l,_ in items)
    for cat, items in sections if items
)

print("Sending email to", EMAIL)
send(html, plain)
