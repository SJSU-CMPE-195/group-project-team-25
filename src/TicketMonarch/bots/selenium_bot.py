"""Selenium bot that runs against TicketMonarch.

Telemetry is captured by the React app's built-in tracking.js and saved to
the Flask backend. After each run, this script pulls telemetry via the API,
saves JSON to data/bot/, and confirms the session as a bot so the RL agent
can do an online PPO update.

Bot types:
  - linear:   Straight-line mouse, uniform typing (obviously robotic)
  - scripted: Bezier curves, varied timing (more sophisticated)
  - replay:   Replays recorded human mouse/scroll patterns with noise

Usage:
    pip install selenium webdriver-manager
    python bots/selenium_bot.py --runs 5 --type linear
    python bots/selenium_bot.py --runs 5 --type scripted
    python bots/selenium_bot.py --runs 3 --type replay --replay-source data/human/telemetry_export.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait


SITE_URL = "http://localhost:3000"
API_URL = "http://localhost:5000"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "bot"

# Realistic fake identities for varied runs
FAKE_PEOPLE = [
    {"full_name": "Maria Gonzalez", "billing_address": "742 Evergreen Terrace", "city": "San Jose", "zip_code": "95112", "state": "California"},
    {"full_name": "James Chen", "billing_address": "1600 Amphitheatre Pkwy", "city": "Mountain View", "zip_code": "94043", "state": "California"},
    {"full_name": "Sarah Johnson", "billing_address": "350 Fifth Avenue", "city": "New York", "zip_code": "10118", "state": "New York"},
    {"full_name": "David Kim", "billing_address": "233 S Wacker Dr", "city": "Chicago", "zip_code": "60606", "state": "Illinois"},
    {"full_name": "Emily Davis", "billing_address": "600 Navarro St", "city": "San Antonio", "zip_code": "78205", "state": "Texas"},
    {"full_name": "Michael Brown", "billing_address": "1 Infinite Loop", "city": "Cupertino", "zip_code": "95014", "state": "California"},
    {"full_name": "Jessica Wilson", "billing_address": "100 Universal City Plz", "city": "Universal City", "zip_code": "91608", "state": "California"},
    {"full_name": "Robert Martinez", "billing_address": "1901 Main St", "city": "Dallas", "zip_code": "75201", "state": "Texas"},
]

CARD_NUMBERS = ["4111111111111111", "4242424242424242", "5500000000000004"]
CARD_EXPIRIES = ["12/28", "03/27", "09/29", "06/26"]


def get_form_data() -> dict:
    """Generate randomized but realistic checkout form data."""
    person = random.choice(FAKE_PEOPLE)
    return {
        "card_number": random.choice(CARD_NUMBERS),
        "card_expiry": random.choice(CARD_EXPIRIES),
        "card_cvv": str(random.randint(100, 999)),
        "full_name": person["full_name"],
        "billing_address": person["billing_address"],
        "apartment": random.choice(["", "", "", "Apt 2B", "Suite 100", "#4"]),
        "city": person["city"],
        "zip_code": person["zip_code"],
        "state": person["state"],
    }


def create_driver() -> webdriver.Chrome:
    """Launch Chrome (no extension needed — tracking.js captures everything)."""
    opts = Options()
    opts.add_argument("--no-first-run")
    opts.add_argument("--no-default-browser-check")
    driver = webdriver.Chrome(options=opts)
    driver.set_window_size(1920, 1080)
    return driver


def wait_for(driver, css, timeout=10):
    return WebDriverWait(driver, timeout).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, css))
    )


def wait_for_url(driver, url_contains, timeout=10):
    WebDriverWait(driver, timeout).until(EC.url_contains(url_contains))


# ---------------------------------------------------------------------------
# Human-like typing helpers
# ---------------------------------------------------------------------------

def _type_human(element, text):
    """Type with human-like timing: variable inter-key delays, occasional pauses,
    and burst typing for familiar sequences (like zip codes)."""
    element.clear()
    time.sleep(random.uniform(0.1, 0.3))

    i = 0
    while i < len(text):
        char = text[i]

        # Burst typing: sometimes type 2-3 chars quickly in a row
        burst_len = 1
        if random.random() < 0.3 and i + 2 < len(text):
            burst_len = random.randint(2, 3)

        for j in range(burst_len):
            if i + j >= len(text):
                break
            element.send_keys(text[i + j])
            # Fast within burst
            time.sleep(random.uniform(0.02, 0.06))

        i += burst_len

        # Inter-key delay: log-normal distribution (most fast, occasional pause)
        if i < len(text):
            if random.random() < 0.08:
                # Occasional thinking pause (looking at card, etc.)
                time.sleep(random.uniform(0.3, 0.8))
            else:
                # Normal typing delay with variance
                delay = random.lognormvariate(math.log(0.08), 0.4)
                delay = max(0.03, min(0.25, delay))
                time.sleep(delay)


def _type_uniform(element, text):
    """Perfectly uniform typing — obviously robotic."""
    element.clear()
    for char in text:
        element.send_keys(char)
        time.sleep(0.02)


# ---------------------------------------------------------------------------
# Human-like mouse movement helpers
# ---------------------------------------------------------------------------

def _human_move_and_click(driver, element, click_only=False):
    """Move to element with natural-looking curve, micro-corrections, and click."""
    if click_only:
        ActionChains(driver).move_to_element(element).click().perform()
        time.sleep(random.uniform(0.05, 0.15))
        return

    # Get current mouse position (approximate via element location)
    loc = element.location
    size = element.size
    target_x = loc['x'] + size['width'] / 2
    target_y = loc['y'] + size['height'] / 2

    # Multi-step approach with Bezier-like curve
    steps = random.randint(10, 25)
    actions = ActionChains(driver)

    # Random control point for the curve
    curve_offset_x = random.uniform(-80, 80)
    curve_offset_y = random.uniform(-40, 40)

    for i in range(steps):
        t = (i + 1) / steps

        # Ease-in-out timing (slow start, fast middle, slow end)
        t_eased = t * t * (3 - 2 * t)

        # Bezier-influenced offset that decreases toward target
        remaining = 1 - t_eased
        offset_x = int(curve_offset_x * remaining * math.sin(t * math.pi))
        offset_y = int(curve_offset_y * remaining * math.sin(t * math.pi))

        # Add micro-jitter (human hands aren't perfectly steady)
        jitter_x = random.gauss(0, 1.5)
        jitter_y = random.gauss(0, 1.0)

        dx = int(offset_x + jitter_x)
        dy = int(offset_y + jitter_y)

        if dx != 0 or dy != 0:
            actions.move_by_offset(dx, dy)

        # Variable speed: slower at start and end, faster in middle
        speed = 0.01 + 0.02 * (1 - abs(2 * t - 1))
        actions.pause(speed + random.uniform(0, 0.01))

    try:
        actions.perform()
    except Exception:
        pass

    # Final move to exact element + click
    time.sleep(random.uniform(0.05, 0.15))
    try:
        ActionChains(driver).move_to_element(element).click().perform()
    except Exception:
        pass

    time.sleep(random.uniform(0.1, 0.4))


def _linear_move_and_click(driver, element, click_only=False):
    """Straight-line move + click. Obviously robotic."""
    ActionChains(driver).move_to_element(element).click().perform()
    time.sleep(0.1)


def _random_scroll(driver, scrolls: int = 2):
    """Scroll up and down randomly to simulate browsing."""
    for _ in range(scrolls):
        dy = random.randint(100, 400) * random.choice([1, -1])
        driver.execute_script(f"window.scrollBy(0, {dy});")
        time.sleep(random.uniform(0.3, 0.8))


def _human_scroll(driver, scrolls: int = 3):
    """Human-like scrolling with momentum and variable speed."""
    for _ in range(scrolls):
        # Initial scroll direction and magnitude
        direction = random.choice([1, -1])
        total_dy = random.randint(150, 500) * direction

        # Break into multiple small scrolls (momentum effect)
        num_steps = random.randint(3, 8)
        for j in range(num_steps):
            # Momentum: decreasing scroll amounts
            factor = 1.0 - (j / num_steps) * 0.7
            step_dy = int(total_dy / num_steps * factor)
            if step_dy == 0:
                break
            driver.execute_script(f"window.scrollBy(0, {step_dy});")
            time.sleep(random.uniform(0.02, 0.08))

        # Pause between scroll gestures
        time.sleep(random.uniform(0.5, 1.5))


# ---------------------------------------------------------------------------
# Shared multi-page flow steps
# ---------------------------------------------------------------------------

def _go_home(driver):
    """Navigate to the home page and wait for concert cards to load."""
    driver.get(SITE_URL)
    wait_for(driver, ".tickets-button", timeout=10)
    # Mark this session as a bot so Confirmation.jsx won't auto-confirm as human
    driver.execute_script("window.sessionStorage.setItem('tm_is_bot', '1');")


def _pick_concert(driver, move_fn):
    """Click a concert's 'Tickets' button on the home page."""
    buttons = driver.find_elements(By.CSS_SELECTOR, ".tickets-button")
    if not buttons:
        print("  WARNING: No .tickets-button found on home page")
        return False
    target = random.choice(buttons)
    move_fn(driver, target)
    wait_for_url(driver, "/seats/")
    return True


def _pick_section(driver, move_fn):
    """Click a section button on the seat selection page, then Continue."""
    wait_for(driver, ".section-button", timeout=10)
    sections = driver.find_elements(By.CSS_SELECTOR, ".section-button")
    if not sections:
        print("  WARNING: No .section-button found")
        return False
    target = random.choice(sections)
    move_fn(driver, target)

    wait_for(driver, ".continue-button", timeout=5)
    cont_btn = driver.find_element(By.CSS_SELECTOR, ".continue-button")
    move_fn(driver, cont_btn)
    wait_for_url(driver, "/checkout")
    return True


def _handle_challenge(driver, move_fn, max_retries=3):
    """Detect and attempt to interact with challenge modals.

    The bot won't solve challenges correctly, but it will try rather than
    hanging. After failing all retries, it moves on so the run completes.
    """
    for attempt in range(max_retries):
        # Check if a challenge overlay is present
        overlays = driver.find_elements(By.CSS_SELECTOR, ".challenge-overlay")
        if not overlays:
            # No challenge — check if we reached confirmation
            if "/confirmation" in driver.current_url:
                print("  Reached /confirmation (challenge passed or allowed)")
                return
            time.sleep(1)
            continue

        print(f"  Challenge detected (attempt {attempt + 1}/{max_retries})")
        time.sleep(0.5)

        # Try the "Go Back" button (blocked state)
        go_back = driver.find_elements(By.CSS_SELECTOR, ".challenge-overlay .challenge-btn")
        if go_back:
            # Check if it's a simple "Go Back" button (blocked)
            btn_text = go_back[0].text.strip().lower()
            if btn_text == "go back":
                print("  Blocked by agent — clicking Go Back")
                go_back[0].click()
                time.sleep(1)
                return

        # Try slider challenge: drag the slider thumb to a random position
        slider_thumb = driver.find_elements(By.CSS_SELECTOR, ".slider-track")
        if slider_thumb:
            print("  Attempting slider challenge...")
            try:
                track = slider_thumb[0]
                track_size = track.size
                # Click somewhere on the track (random position — will probably miss)
                ActionChains(driver) \
                    .move_to_element_with_offset(track, int(track_size['width'] * random.uniform(0.2, 0.8)), int(track_size['height'] / 2)) \
                    .click_and_hold() \
                    .move_by_offset(int(track_size['width'] * random.uniform(-0.3, 0.3)), 0) \
                    .release() \
                    .perform()
                time.sleep(1)
            except Exception as e:
                print(f"  Slider attempt failed: {e}")
            continue

        # Try canvas text challenge: type random text
        captcha_input = driver.find_elements(By.CSS_SELECTOR, ".captcha-form input")
        if captcha_input:
            print("  Attempting canvas text challenge...")
            try:
                inp = captcha_input[0]
                inp.clear()
                # Type a random guess
                guess = ''.join(random.choices('ABCDEFGHJKMNPQRSTUVWXYZ23456789', k=5))
                inp.send_keys(guess)
                submit_btn = driver.find_elements(By.CSS_SELECTOR, ".captcha-form .challenge-btn")
                if submit_btn:
                    submit_btn[0].click()
                time.sleep(1)
            except Exception as e:
                print(f"  Canvas text attempt failed: {e}")
            continue

        # Try timed click challenge: click on the canvas
        click_canvas = driver.find_elements(By.CSS_SELECTOR, ".click-canvas")
        if click_canvas:
            print("  Attempting timed click challenge...")
            try:
                canvas = click_canvas[0]
                canvas_size = canvas.size
                # Click 4 random spots on the canvas (will almost certainly fail)
                for _ in range(4):
                    x_off = int(canvas_size['width'] * random.uniform(0.1, 0.9))
                    y_off = int(canvas_size['height'] * random.uniform(0.1, 0.9))
                    ActionChains(driver) \
                        .move_to_element_with_offset(canvas, x_off, y_off) \
                        .click() \
                        .perform()
                    time.sleep(random.uniform(0.3, 0.8))
                time.sleep(2)
            except Exception as e:
                print(f"  Timed click attempt failed: {e}")

            # Check for "Try Again" button
            retry_btns = driver.find_elements(By.CSS_SELECTOR, ".challenge-overlay .challenge-btn")
            for btn in retry_btns:
                if "try again" in btn.text.strip().lower():
                    print("  Clicking 'Try Again'...")
                    btn.click()
                    time.sleep(1)
                    break
            continue

        # Generic fallback: click any visible challenge button
        buttons = driver.find_elements(By.CSS_SELECTOR, ".challenge-overlay .challenge-btn")
        for btn in buttons:
            try:
                btn.click()
                time.sleep(1)
                break
            except Exception:
                pass

    # After all retries, check final state
    if "/confirmation" in driver.current_url:
        print("  Reached /confirmation after challenge")
    else:
        print("  Challenge not solved after all retries — moving on")


def _fill_checkout(driver, type_fn, move_fn):
    """Fill out the checkout form and submit.

    Fills every input field found on the page. Known fields get realistic
    fake data; any unknown fields (e.g. hidden honeypots) get generic
    filler — just like a real scraper bot that parses the DOM.
    """
    form = get_form_data()
    wait_for(driver, "#card_number", timeout=10)

    # Known fields with specific fake data
    known_values = {
        "card_number": form["card_number"],
        "card_expiry": form["card_expiry"],
        "card_cvv": form["card_cvv"],
        "full_name": form["full_name"],
        "billing_address": form["billing_address"],
        "apartment": form["apartment"],
        "city": form["city"],
        "zip_code": form["zip_code"],
    }

    # Generic filler for unknown fields (bot doesn't know what they are)
    GENERIC_FILLERS = [
        "test@email.com", "5551234567", "John Doe", "123 Main St",
        "Springfield", "12345", "some value",
    ]

    # Discover ALL input fields on the page and fill them
    all_inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='text'], input[type='tel'], input[type='email'], input:not([type])")
    filler_idx = 0

    for inp in all_inputs:
        try:
            field_id = inp.get_attribute("id") or inp.get_attribute("name") or ""

            # Skip fields that are part of dropdowns or already filled
            if not field_id:
                continue

            # Determine value: use known data if available, generic filler otherwise
            value = known_values.get(field_id)
            if value is None:
                # Unknown field — fill with generic data (this catches honeypots)
                value = GENERIC_FILLERS[filler_idx % len(GENERIC_FILLERS)]
                filler_idx += 1

            if not value:
                continue

            # Try to interact normally first; if the field isn't interactable
            # (off-screen, hidden), use JS to set value and dispatch events
            # WITHOUT unhiding — keeps the field invisible on screen
            if not inp.is_displayed():
                driver.execute_script("""
                    var el = arguments[0];
                    var value = arguments[1];
                    // Set value via React's value setter to trigger onChange
                    var nativeInputValueSetter = Object.getOwnPropertyDescriptor(
                        window.HTMLInputElement.prototype, 'value').set;
                    nativeInputValueSetter.call(el, value);
                    el.dispatchEvent(new Event('input', { bubbles: true }));
                    el.dispatchEvent(new Event('change', { bubbles: true }));
                    // Dispatch keydown/keyup events so tracking.js captures keystrokes
                    for (var i = 0; i < value.length; i++) {
                        el.dispatchEvent(new KeyboardEvent('keydown', {
                            key: value[i], code: 'Key' + value[i].toUpperCase(),
                            bubbles: true
                        }));
                        el.dispatchEvent(new KeyboardEvent('keyup', {
                            key: value[i], code: 'Key' + value[i].toUpperCase(),
                            bubbles: true
                        }));
                    }
                """, inp, value)
            else:
                move_fn(driver, inp, click_only=True)
                type_fn(inp, value)

            time.sleep(random.uniform(0.2, 0.6))
        except Exception as e:
            print(f"  WARNING: Could not fill field: {e}")

    # Select state dropdown
    try:
        state_el = driver.find_element(By.ID, "state")
        move_fn(driver, state_el, click_only=True)
        select = Select(state_el)
        select.select_by_visible_text(form["state"])
        time.sleep(random.uniform(0.3, 0.6))
    except Exception as e:
        print(f"  WARNING: Could not select state: {e}")

    # Click Purchase
    try:
        purchase = wait_for(driver, ".purchase-button", timeout=5)
        move_fn(driver, purchase)
    except Exception as e:
        print(f"  WARNING: Could not click Purchase: {e}")

    # Wait for either confirmation page or a challenge modal
    try:
        wait_for_url(driver, "/confirmation", timeout=5)
    except Exception:
        # Check if a challenge appeared
        _handle_challenge(driver, move_fn)


# ---------------------------------------------------------------------------
# Bot behaviors
# ---------------------------------------------------------------------------

def linear_bot(driver):
    """Straight-line mouse, instant clicks, uniform typing. Very bot-like."""
    _go_home(driver)
    time.sleep(0.3)

    if not _pick_concert(driver, _linear_move_and_click):
        return
    time.sleep(0.2)

    if not _pick_section(driver, _linear_move_and_click):
        return
    time.sleep(0.2)

    _fill_checkout(driver, _type_uniform, _linear_move_and_click)


def scripted_bot(driver):
    """Bezier curve mouse, human-like typing, scrolling. More sophisticated."""
    _go_home(driver)
    time.sleep(random.uniform(0.8, 2.0))

    # Browse around first
    _human_scroll(driver, scrolls=random.randint(1, 3))

    if not _pick_concert(driver, _human_move_and_click):
        return
    time.sleep(random.uniform(0.5, 1.2))

    # Look at seats
    _human_scroll(driver, scrolls=random.randint(1, 2))

    if not _pick_section(driver, _human_move_and_click):
        return
    time.sleep(random.uniform(0.3, 0.8))

    _fill_checkout(driver, _type_human, _human_move_and_click)


def replay_bot(driver, source_path: str):
    """Replay recorded human mouse/scroll patterns with noise, then complete flow."""
    segments = _load_replay_segments(source_path)
    if not segments:
        print("  No segments found in replay source")
        return

    # --- Page 1: Home ---
    _go_home(driver)
    time.sleep(random.uniform(0.5, 1.0))

    # Replay human mouse movement on the home page
    seg = random.choice(segments)
    _replay_mouse(driver, seg.get("mouse", []), max_events=100)
    _replay_scroll(driver, seg.get("scroll", []), max_events=10)
    time.sleep(random.uniform(0.2, 0.5))

    if not _pick_concert(driver, _human_move_and_click):
        return
    time.sleep(random.uniform(0.3, 0.8))

    # --- Page 2: Seat Selection ---
    seg = random.choice(segments)
    _replay_mouse(driver, seg.get("mouse", []), max_events=60)
    time.sleep(random.uniform(0.2, 0.4))

    if not _pick_section(driver, _human_move_and_click):
        return
    time.sleep(random.uniform(0.3, 0.7))

    # --- Page 3: Checkout ---
    seg = random.choice(segments)
    _replay_mouse(driver, seg.get("mouse", []), max_events=50)
    _replay_scroll(driver, seg.get("scroll", []), max_events=5)
    time.sleep(random.uniform(0.2, 0.5))

    _fill_checkout(driver, _type_human, _human_move_and_click)


# ---------------------------------------------------------------------------
# Replay helpers
# ---------------------------------------------------------------------------

def _load_replay_segments(source_path: str) -> list[dict]:
    """Load segments from a Chrome extension JSON export."""
    with open(source_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = []
    if isinstance(data, dict):
        for key, val in data.items():
            if isinstance(val, dict) and "segments" in val:
                segments.extend(val["segments"])
            elif isinstance(val, dict):
                segments.append(val)
    elif isinstance(data, list):
        segments = data

    return [s for s in segments if isinstance(s, dict)]


def _replay_mouse(driver, mouse_events: list[dict], max_events: int = 100):
    """Replay mouse movements with Gaussian noise and human-like timing."""
    if not mouse_events:
        return

    events = mouse_events[:max_events]
    actions = ActionChains(driver)
    prev_x, prev_y = None, None
    prev_t = 0

    for evt in events:
        x = evt.get("x", evt.get("pageX", 0))
        y = evt.get("y", evt.get("pageY", 0))
        t = evt.get("t", evt.get("timestamp", 0))

        if x is None or y is None:
            continue

        # Add human-like noise (Gaussian)
        x += random.gauss(0, 6)
        y += random.gauss(0, 4)
        x = max(10, min(1900, x))
        y = max(10, min(1060, y))

        if prev_x is not None:
            dx = int(x - prev_x)
            dy = int(y - prev_y)
            dx = max(-200, min(200, dx))
            dy = max(-200, min(200, dy))

            dt = (t - prev_t) / 1000 if prev_t else 0.015
            # Add timing jitter
            dt = max(0.005, min(0.3, dt)) * random.uniform(0.7, 1.3)

            if dx != 0 or dy != 0:
                actions.move_by_offset(dx, dy)
                actions.pause(dt)

                # Occasional micro-correction (human steadying hand)
                if random.random() < 0.1:
                    actions.move_by_offset(
                        random.randint(-2, 2), random.randint(-2, 2)
                    )
                    actions.pause(random.uniform(0.01, 0.03))

        prev_x, prev_y, prev_t = x, y, t

    try:
        actions.perform()
    except Exception as e:
        print(f"  Replay mouse error (non-fatal): {e}")


def _replay_scroll(driver, scroll_events: list[dict], max_events: int = 10):
    """Replay scroll events with human-like timing."""
    if not scroll_events:
        return

    for evt in scroll_events[:max_events]:
        dy = evt.get("dy", evt.get("deltaY", 0))
        if dy is None or int(dy) == 0:
            continue

        # Add noise to scroll amount
        dy = int(dy * random.uniform(0.8, 1.2))
        driver.execute_script(f"window.scrollBy(0, {dy});")
        time.sleep(random.uniform(0.05, 0.2))


# ---------------------------------------------------------------------------
# Auto-export and RL confirmation
# ---------------------------------------------------------------------------

def _get_session_id(driver) -> str | None:
    """Read the tracking session ID from the browser's sessionStorage."""
    try:
        return driver.execute_script(
            "return window.sessionStorage.getItem('tm_session_id');"
        )
    except Exception:
        return None


def _export_and_confirm(driver, run_index: int) -> None:
    """Pull telemetry from backend, save to data/bot/, confirm as bot."""
    import urllib.request

    # Wait for tracking.js to flush
    print("  Waiting for telemetry flush...")
    time.sleep(8)

    session_id = _get_session_id(driver)
    if not session_id:
        print("  WARNING: Could not get session ID from browser")
        return

    print(f"  Session ID: {session_id}")

    # Pull raw telemetry
    try:
        url = f"{API_URL}/api/session/raw/{session_id}"
        with urllib.request.urlopen(url, timeout=15) as resp:
            raw = json.loads(resp.read().decode())
    except Exception as e:
        print(f"  WARNING: Could not fetch telemetry: {e}")
        return

    if not raw.get("success"):
        print("  WARNING: No telemetry in backend")
        return

    mouse = raw.get("mouse", [])
    clicks = raw.get("clicks", [])
    keystrokes = raw.get("keystrokes", [])
    scroll = raw.get("scroll", [])
    total = len(mouse) + len(clicks) + len(keystrokes) + len(scroll)

    if total == 0:
        print("  WARNING: 0 events captured")
        return

    print(f"  Events: {len(mouse)} mouse, {len(clicks)} clicks, "
          f"{len(keystrokes)} keystrokes, {len(scroll)} scroll")

    # Save JSON
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    filename = f"telemetry_export_{ts}_selenium_run{run_index}.json"
    out_path = DATA_DIR / filename

    consolidated = {
        session_id: {
            "sessionId": session_id,
            "startTime": int(time.time() * 1000),
            "pageMeta": [],
            "totalSegments": 1,
            "segments": [{
                "segmentId": 1, "url": SITE_URL, "hostname": "localhost",
                "startTime": int(time.time() * 1000),
                "endTime": int(time.time() * 1000),
                "mouse": mouse, "clicks": clicks,
                "keystrokes": keystrokes, "scroll": scroll,
            }],
        }
    }
    with open(out_path, "w") as f:
        json.dump(consolidated, f, indent=2)
    print(f"  Saved: {filename} ({total} events)")

    # Confirm as bot for RL online learning
    print("  Confirming bot label + triggering RL update...")
    try:
        body = json.dumps({"session_id": session_id, "true_label": 0}).encode()
        req = urllib.request.Request(
            f"{API_URL}/api/agent/confirm", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode())
        if result.get("updated"):
            metrics = result.get("metrics", {})
            print(f"  RL agent updated! (loss: {metrics.get('policy_loss', '?')}, steps: {result.get('steps', '?')})")
        else:
            print(f"  RL confirmed (no update: {result.get('reason', '?')})")
    except Exception as e:
        print(f"  WARNING: Could not confirm with RL agent: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run bot against TicketMonarch")
    parser.add_argument("--runs", type=int, default=3, help="Number of bot sessions")
    parser.add_argument("--type", choices=["linear", "scripted", "replay"], default="scripted")
    parser.add_argument("--replay-source", type=str, help="JSON file for replay bot")
    parser.add_argument("--pause-between", type=float, default=2.0, help="Seconds between runs")
    args = parser.parse_args()

    print(f"Selenium Bot — {args.runs} {args.type} runs")
    print(f"Target: {SITE_URL}")
    print(f"Output: {DATA_DIR}")
    print()
    print("Make sure backend (python app.py) and frontend (npm run dev) are running!")
    print()

    driver = create_driver()

    try:
        for i in range(args.runs):
            print(f"\n{'='*50}")
            print(f"Run {i + 1}/{args.runs} ({args.type})")
            print(f"{'='*50}")

            try:
                if args.type == "linear":
                    linear_bot(driver)
                elif args.type == "scripted":
                    scripted_bot(driver)
                elif args.type == "replay":
                    if not args.replay_source:
                        print("Error: --replay-source required for replay bot")
                        return
                    replay_bot(driver, args.replay_source)
                print(f"  Run {i + 1} complete.")
            except Exception as e:
                print(f"  Run {i + 1} failed: {e}")

            # Auto-export telemetry and confirm as bot
            _export_and_confirm(driver, i + 1)

            if i < args.runs - 1:
                print(f"  Waiting {args.pause_between}s...")
                time.sleep(args.pause_between)

        print(f"\n{'='*50}")
        print("All runs complete!")
        print(f"Telemetry saved to: {DATA_DIR}")
        print("="*50)
        input("Press Enter to close the browser...")

    finally:
        driver.quit()


if __name__ == "__main__":
    main()
