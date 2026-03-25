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
    python bots/selenium_bot.py --runs 3 --type replay --replay-source data/human/session_example.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait


SITE_URL = "http://localhost:3000"
API_URL = "http://localhost:5000"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "bot"

# Bot type → adversarial tier (must match rl_captcha.data.loader.BOT_TYPE_TO_TIER)
BOT_TIER: dict[str, int] = {
    "linear": 1, "tabber": 1, "speedrun": 1,
    "scripted": 2, "stealth": 2, "slow": 2, "erratic": 2, "replay": 2,
    "semi_auto": 3, "trace_conditioned": 4,
}

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

def _type_with_hold(driver, element, text, hold_min=0.04, hold_max=0.12,
                     gap_min=0.03, gap_max=0.25, think_prob=0.08,
                     think_range=(0.3, 0.8)):
    """Type with realistic key hold durations using JS keydown/keyup events.

    Selenium's send_keys() fires keydown+keyup instantly (0ms hold time),
    which makes mean_key_hold ≈ 0 — a dead giveaway.  This function dispatches
    separate keydown and keyup events with a realistic gap between them.
    """
    element.click()
    # Clear existing value via JS (element.clear() can reset React state weirdly)
    driver.execute_script(
        "arguments[0].value = ''; arguments[0].dispatchEvent(new Event('input', {bubbles:true}));",
        element,
    )
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
            c = text[i + j]
            # Dispatch keydown, wait (hold), then keyup — tracking.js measures this gap
            hold_time = random.uniform(hold_min, hold_max)
            driver.execute_script("""
                var el = arguments[0], key = arguments[1], holdMs = arguments[2];
                el.dispatchEvent(new KeyboardEvent('keydown', {key: key, code: 'Key'+key.toUpperCase(), bubbles: true}));
                // Update value to include new char (React controlled input)
                var nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
                nativeSetter.call(el, el.value + key);
                el.dispatchEvent(new Event('input', {bubbles: true}));
            """, element, c)
            time.sleep(hold_time)
            driver.execute_script("""
                var el = arguments[0], key = arguments[1];
                el.dispatchEvent(new KeyboardEvent('keyup', {key: key, code: 'Key'+key.toUpperCase(), bubbles: true}));
            """, element, c)
            # Fast within burst
            time.sleep(random.uniform(0.02, 0.06))

        i += burst_len

        # Inter-key delay
        if i < len(text):
            if random.random() < think_prob:
                time.sleep(random.uniform(*think_range))
            else:
                delay = random.lognormvariate(math.log(0.08), 0.4)
                delay = max(gap_min, min(gap_max, delay))
                time.sleep(delay)


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
    """Uniform typing with slight per-run variance — still robotic but not identical."""
    element.clear()
    base_delay = random.uniform(0.015, 0.035)
    for char in text:
        element.send_keys(char)
        delay = max(0.005, base_delay + random.gauss(0, 0.005))
        time.sleep(delay)
        # Occasional micro-pause (e.g. switching mental focus)
        if random.random() < 0.05:
            time.sleep(random.uniform(0.05, 0.15))


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
    """Straight-line move + click with intermediate mouse events.

    Instead of teleporting (which produces zero mouse speed/accel in
    telemetry), we step toward the target with small offsets so that
    tracking.js actually captures mouse movement.
    """
    if click_only:
        ActionChains(driver).move_to_element(element).click().perform()
        time.sleep(random.uniform(0.05, 0.15))
        return

    steps = random.randint(6, 14)
    actions = ActionChains(driver)

    # Move toward target in steps: start offset far from element, converge
    for i in range(steps):
        remaining = steps - i - 1
        # Offsets shrink to zero as we approach the element
        off_x = int(remaining * random.uniform(-8, 8))
        off_y = int(remaining * random.uniform(-5, 5))
        actions.move_to_element_with_offset(element, off_x, off_y)
        # Uniform-ish speed (slightly robotic but generates real events)
        actions.pause(random.uniform(0.015, 0.04))

    actions.move_to_element(element)
    actions.click()
    try:
        actions.perform()
    except Exception:
        # Fallback: direct click
        ActionChains(driver).move_to_element(element).click().perform()
    time.sleep(random.uniform(0.05, 0.2))


def _idle_fidget(driver, duration: float = None):
    """Simulate idle mouse fidgeting — small drifts and micro-movements that
    humans naturally make while reading, thinking, or waiting.

    This is the KEY missing behavior that makes bot telemetry trivially
    detectable: real humans never hold the mouse perfectly still.
    """
    if duration is None:
        duration = random.uniform(0.2, 0.7)

    elapsed = 0.0
    while elapsed < duration:
        # Pick a fidget behavior
        behavior = random.choices(
            ["drift", "jitter", "circle", "pause"],
            weights=[0.35, 0.30, 0.15, 0.20],
        )[0]

        if behavior == "drift":
            # Slow drift in a random direction (reading, scanning)
            steps = random.randint(4, 9)
            dx_bias = random.gauss(0, 35)
            dy_bias = random.gauss(0, 22)
            actions = ActionChains(driver)
            for _ in range(steps):
                dx = int(dx_bias / steps + random.gauss(0, 2))
                dy = int(dy_bias / steps + random.gauss(0, 2))
                if dx != 0 or dy != 0:
                    actions.move_by_offset(dx, dy)
                pause = random.uniform(0.02, 0.06)
                actions.pause(pause)
                elapsed += pause
            try:
                actions.perform()
            except Exception:
                pass

        elif behavior == "jitter":
            # Tremor-like movements (hand not perfectly steady)
            actions = ActionChains(driver)
            jitter_count = random.randint(2, 5)
            for _ in range(jitter_count):
                dx = random.randint(-6, 6)
                dy = random.randint(-4, 4)
                if dx != 0 or dy != 0:
                    actions.move_by_offset(dx, dy)
                pause = random.uniform(0.01, 0.04)
                actions.pause(pause)
                elapsed += pause
            try:
                actions.perform()
            except Exception:
                pass

        elif behavior == "circle":
            # Circular/arc movement (hovering indecisively)
            actions = ActionChains(driver)
            radius = random.uniform(12, 35)
            arc_steps = random.randint(4, 8)
            start_angle = random.uniform(0, 2 * math.pi)
            for i in range(arc_steps):
                angle = start_angle + (i / arc_steps) * math.pi * random.uniform(0.5, 1.5)
                dx = int(radius * math.cos(angle) / arc_steps * 2)
                dy = int(radius * math.sin(angle) / arc_steps * 2)
                if dx != 0 or dy != 0:
                    actions.move_by_offset(dx, dy)
                pause = random.uniform(0.01, 0.05)
                actions.pause(pause)
                elapsed += pause
            try:
                actions.perform()
            except Exception:
                pass

        else:  # pause
            # Brief stillness (but not too long — humans fidget again quickly)
            pause = random.uniform(0.08, 0.25)
            time.sleep(pause)
            elapsed += pause


def _page_sweep(driver):
    """Move mouse across a large area of the page — simulates a human visually
    scanning the page (looking at header, sidebar, content, footer).

    This is critical for spatial diversity: humans don't keep the mouse near
    one element — they sweep across hundreds of pixels while browsing.
    """
    # Keep sweeps small so bot sessions do not dwarf real checkout sessions.
    num_targets = random.randint(1, 2)
    actions = ActionChains(driver)

    for _ in range(num_targets):
        # Random point across the full viewport
        target_x = random.randint(-180, 180)
        target_y = random.randint(-120, 120)

        # Move there in a Bezier-like curve with multiple steps
        steps = random.randint(5, 10)
        curve_x = random.uniform(-25, 25)
        curve_y = random.uniform(-18, 18)

        for i in range(steps):
            t = (i + 1) / steps
            t_eased = t * t * (3 - 2 * t)
            remaining = 1 - t_eased

            dx = int(target_x / steps + curve_x * remaining * math.sin(t * math.pi) / steps
                     + random.gauss(0, 3))
            dy = int(target_y / steps + curve_y * remaining * math.sin(t * math.pi) / steps
                     + random.gauss(0, 2))
            if dx != 0 or dy != 0:
                actions.move_by_offset(dx, dy)
            actions.pause(random.uniform(0.01, 0.02))

        # Pause at the target (reading/looking)
        actions.pause(random.uniform(0.08, 0.25))

    try:
        actions.perform()
    except Exception:
        pass


def _wander_mouse(driver, duration=None):
    """Continuous mouse movement across the page — simulates a human reading,
    scanning, or just idly moving the mouse while thinking.

    This is critical because humans have mean_mouse_spd ≈ 0.098 but bots are
    at 0.004 — bots only move mouse to click elements, humans move it constantly.
    """
    if duration is None:
        duration = random.uniform(0.8, 2.5)

    elapsed = 0.0
    while elapsed < duration:
        actions = ActionChains(driver)
        # Pick a random movement style
        style = random.choices(["sweep", "drift", "read_scan"], weights=[0.3, 0.4, 0.3])[0]

        if style == "sweep":
            # Broad sweep across page (like scanning a section)
            dx_total = random.randint(-300, 300)
            dy_total = random.randint(-200, 200)
            steps = random.randint(8, 20)
            for s in range(steps):
                t = (s + 1) / steps
                # Ease-in-out
                t_eased = t * t * (3 - 2 * t)
                dx = int(dx_total / steps + random.gauss(0, 4))
                dy = int(dy_total / steps + random.gauss(0, 3))
                if dx != 0 or dy != 0:
                    actions.move_by_offset(dx, dy)
                pause = random.uniform(0.015, 0.04)
                actions.pause(pause)
                elapsed += pause

        elif style == "drift":
            # Slow aimless drift (mouse following eyes)
            steps = random.randint(6, 15)
            for _ in range(steps):
                dx = int(random.gauss(0, 20))
                dy = int(random.gauss(0, 15))
                if dx != 0 or dy != 0:
                    actions.move_by_offset(dx, dy)
                pause = random.uniform(0.03, 0.08)
                actions.pause(pause)
                elapsed += pause

        else:  # read_scan
            # Horizontal scanning (like reading text left to right)
            line_width = random.randint(100, 400)
            steps = random.randint(8, 16)
            for s in range(steps):
                dx = int(line_width / steps + random.gauss(0, 3))
                dy = int(random.gauss(0, 2))  # mostly horizontal
                if dx != 0 or dy != 0:
                    actions.move_by_offset(dx, dy)
                pause = random.uniform(0.02, 0.05)
                actions.pause(pause)
                elapsed += pause
            # Return-sweep (like going to next line)
            actions.move_by_offset(-int(line_width * 0.8), random.randint(15, 35))
            actions.pause(random.uniform(0.05, 0.1))
            elapsed += 0.1

        try:
            actions.perform()
        except Exception:
            pass


def _random_page_click(driver, count=1):
    """Click on non-interactive areas of the page. Humans click randomly on
    empty space, text, headers, etc. — not just form elements.

    Bot interactive_click ratio is 0.875 vs human 0.490, meaning bots only
    click on buttons/inputs. These random clicks dilute that ratio.
    """
    # Target safe non-interactive elements to avoid accidentally clicking
    # stepper buttons, disabled buttons, or form controls.
    safe_selectors = [
        "h1", "h2", "h3", "p", "header", ".logo",
        ".ss-panel-header", ".ss-legend", ".ss-event-bar",
        ".ss-stadium-note", ".ss-stage",
    ]
    for _ in range(count):
        try:
            # Try to click a known safe element first
            safe_targets = []
            for sel in safe_selectors:
                safe_targets.extend(driver.find_elements(By.CSS_SELECTOR, sel))
            if safe_targets:
                target = random.choice(safe_targets)
                ActionChains(driver).move_to_element(target).click().perform()
            else:
                # Fallback: click on body at random coordinates
                body = driver.find_element(By.TAG_NAME, "body")
                x_off = random.randint(-400, 400)
                y_off = random.randint(-200, 200)
                ActionChains(driver).move_to_element_with_offset(body, x_off, y_off).click().perform()
            time.sleep(random.uniform(0.1, 0.3))
        except Exception:
            pass


def _dispatch_wheel(driver, dy: int):
    """Scroll by dispatching a real WheelEvent so tracking.js captures it."""
    driver.execute_script("""
        var dy = arguments[0];
        window.dispatchEvent(new WheelEvent('wheel', {
            deltaY: dy, deltaMode: 0, bubbles: true, cancelable: true
        }));
        window.scrollBy(0, dy);
    """, dy)


def _random_scroll(driver, scrolls: int = 2):
    """Scroll up and down randomly to simulate browsing."""
    for _ in range(scrolls):
        dy = random.randint(100, 400) * random.choice([1, -1])
        _dispatch_wheel(driver, dy)
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
            _dispatch_wheel(driver, step_dy)
            time.sleep(random.uniform(0.02, 0.08))

        # Pause between scroll gestures
        time.sleep(random.uniform(0.5, 1.5))


# ---------------------------------------------------------------------------
# Shared multi-page flow steps
# ---------------------------------------------------------------------------

def _go_home(driver):
    """Navigate to the home page and wait for concert cards to load.

    Clears sessionStorage first so tracking.js creates a fresh session ID.
    Without this, sessionStorage persists across page reloads in the same
    tab, causing every run to reuse the previous session ID and accumulate
    telemetry into one giant session.
    """
    # Load page first so we have a JS context to clear storage
    driver.get(SITE_URL)
    driver.execute_script("window.sessionStorage.clear();")
    # Reload — tracking.js will see no tm_session_id and generate a fresh one
    driver.get(SITE_URL)
    wait_for(driver, ".tickets-button", timeout=10)
    # Mark this session as a bot so Confirmation.jsx won't auto-confirm as human
    driver.execute_script("window.sessionStorage.setItem('tm_is_bot', '1');")


def _pick_concert(driver, move_fn):
    """Click a concert's 'Tickets' button on the home page."""
    for _ in range(3):
        buttons = driver.find_elements(By.CSS_SELECTOR, ".tickets-button")
        if not buttons:
            print("  WARNING: No .tickets-button found on home page")
            return False
        target = random.choice(buttons)
        try:
            move_fn(driver, target)
            wait_for_url(driver, "/seats/")
            return True
        except StaleElementReferenceException:
            time.sleep(0.5)
    print("  WARNING: Failed to click concert button after retries")
    return False


def _pick_section(driver, move_fn):
    """Select tickets on the seat selection page, then click Checkout.

    The UI uses a stadium grid with stepper buttons (+/-) per section.
    Mimics human behavior: browse sections visually, pick 1-2 sections,
    add tickets with natural pauses. Sometimes change mind (add then remove).
    Never clicks minus on a section with 0 tickets.
    """
    wait_for(driver, ".ss-section-cell", timeout=10)
    cells = driver.find_elements(By.CSS_SELECTOR, ".ss-section-cell")
    if not cells:
        print("  WARNING: No .ss-section-cell found")
        return False

    # --- Browse phase: hover over a few sections like a human scanning ---
    browse_count = random.randint(1, 4)
    browse_targets = random.sample(cells, min(browse_count, len(cells)))
    for cell in browse_targets:
        try:
            ActionChains(driver).move_to_element(cell).perform()
            time.sleep(random.uniform(0.2, 0.6))
        except Exception:
            pass

    # --- Selection phase: pick 1-2 sections (rarely 3) ---
    num_sections = random.choices([1, 2, 3], weights=[0.45, 0.45, 0.10])[0]
    num_sections = min(num_sections, len(cells))
    chosen = random.sample(cells, num_sections)

    sections_with_tickets = []  # track which cells we added tickets to

    for cell in chosen:
        try:
            btns = cell.find_elements(By.CSS_SELECTOR, ".ss-step-btn")
            if len(btns) < 2:
                continue
            minus_btn, plus_btn = btns[0], btns[1]

            # Add 1-4 tickets (weighted toward 1-2 like a real person)
            num_tickets = random.choices([1, 2, 3, 4], weights=[0.35, 0.35, 0.20, 0.10])[0]
            for i in range(num_tickets):
                move_fn(driver, plus_btn)
                # Slightly longer pause on first click (deciding), shorter after
                if i == 0:
                    time.sleep(random.uniform(0.3, 0.7))
                else:
                    time.sleep(random.uniform(0.12, 0.35))

            sections_with_tickets.append((cell, minus_btn, num_tickets))

            # Small pause between sections
            time.sleep(random.uniform(0.3, 0.8))

        except StaleElementReferenceException:
            print("  WARNING: Stale element in seat selection, skipping section")
            continue

    # --- Occasionally change mind: remove tickets from one section ---
    if len(sections_with_tickets) > 1 and random.random() < 0.15:
        # Remove from a random section (only click minus if qty > 0)
        remove_cell, minus_btn, qty = random.choice(sections_with_tickets)
        try:
            remove_count = random.randint(1, qty)
            for _ in range(remove_count):
                move_fn(driver, minus_btn)
                time.sleep(random.uniform(0.15, 0.35))
        except Exception:
            pass

    # Click the checkout button
    try:
        checkout_btn = wait_for(driver, ".ss-checkout-btn", timeout=5)
        move_fn(driver, checkout_btn)
        wait_for_url(driver, "/checkout")
        return True
    except Exception:
        print("  WARNING: Could not click checkout button")
        return False


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
        try:
            go_back = driver.find_elements(By.CSS_SELECTOR, ".challenge-overlay .challenge-btn")
            if go_back:
                btn_text = go_back[0].text.strip().lower()
                if btn_text == "go back":
                    print("  Blocked by agent — clicking Go Back")
                    go_back[0].click()
                    time.sleep(1)
                    return
        except StaleElementReferenceException:
            continue

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


def _fill_checkout(driver, type_fn, move_fn, skip_honeypot=False):
    """Fill out the checkout form and submit.

    Fills every input field found on the page. Known fields get realistic
    fake data; any unknown fields (e.g. hidden honeypots) get generic
    filler — just like a real scraper bot that parses the DOM.

    If skip_honeypot=True, only visible known fields are filled (skips
    hidden/unknown fields). This forces the RL agent to make the detection
    decision instead of the honeypot short-circuiting it.
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

    for i, inp in enumerate(all_inputs):
        try:
            field_id = inp.get_attribute("id") or inp.get_attribute("name") or ""

            # Skip fields that are part of dropdowns or already filled
            if not field_id:
                continue

            # Determine value: use known data if available, generic filler otherwise
            value = known_values.get(field_id)
            if value is None:
                if skip_honeypot:
                    # Smart bot: skip unknown fields (avoids honeypots)
                    continue
                # Dumb bot: fill with generic data (this catches honeypots)
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

            # Idle fidget between fields (human reads next label, glances at card)
            if random.random() < 0.4:
                _idle_fidget(driver, random.uniform(0.3, 1.0))
            else:
                time.sleep(random.uniform(0.2, 0.6))
        except StaleElementReferenceException:
            # Re-find input fields and retry this one
            try:
                refreshed = driver.find_elements(By.CSS_SELECTOR, "input[type='text'], input[type='tel'], input[type='email'], input:not([type])")
                if i < len(refreshed):
                    inp = refreshed[i]
                    field_id = inp.get_attribute("id") or inp.get_attribute("name") or ""
                    value = known_values.get(field_id, GENERIC_FILLERS[filler_idx % len(GENERIC_FILLERS)])
                    if value and inp.is_displayed():
                        move_fn(driver, inp, click_only=True)
                        type_fn(inp, value)
            except Exception:
                pass
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

    # Capture session ID BEFORE clicking Purchase — the Confirmation page
    # calls resetSession() which replaces it with a new UUID immediately.
    captured_sid = _get_session_id(driver)

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

    return captured_sid


def _fill_checkout_targeted(driver, type_fn, move_fn, skip_honeypot=False):
    """Fill checkout by targeting known fields by ID — like a human would.

    Unlike _fill_checkout which sweeps querySelectorAll and fills every input
    (a telltale bot pattern), this fills only the fields a human would see
    and interact with, in a natural top-to-bottom order with occasional
    skips/revisits.
    """
    form = get_form_data()
    wait_for(driver, "#card_number", timeout=10)

    # Fields in visual order (as a human would see them on the page)
    field_order = [
        ("full_name", form["full_name"]),
        ("billing_address", form["billing_address"]),
        ("apartment", form["apartment"]),
        ("city", form["city"]),
        ("zip_code", form["zip_code"]),
        ("card_number", form["card_number"]),
        ("card_expiry", form["card_expiry"]),
        ("card_cvv", form["card_cvv"]),
    ]

    # Track all known checkout field IDs (including ones we skip)
    # so the honeypot pass doesn't accidentally fill them with garbage.
    all_known_ids = {fid for fid, _ in field_order}

    # Occasionally skip apartment (humans often leave it blank)
    if not form["apartment"] or random.random() < 0.3:
        field_order = [(fid, val) for fid, val in field_order if fid != "apartment"]

    # Occasionally fill out of order (human glances at card first, etc.)
    if random.random() < 0.25:
        # Move card fields to the front
        card_fields = [(f, v) for f, v in field_order if f.startswith("card_")]
        other_fields = [(f, v) for f, v in field_order if not f.startswith("card_")]
        field_order = card_fields + other_fields

    for field_id, value in field_order:
        if not value:
            continue
        try:
            inp = driver.find_element(By.ID, field_id)
            if not inp.is_displayed():
                continue  # skip hidden fields naturally
            move_fn(driver, inp, click_only=True)
            type_fn(inp, value)

            # Human-like pauses between fields
            if random.random() < 0.4:
                _idle_fidget(driver, random.uniform(0.3, 1.0))
            else:
                time.sleep(random.uniform(0.2, 0.6))
        except Exception:
            pass

    # If NOT skipping honeypot, also fill any unknown visible fields
    # (dumb bots still fill everything they find)
    if not skip_honeypot:
        all_inputs = driver.find_elements(By.CSS_SELECTOR,
            "input[type='text'], input[type='tel'], input[type='email'], input:not([type])")
        GENERIC_FILLERS = [
            "test@email.com", "5551234567", "John Doe", "123 Main St",
            "Springfield", "12345", "some value",
        ]
        filler_idx = 0
        for inp in all_inputs:
            try:
                fid = inp.get_attribute("id") or inp.get_attribute("name") or ""
                if not fid or fid in all_known_ids:
                    continue
                value = GENERIC_FILLERS[filler_idx % len(GENERIC_FILLERS)]
                filler_idx += 1
                if inp.is_displayed():
                    move_fn(driver, inp, click_only=True)
                    type_fn(inp, value)
                else:
                    # Hidden field — fill via JS (honeypot trap)
                    driver.execute_script("""
                        var el = arguments[0], value = arguments[1];
                        var setter = Object.getOwnPropertyDescriptor(
                            window.HTMLInputElement.prototype, 'value').set;
                        setter.call(el, value);
                        el.dispatchEvent(new Event('input', {bubbles: true}));
                        el.dispatchEvent(new Event('change', {bubbles: true}));
                    """, inp, value)
                time.sleep(random.uniform(0.1, 0.3))
            except Exception:
                pass

    # Select state dropdown
    try:
        state_el = driver.find_element(By.ID, "state")
        move_fn(driver, state_el, click_only=True)
        select = Select(state_el)
        select.select_by_visible_text(form["state"])
        time.sleep(random.uniform(0.3, 0.6))
    except Exception as e:
        print(f"  WARNING: Could not select state: {e}")

    # Capture session ID BEFORE clicking Purchase
    captured_sid = _get_session_id(driver)

    # Click Purchase
    try:
        purchase = wait_for(driver, ".purchase-button", timeout=5)
        move_fn(driver, purchase)
    except Exception as e:
        print(f"  WARNING: Could not click Purchase: {e}")

    try:
        wait_for_url(driver, "/confirmation", timeout=5)
    except Exception:
        _handle_challenge(driver, move_fn)

    return captured_sid


# ---------------------------------------------------------------------------
# Bot behaviors
# ---------------------------------------------------------------------------

def linear_bot(driver, skip_honeypot=False):
    """Straight-line mouse, uniform typing with slight variance.
    Obviously robotic — easy negative for the RL agent."""
    _go_home(driver)
    _wander_mouse(driver, random.uniform(0.5, 1.0))
    _random_page_click(driver, random.randint(0, 1))
    _random_scroll(driver, scrolls=random.randint(1, 2))
    time.sleep(random.uniform(0.2, 0.5))

    if not _pick_concert(driver, _linear_move_and_click):
        return
    _wander_mouse(driver, random.uniform(0.3, 0.8))
    _random_scroll(driver, scrolls=1)
    time.sleep(random.uniform(0.2, 0.4))

    if not _pick_section(driver, _linear_move_and_click):
        return
    _wander_mouse(driver, random.uniform(0.2, 0.5))
    _random_page_click(driver, random.randint(0, 1))
    time.sleep(random.uniform(0.1, 0.3))

    return _fill_checkout(driver, _type_uniform, _linear_move_and_click, skip_honeypot=skip_honeypot)


def scripted_bot(driver, skip_honeypot=False):
    """Bezier curve mouse, human-like typing with key hold, scrolling. More sophisticated."""
    _go_home(driver)
    _wander_mouse(driver, random.uniform(0.8, 1.5))
    _random_page_click(driver, random.randint(1, 2))

    # Browse around first
    _human_scroll(driver, scrolls=random.randint(2, 4))
    _idle_fidget(driver, random.uniform(0.3, 0.8))

    if not _pick_concert(driver, _human_move_and_click):
        return
    _wander_mouse(driver, random.uniform(0.5, 1.0))
    _random_page_click(driver, random.randint(1, 2))

    # Look at seats
    _human_scroll(driver, scrolls=random.randint(1, 3))
    _idle_fidget(driver, random.uniform(0.3, 0.8))

    if not _pick_section(driver, _human_move_and_click):
        return
    _wander_mouse(driver, random.uniform(0.3, 0.8))
    _random_page_click(driver, random.randint(0, 1))
    _human_scroll(driver, scrolls=1)

    # Use key-hold typing for ~50% of scripted bots
    if random.random() < 0.5:
        def _type_scripted(element, text):
            _type_with_hold(driver, element, text)
        return _fill_checkout_targeted(driver, _type_scripted, _human_move_and_click, skip_honeypot=skip_honeypot)

    return _fill_checkout_targeted(driver, _type_human, _human_move_and_click, skip_honeypot=skip_honeypot)


def tabber_bot(driver, skip_honeypot=False):
    """Keyboard-only bot — navigates entirely via Tab/Enter, no mouse at all.
    Easy to detect: zero mouse events, perfectly regular key timing."""
    _go_home(driver)
    # Even keyboard users move mouse a bit while reading
    _wander_mouse(driver, random.uniform(0.3, 0.8))
    _random_scroll(driver, scrolls=1)
    time.sleep(random.uniform(0.5, 1.0))

    # Tab to a tickets button and press Enter
    body = driver.find_element(By.TAG_NAME, "body")
    tab_count = random.randint(5, 15)
    for _ in range(tab_count):
        body.send_keys(Keys.TAB)
        time.sleep(random.uniform(0.08, 0.15))
    body.send_keys(Keys.ENTER)

    try:
        wait_for_url(driver, "/seats/", timeout=5)
    except Exception:
        # Fallback: click directly
        if not _pick_concert(driver, _linear_move_and_click):
            return

    time.sleep(random.uniform(0.5, 1.0))

    # Tab to section + continue
    for _ in range(random.randint(3, 8)):
        body.send_keys(Keys.TAB)
        time.sleep(random.uniform(0.08, 0.15))
    body.send_keys(Keys.ENTER)
    time.sleep(0.5)

    for _ in range(random.randint(2, 5)):
        body.send_keys(Keys.TAB)
        time.sleep(random.uniform(0.08, 0.15))
    body.send_keys(Keys.ENTER)

    try:
        wait_for_url(driver, "/checkout", timeout=5)
    except Exception:
        if not _pick_section(driver, _linear_move_and_click):
            return

    time.sleep(random.uniform(0.5, 1.0))
    return _fill_checkout(driver, _type_uniform, _linear_move_and_click, skip_honeypot=skip_honeypot)


def slow_bot(driver, skip_honeypot=False):
    """Slow methodical bot — moderate pauses between actions, very regular.
    Mimics a careful person but timing is unnaturally consistent."""
    _go_home(driver)
    _wander_mouse(driver, random.uniform(0.8, 1.5))
    _random_page_click(driver, random.randint(0, 1))
    time.sleep(random.uniform(0.5, 1.0))

    _human_scroll(driver, scrolls=1)
    time.sleep(random.uniform(0.5, 1.0))

    if not _pick_concert(driver, _human_move_and_click):
        return
    _wander_mouse(driver, random.uniform(0.5, 1.0))
    time.sleep(random.uniform(0.5, 1.0))

    _human_scroll(driver, scrolls=1)
    _random_page_click(driver, random.randint(0, 1))
    time.sleep(random.uniform(0.5, 1.0))

    if not _pick_section(driver, _human_move_and_click):
        return
    _wander_mouse(driver, random.uniform(0.3, 0.8))
    time.sleep(random.uniform(0.5, 1.0))

    return _fill_checkout_targeted(driver, _type_human, _human_move_and_click, skip_honeypot=skip_honeypot)


def erratic_bot(driver, skip_honeypot=False):
    """Erratic bot — random mouse movements everywhere, clicks randomly,
    eventually finds the right elements. High spatial diversity but
    unnatural patterns (no purposeful movement toward targets)."""
    _go_home(driver)

    # Thrash mouse around randomly
    for _ in range(random.randint(3, 6)):
        _page_sweep(driver)
        _random_scroll(driver, scrolls=random.randint(1, 3))
        time.sleep(random.uniform(0.1, 0.3))

    # Random clicks on whatever is nearby
    actions = ActionChains(driver)
    for _ in range(random.randint(3, 8)):
        actions.move_by_offset(random.randint(-200, 200), random.randint(-150, 150))
        actions.click()
        actions.pause(random.uniform(0.1, 0.3))
    try:
        actions.perform()
    except Exception:
        pass

    time.sleep(random.uniform(0.3, 0.8))

    if not _pick_concert(driver, _human_move_and_click):
        return

    # More thrashing on seats page
    _random_page_click(driver, random.randint(1, 3))
    for _ in range(random.randint(2, 4)):
        _page_sweep(driver)
        time.sleep(random.uniform(0.1, 0.3))

    if not _pick_section(driver, _human_move_and_click):
        return

    # Erratic checkout — fidget excessively between fields
    _random_page_click(driver, random.randint(1, 2))
    return _fill_checkout_targeted(driver, _type_human, _human_move_and_click, skip_honeypot=skip_honeypot)


def speedrun_bot(driver, skip_honeypot=False):
    """Speed-run bot — completes the entire flow as fast as possible.
    Minimal mouse movement, instant typing, near-zero pauses.
    Very easy to detect: session duration is unnaturally short."""
    _go_home(driver)
    _wander_mouse(driver, random.uniform(0.2, 0.5))
    _random_scroll(driver, scrolls=1)
    time.sleep(0.2)

    if not _pick_concert(driver, _linear_move_and_click):
        return
    time.sleep(0.2)

    if not _pick_section(driver, _linear_move_and_click):
        return
    time.sleep(0.2)

    # Instant typing — machine speed
    form = get_form_data()
    wait_for(driver, "#card_number", timeout=10)

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

    GENERIC_FILLERS = [
        "test@email.com", "5551234567", "John Doe", "123 Main St",
        "Springfield", "12345", "some value",
    ]

    all_inputs = driver.find_elements(By.CSS_SELECTOR,
        "input[type='text'], input[type='tel'], input[type='email'], input:not([type])")
    filler_idx = 0

    for inp in all_inputs:
        try:
            field_id = inp.get_attribute("id") or inp.get_attribute("name") or ""
            if not field_id:
                continue
            value = known_values.get(field_id)
            if value is None:
                if skip_honeypot:
                    continue
                value = GENERIC_FILLERS[filler_idx % len(GENERIC_FILLERS)]
                filler_idx += 1
            if not value:
                continue

            if not inp.is_displayed():
                driver.execute_script("""
                    var el = arguments[0]; var value = arguments[1];
                    var setter = Object.getOwnPropertyDescriptor(
                        window.HTMLInputElement.prototype, 'value').set;
                    setter.call(el, value);
                    el.dispatchEvent(new Event('input', { bubbles: true }));
                    el.dispatchEvent(new Event('change', { bubbles: true }));
                """, inp, value)
            else:
                inp.click()
                # Blast all chars at once — inhuman speed
                for char in value:
                    inp.send_keys(char)
                    time.sleep(random.uniform(0.005, 0.015))
            time.sleep(random.uniform(0.05, 0.1))
        except Exception:
            pass

    try:
        state_el = driver.find_element(By.ID, "state")
        Select(state_el).select_by_visible_text(form["state"])
        time.sleep(0.1)
    except Exception:
        pass

    # Capture session ID before Purchase (Confirmation resets it)
    captured_sid = _get_session_id(driver)

    try:
        purchase = wait_for(driver, ".purchase-button", timeout=5)
        purchase.click()
    except Exception:
        pass

    try:
        wait_for_url(driver, "/confirmation", timeout=5)
    except Exception:
        _handle_challenge(driver, _linear_move_and_click)

    return captured_sid


# ---------------------------------------------------------------------------
# Human data loading for stealth bot
# ---------------------------------------------------------------------------

HUMAN_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "human"


_cached_human_profiles: list[dict] | None = None


def _load_human_timing_profiles() -> list[dict]:
    """Load human session data and extract timing profiles (cached after first call).

    Returns a list of timing profiles, each containing:
    - mouse_intervals: list of inter-event gaps (ms) during mouse movement
    - click_intervals: list of inter-click gaps (ms)
    - key_intervals: list of inter-keystroke gaps (ms)
    - scroll_deltas: list of scroll dy values
    - scroll_intervals: list of inter-scroll gaps (ms)
    - duration: total session duration (ms)
    - event_counts: dict of event type counts
    """
    global _cached_human_profiles
    if _cached_human_profiles is not None:
        return _cached_human_profiles

    profiles = []
    if not HUMAN_DATA_DIR.exists():
        return profiles

    for json_file in sorted(HUMAN_DATA_DIR.glob("*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        # Handle live_confirm format (has "segments" at top level)
        segments = []
        if isinstance(data, dict):
            if "segments" in data:
                segments = data["segments"]
            else:
                for val in data.values():
                    if isinstance(val, dict) and "segments" in val:
                        segments.extend(val["segments"])
        if not segments:
            continue

        # Merge events across segments
        all_mouse = []
        all_clicks = []
        all_keys = []
        all_scroll = []
        for seg in segments:
            all_mouse.extend(seg.get("mouse", []))
            all_clicks.extend(seg.get("clicks", []))
            all_keys.extend(seg.get("keystrokes", []))
            all_scroll.extend(seg.get("scroll", []))

        # Extract timing intervals
        mouse_intervals = []
        for i in range(1, len(all_mouse)):
            t0 = all_mouse[i - 1].get("t", 0)
            t1 = all_mouse[i].get("t", 0)
            dt = t1 - t0
            if 0 < dt < 5000:  # filter outlier gaps
                mouse_intervals.append(dt)

        click_intervals = []
        for i in range(1, len(all_clicks)):
            t0 = all_clicks[i - 1].get("t", 0)
            t1 = all_clicks[i].get("t", 0)
            dt = t1 - t0
            if 0 < dt < 30000:
                click_intervals.append(dt)

        key_intervals = []
        key_downs = [k for k in all_keys if k.get("type") == "down"]
        for i in range(1, len(key_downs)):
            t0 = key_downs[i - 1].get("t", 0)
            t1 = key_downs[i].get("t", 0)
            dt = t1 - t0
            if 0 < dt < 5000:
                key_intervals.append(dt)

        scroll_deltas = [s.get("dy", 0) for s in all_scroll if s.get("dy")]
        scroll_intervals = []
        for i in range(1, len(all_scroll)):
            t0 = all_scroll[i - 1].get("t", 0)
            t1 = all_scroll[i].get("t", 0)
            dt = t1 - t0
            if 0 < dt < 5000:
                scroll_intervals.append(dt)

        # Session duration
        all_times = (
            [m.get("t", 0) for m in all_mouse]
            + [c.get("t", 0) for c in all_clicks]
            + [k.get("t", 0) for k in all_keys]
            + [s.get("t", 0) for s in all_scroll]
        )
        duration = max(all_times) - min(all_times) if all_times else 0

        profiles.append({
            "mouse_intervals": mouse_intervals,
            "click_intervals": click_intervals,
            "key_intervals": key_intervals,
            "scroll_deltas": scroll_deltas,
            "scroll_intervals": scroll_intervals,
            "duration": duration,
            "event_counts": {
                "mouse": len(all_mouse),
                "clicks": len(all_clicks),
                "keys": len(all_keys),
                "scroll": len(all_scroll),
            },
        })

    _cached_human_profiles = profiles
    return profiles


def _sample_from_human(intervals: list[float], fallback_min: float = 50,
                       fallback_max: float = 300) -> float:
    """Sample a timing value from real human intervals with noise.

    Picks a random value from the list, adds Gaussian jitter (±20%),
    and clamps to reasonable bounds. Falls back to uniform if no data.
    Returns seconds.
    """
    if not intervals:
        return random.uniform(fallback_min, fallback_max) / 1000.0

    base = random.choice(intervals)
    jittered = base * random.uniform(0.8, 1.2)
    clamped = max(5, min(500, jittered))
    return clamped / 1000.0


def stealth_bot(driver, skip_honeypot=False):
    """Stealth bot — designed to mimic real human behavior using timing
    profiles extracted from actual human sessions in data/human/.

    Uses realistic key hold times (JS keydown/keyup), continuous mouse
    wandering, non-interactive page clicks, human scroll patterns, and
    natural fidgeting/browsing behavior. Hardest bot type to detect.
    """
    # Load human timing profiles
    profiles = _load_human_timing_profiles()
    if profiles:
        profile = random.choice(profiles)
        print(f"  Using human timing profile ({profile['event_counts']})")
    else:
        profile = None
        print("  WARNING: No human data in data/human/, using synthetic timing")

    key_intervals = profile["key_intervals"] if profile else []

    # Randomly pick a persona: 60% quick decisive, 40% indecisive browser
    indecisive = random.random() < 0.4
    if indecisive:
        print("  Persona: indecisive browser")
    else:
        print("  Persona: quick decisive")

    # --- Page 1: Home ---
    _go_home(driver)

    # Continuous mouse movement while browsing (critical for mouse_spd/accel)
    _wander_mouse(driver, random.uniform(1.0, 2.0))
    _human_scroll(driver, scrolls=random.randint(2, 4))
    _idle_fidget(driver, random.uniform(0.3, 0.8))

    if indecisive:
        _wander_mouse(driver, random.uniform(0.8, 1.5))
        _human_scroll(driver, scrolls=random.randint(1, 2))
        _random_page_click(driver, 1)
        _idle_fidget(driver, random.uniform(0.5, 1.5))

    if not _pick_concert(driver, _human_move_and_click):
        return

    # --- Page 2: Seat Selection ---
    _wander_mouse(driver, random.uniform(0.8, 1.5))
    _human_scroll(driver, scrolls=random.randint(1, 3))
    _idle_fidget(driver, random.uniform(0.3, 0.8))

    if indecisive:
        _wander_mouse(driver, random.uniform(0.5, 1.2))
        _human_scroll(driver, scrolls=1)
        _random_page_click(driver, 1)

    if not _pick_section(driver, _human_move_and_click):
        return

    # --- Page 3: Checkout ---
    _wander_mouse(driver, random.uniform(0.5, 1.2))
    _human_scroll(driver, scrolls=random.randint(1, 2))

    # Typing with realistic key hold durations (JS keydown/sleep/keyup)
    think_prob = 0.12 if indecisive else 0.06
    think_range = (0.3, 0.8) if indecisive else (0.2, 0.5)
    hold_min = 0.05 if indecisive else 0.04
    hold_max = 0.14 if indecisive else 0.10

    def _type_stealth(element, text):
        """Type with realistic key hold times using JS events."""
        _type_with_hold(driver, element, text,
                        hold_min=hold_min, hold_max=hold_max,
                        think_prob=think_prob, think_range=think_range)

    fidget_prob = 0.5 if indecisive else 0.3

    def _stealth_move_and_click(driver_arg, element, click_only=False):
        """Move to element with mouse wandering and occasional fidgeting."""
        # Sometimes wander mouse before clicking (reading nearby text)
        if random.random() < 0.3:
            _wander_mouse(driver_arg, random.uniform(0.3, 0.8))
        if random.random() < fidget_prob:
            _idle_fidget(driver_arg, random.uniform(0.1, 0.3))
        _human_move_and_click(driver_arg, element, click_only=click_only)

    return _fill_checkout_targeted(driver, _type_stealth, _stealth_move_and_click,
                                   skip_honeypot=skip_honeypot)


def replay_bot(driver, source_path: str, skip_honeypot=False):
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

    return _fill_checkout_targeted(driver, _type_human, _human_move_and_click, skip_honeypot=skip_honeypot)


def semi_auto_bot(driver, skip_honeypot=False):
    """Tier 3 — Semi-automated: bot automates navigation, then pauses for
    a simulated 'human operator' to complete checkout.

    The idea is that a real attacker might script the browsing/seat-selection
    but hand off to a human (or slower, more careful automation) at checkout.
    This creates a session that looks automated early and human-like late,
    making it harder to classify with a single global decision.

    Randomly picks one of three handoff strategies:
      A) Bot navigates → human does checkout  (most common)
      B) Human browses → bot does checkout    (less common)
      C) Bot navigates & selects → human fills payment only
    """
    strategy = random.choices(["bot_then_human", "human_then_bot", "split_checkout"],
                              weights=[0.50, 0.25, 0.25])[0]
    print(f"  Semi-auto strategy: {strategy}")

    if strategy == "bot_then_human":
        # ── BOT phase: fast, efficient navigation ──
        _go_home(driver)
        _wander_mouse(driver, random.uniform(0.3, 0.6))
        _random_scroll(driver, scrolls=1)
        time.sleep(random.uniform(0.2, 0.4))

        if not _pick_concert(driver, _linear_move_and_click):
            return
        time.sleep(random.uniform(0.2, 0.5))

        if not _pick_section(driver, _linear_move_and_click):
            return

        # ── HUMAN phase: slow, careful checkout ──
        _wander_mouse(driver, random.uniform(0.8, 1.5))
        _human_scroll(driver, scrolls=random.randint(1, 2))
        _idle_fidget(driver, random.uniform(0.3, 0.8))

        def _type_careful(element, text):
            _type_with_hold(driver, element, text,
                            hold_min=0.05, hold_max=0.15,
                            think_prob=0.10, think_range=(0.3, 1.0))

        return _fill_checkout_targeted(driver, _type_careful, _human_move_and_click,
                                       skip_honeypot=skip_honeypot)

    elif strategy == "human_then_bot":
        # ── HUMAN phase: browsing and exploring ──
        _go_home(driver)
        _wander_mouse(driver, random.uniform(1.0, 2.0))
        _human_scroll(driver, scrolls=random.randint(2, 4))
        _idle_fidget(driver, random.uniform(0.5, 1.0))
        _random_page_click(driver, random.randint(1, 2))

        if not _pick_concert(driver, _human_move_and_click):
            return

        _wander_mouse(driver, random.uniform(0.8, 1.5))
        _human_scroll(driver, scrolls=random.randint(1, 3))
        _idle_fidget(driver, random.uniform(0.3, 0.8))

        if not _pick_section(driver, _human_move_and_click):
            return

        # ── BOT phase: fast checkout ──
        time.sleep(random.uniform(0.1, 0.3))
        return _fill_checkout_targeted(driver, _type_uniform, _linear_move_and_click,
                                       skip_honeypot=skip_honeypot)

    else:  # split_checkout
        # ── BOT phase: navigate + select ──
        _go_home(driver)
        _wander_mouse(driver, random.uniform(0.5, 1.0))
        time.sleep(random.uniform(0.2, 0.4))

        if not _pick_concert(driver, _human_move_and_click):
            return

        if not _pick_section(driver, _human_move_and_click):
            return

        # ── MIXED phase: bot fills address, human fills payment ──
        _wander_mouse(driver, random.uniform(0.5, 1.0))

        # We use targeted checkout but alternate typing style per field group
        form = get_form_data()
        wait_for(driver, "#card_number", timeout=10)

        # Address fields — bot-like (fast uniform typing)
        address_fields = [
            ("full_name", form["full_name"]),
            ("billing_address", form["billing_address"]),
            ("city", form["city"]),
            ("zip_code", form["zip_code"]),
        ]
        for fid, val in address_fields:
            try:
                el = driver.find_element(By.ID, fid)
                _linear_move_and_click(driver, el)
                _type_uniform(el, val)
                time.sleep(random.uniform(0.05, 0.15))
            except Exception:
                pass

        # Payment fields — human-like (hold typing, pauses)
        _idle_fidget(driver, random.uniform(0.3, 0.8))
        payment_fields = [
            ("card_number", form["card_number"]),
            ("card_expiry", form["card_expiry"]),
            ("card_cvv", form["card_cvv"]),
        ]
        for fid, val in payment_fields:
            try:
                el = driver.find_element(By.ID, fid)
                _human_move_and_click(driver, el)
                _type_with_hold(driver, el, val,
                                hold_min=0.05, hold_max=0.14,
                                think_prob=0.08, think_range=(0.2, 0.6))
                time.sleep(random.uniform(0.1, 0.3))
            except Exception:
                pass

        # Submit
        try:
            submit = wait_for(driver, "button[type='submit']", timeout=5)
            _human_move_and_click(driver, submit)
        except Exception:
            pass

        return _get_session_id(driver)


def trace_conditioned_bot(driver, skip_honeypot=False):
    """Tier 4 — Trace-conditioned: replays perturbed human traces for
    mouse/scroll on every page, combined with human-like typing.

    Unlike replay_bot (tier 2) which replays mouse but types mechanically,
    this bot uses human timing profiles for ALL modalities: mouse trajectories,
    scroll patterns, keystroke intervals, and click cadence — all drawn from
    real human data with noise injection.

    The result is a session whose statistical properties closely match real
    human sessions, making it the hardest scripted bot to detect.
    """
    # Load both human segments (for mouse/scroll replay) and timing profiles
    human_files = sorted(HUMAN_DATA_DIR.glob("*.json")) if HUMAN_DATA_DIR.exists() else []
    if not human_files:
        print("  WARNING: No human data — falling back to stealth_bot")
        return stealth_bot(driver, skip_honeypot=skip_honeypot)

    # Pick a random human session to condition on
    source_path = random.choice(human_files)
    segments = _load_replay_segments(str(source_path))
    profiles = _load_human_timing_profiles()
    profile = random.choice(profiles) if profiles else None

    if not segments:
        print("  WARNING: Empty human session — falling back to stealth_bot")
        return stealth_bot(driver, skip_honeypot=skip_honeypot)

    print(f"  Conditioning on: {source_path.name}")

    # Timing from human profile
    key_intervals = profile["key_intervals"] if profile else []
    click_intervals = profile["click_intervals"] if profile else []

    # --- Page 1: Home (replayed human mouse + scroll) ---
    _go_home(driver)
    seg = random.choice(segments)
    _replay_mouse(driver, seg.get("mouse", []), max_events=40)
    _replay_scroll(driver, seg.get("scroll", []), max_events=5)
    _idle_fidget(driver, random.uniform(0.3, 0.8))

    if not _pick_concert(driver, _human_move_and_click):
        return

    # --- Page 2: Seat Selection (replayed human mouse) ---
    seg = random.choice(segments)
    _replay_mouse(driver, seg.get("mouse", []), max_events=80)
    _replay_scroll(driver, seg.get("scroll", []), max_events=5)
    _idle_fidget(driver, random.uniform(0.2, 0.6))

    if not _pick_section(driver, _human_move_and_click):
        return

    # --- Page 3: Checkout (human-profiled typing + replayed mouse) ---
    seg = random.choice(segments)
    _replay_mouse(driver, seg.get("mouse", []), max_events=40)
    time.sleep(random.uniform(0.3, 0.6))

    # Type using human-sampled intervals with key hold
    def _type_trace_conditioned(element, text):
        _type_with_hold(driver, element, text,
                        hold_min=0.04, hold_max=0.13,
                        think_prob=0.08, think_range=(0.2, 0.7))
        # Occasionally add a human-profiled pause between fields
        if key_intervals and random.random() < 0.3:
            pause = _sample_from_human(key_intervals, 80, 400)
            time.sleep(pause)

    def _trace_move_and_click(driver_arg, element, click_only=False):
        # Occasionally replay a small mouse segment before clicking
        if random.random() < 0.25 and segments:
            s = random.choice(segments)
            _replay_mouse(driver_arg, s.get("mouse", []), max_events=15)
        _human_move_and_click(driver_arg, element, click_only=click_only)
        # Use human click cadence
        if click_intervals and random.random() < 0.4:
            pause = _sample_from_human(click_intervals, 200, 1500)
            time.sleep(min(pause, 1.5))

    return _fill_checkout_targeted(driver, _type_trace_conditioned,
                                   _trace_move_and_click,
                                   skip_honeypot=skip_honeypot)


# ---------------------------------------------------------------------------
# Replay helpers
# ---------------------------------------------------------------------------

def _load_replay_segments(source_path: str) -> list[dict]:
    """Load segments from a session JSON file (live-confirm format)."""
    with open(source_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = []
    if isinstance(data, dict):
        if "segments" in data and isinstance(data["segments"], list):
            segments = data["segments"]
    elif isinstance(data, list):
        segments = data

    return [s for s in segments if isinstance(s, dict)]


def _replay_mouse(driver, mouse_events: list[dict], max_events: int = 100):
    """Replay mouse movements via JS dispatch — no out-of-bounds errors.

    Uses dispatchEvent with absolute clientX/clientY so coordinates never
    drift. tracking.js picks these up the same as real mouse events.
    """
    if not mouse_events:
        return

    events = mouse_events[:max_events]

    # Get actual viewport dimensions
    vp_w = driver.execute_script("return window.innerWidth") or 1280
    vp_h = driver.execute_script("return window.innerHeight") or 720
    margin = 30

    # Find the bounding box of the recorded trace
    xs = [e.get("x", e.get("pageX", 0)) or 0 for e in events]
    ys = [e.get("y", e.get("pageY", 0)) or 0 for e in events]
    src_min_x, src_max_x = min(xs), max(xs)
    src_min_y, src_max_y = min(ys), max(ys)
    src_w = max(src_max_x - src_min_x, 1)
    src_h = max(src_max_y - src_min_y, 1)

    # Scale to fit within viewport
    scale = min((vp_w - 2 * margin) / src_w,
                (vp_h - 2 * margin) / src_h, 1.0)

    prev_t = 0
    for evt in events:
        raw_x = evt.get("x", evt.get("pageX", 0)) or 0
        raw_y = evt.get("y", evt.get("pageY", 0)) or 0
        t = evt.get("t", evt.get("timestamp", 0))

        # Normalize + noise + clamp
        nx = (raw_x - src_min_x) * scale + margin + random.gauss(0, 3)
        ny = (raw_y - src_min_y) * scale + margin + random.gauss(0, 2)
        nx = int(max(margin, min(vp_w - margin, nx)))
        ny = int(max(margin, min(vp_h - margin, ny)))

        # Dispatch a real mousemove event via JS
        driver.execute_script(
            "document.elementFromPoint(arguments[0],arguments[1])"
            "?.dispatchEvent(new MouseEvent('mousemove',{clientX:arguments[0],"
            "clientY:arguments[1],bubbles:true}));",
            nx, ny,
        )

        dt = (t - prev_t) / 1000 if prev_t else 0.015
        dt = max(0.005, min(0.15, dt)) * random.uniform(0.8, 1.2)
        time.sleep(dt)
        prev_t = t


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
        _dispatch_wheel(driver, dy)
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


def _export_and_confirm(driver, run_index: int, session_id: str | None = None,
                        bot_type: str = "unknown") -> None:
    """Pull telemetry from backend, save to data/bot/, confirm as bot."""
    import urllib.request

    # Wait for tracking.js to flush
    print("  Waiting for telemetry flush...")
    time.sleep(8)

    # Use pre-captured session ID if available (Confirmation page resets it)
    if not session_id:
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

    # Save JSON — use the same live_confirm format as human sessions
    # so the export structure itself doesn't leak the label.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-%f")
    filename = f"session_{session_id[:13]}_{ts}.json"
    out_path = DATA_DIR / filename

    consolidated = {
        "sessionId": session_id,
        "label": 0,
        "exportedAt": datetime.now(timezone.utc).isoformat(),
        "source": "live_confirm",
        "bot_type": bot_type,
        "tier": BOT_TIER.get(bot_type, 0),
        "segments": [{
            "mouse": mouse, "clicks": clicks,
            "keystrokes": keystrokes, "scroll": scroll,
        }],
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
    parser.add_argument("--type", choices=[
        "linear", "scripted", "replay", "tabber", "slow", "erratic",
        "speedrun", "stealth", "semi_auto", "trace_conditioned", "mixed",
    ], default="scripted")
    parser.add_argument("--replay-source", type=str, help="JSON file for replay bot")
    parser.add_argument("--pause-between", type=float, default=2.0, help="Seconds between runs")
    parser.add_argument("--skip-honeypot", action="store_true", help="Skip unknown form fields (avoids honeypot traps)")
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
                bot_type = args.type
                if bot_type == "mixed":
                    # Diverse mix across all tiers
                    bot_type = random.choices(
                        ["linear", "scripted", "tabber", "slow", "erratic",
                         "speedrun", "stealth", "semi_auto", "trace_conditioned"],
                        weights=[0.06, 0.18, 0.04, 0.10, 0.05,
                                 0.04, 0.25, 0.15, 0.13],
                    )[0]
                    print(f"  Mixed mode → {bot_type}")

                skip = args.skip_honeypot
                captured_sid = None
                if bot_type == "linear":
                    captured_sid = linear_bot(driver, skip_honeypot=skip)
                elif bot_type == "scripted":
                    captured_sid = scripted_bot(driver, skip_honeypot=skip)
                elif bot_type == "replay":
                    if not args.replay_source:
                        print("Error: --replay-source required for replay bot")
                        return
                    captured_sid = replay_bot(driver, args.replay_source, skip_honeypot=skip)
                elif bot_type == "tabber":
                    captured_sid = tabber_bot(driver, skip_honeypot=skip)
                elif bot_type == "slow":
                    captured_sid = slow_bot(driver, skip_honeypot=skip)
                elif bot_type == "erratic":
                    captured_sid = erratic_bot(driver, skip_honeypot=skip)
                elif bot_type == "speedrun":
                    captured_sid = speedrun_bot(driver, skip_honeypot=skip)
                elif bot_type == "stealth":
                    captured_sid = stealth_bot(driver, skip_honeypot=skip)
                elif bot_type == "semi_auto":
                    captured_sid = semi_auto_bot(driver, skip_honeypot=skip)
                elif bot_type == "trace_conditioned":
                    captured_sid = trace_conditioned_bot(driver, skip_honeypot=skip)
                run_succeeded = captured_sid is not None
                if run_succeeded:
                    print(f"  Run {i + 1} complete.")
                else:
                    print(f"  Run {i + 1} did not reach checkout.")
            except Exception as e:
                run_succeeded = False
                captured_sid = None
                print(f"  Run {i + 1} failed: {e}")

            # Auto-export telemetry and confirm as bot (skip if run failed)
            if run_succeeded:
                _export_and_confirm(driver, i + 1, session_id=captured_sid,
                                    bot_type=bot_type)
            else:
                print("  Skipping export — run did not complete successfully")

            if i < args.runs - 1:
                print(f"  Waiting {args.pause_between}s...")
                time.sleep(args.pause_between)

        print(f"\n{'='*50}")
        print("All runs complete!")
        print(f"Telemetry saved to: {DATA_DIR}")
        print("="*50)
        if sys.stdin.isatty():
            input("Press Enter to close the browser...")

    finally:
        driver.quit()


if __name__ == "__main__":
    main()
