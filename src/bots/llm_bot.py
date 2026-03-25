"""LLM-powered bot using browser-use to autonomously navigate TicketMonarch.

Telemetry is captured automatically by the React app's built-in tracking
(tracking.js) and saved to the Flask backend. After the agent finishes,
this script pulls the telemetry from the backend API, saves it to
data/bot/ as JSON, and confirms the session as a bot so the RL agent
can learn from it.

Supports --inject-events mode which patches the page to generate real
DOM events (mousemove, click, keydown/keyup, scroll) from CDP actions,
so tracking.js captures full telemetry. Without this flag, telemetry
will be sparse (CDP bypasses DOM listeners).

Supports --mode to control how the LLM perceives the page:
  - screenshot:     Vision-based (LLM sees screenshots)
  - dom:            Text-based (LLM reads DOM element descriptions)
  - accessibility:  Text-based + accessibility tree context via CDP
  - mixed:          Cycles through all three modes round-robin

Setup:
    pip install browser-use
    # browser-use v0.11+ uses CDP directly (no Playwright needed)

Usage:
    export ANTHROPIC_API_KEY=sk-...   # or OPENAI_API_KEY=sk-...

    python bots/llm_bot.py --runs 3 --provider anthropic
    python bots/llm_bot.py --runs 3 --provider anthropic --inject-events
    python bots/llm_bot.py --runs 3 --mode mixed --provider anthropic
    python bots/llm_bot.py --runs 3 --mode dom --provider openai
    python bots/llm_bot.py --runs 1 --mode accessibility --skip-honeypot

Requires:
    - browser-use >= 0.9.0
    - TicketMonarch backend running (python app.py)
    - TicketMonarch frontend running (npm run dev)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path


SITE_URL = "http://localhost:3000"
API_URL = "http://localhost:5000"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "bot"

# Perception modes for the LLM agent
MODES = ["screenshot", "dom", "accessibility"]

# JavaScript that patches the page to generate real DOM events from
# CDP-level automation. Injected via addScriptToEvaluateOnNewDocument
# so it runs on every page load including navigations.
INJECT_EVENTS_JS = r"""
(function() {
    if (window.__tmEventInjectorLoaded) return;
    window.__tmEventInjectorLoaded = true;

    function dispatchMouseMoveBurst(x, y) {
        for (let i = 0; i < 2; i++) {
            const jitterX = x + (Math.random() - 0.5) * 12;
            const jitterY = y + (Math.random() - 0.5) * 8;
            window.dispatchEvent(new MouseEvent('mousemove', {
                clientX: jitterX, clientY: jitterY, bubbles: true
            }));
        }
    }

    // Patch click() to also dispatch a real MouseEvent on window
    const origClick = HTMLElement.prototype.click;
    HTMLElement.prototype.click = function() {
        origClick.apply(this, arguments);
        const rect = this.getBoundingClientRect();
        const x = rect.left + rect.width / 2;
        const y = rect.top + rect.height / 2;
        dispatchMouseMoveBurst(x, y);
        window.dispatchEvent(new MouseEvent('click', {
            clientX: x, clientY: y, bubbles: true, button: 0
        }));
    };

    // Patch focus() to generate a synthetic mousemove to the element
    const origFocus = HTMLElement.prototype.focus;
    HTMLElement.prototype.focus = function() {
        origFocus.apply(this, arguments);
        const rect = this.getBoundingClientRect();
        const x = rect.left + rect.width / 2;
        const y = rect.top + rect.height / 2;
        dispatchMouseMoveBurst(x, y);
    };

    // Monitor input events (from CDP insertText) and convert to keydown/keyup
    // so tracking.js keystroke listeners fire
    document.addEventListener('input', function(e) {
        const target = e.target;
        if (!target || !['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return;
        // Dispatch a keydown and keyup for each character of inserted data
        const data = e.data || '';
        for (let i = 0; i < data.length; i++) {
            target.dispatchEvent(new KeyboardEvent('keydown', {
                key: data[i], bubbles: true, cancelable: true
            }));
            target.dispatchEvent(new KeyboardEvent('keyup', {
                key: data[i], bubbles: true, cancelable: true
            }));
        }
    }, true);

    // Patch scrollTo/scrollBy to dispatch real scroll events
    const origScrollTo = window.scrollTo;
    window.scrollTo = function() {
        origScrollTo.apply(this, arguments);
        window.dispatchEvent(new Event('scroll'));
    };
    const origScrollBy = window.scrollBy;
    window.scrollBy = function() {
        origScrollBy.apply(this, arguments);
        window.dispatchEvent(new Event('scroll'));
    };

    console.log('[TM] Event injector loaded — DOM events will be generated');
})();
"""


def api_get(path: str, timeout: int = 10) -> dict | None:
    """GET request to Flask backend. Returns parsed JSON or None."""
    try:
        url = f"{API_URL}{path}"
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"  WARNING: GET {path} failed: {e}")
        return None


def api_post(path: str, body: dict, timeout: int = 10) -> dict | None:
    """POST JSON to Flask backend. Returns parsed JSON or None."""
    try:
        url = f"{API_URL}{path}"
        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"  WARNING: POST {path} failed: {e}")
        return None


def get_recent_session_ids() -> list[str]:
    """Get list of recent session IDs from the backend (lightweight endpoint)."""
    # Try the lightweight endpoint first (no JSON parsing, less likely to 500)
    data = api_get("/api/agent/session-ids?limit=10")
    if data and data.get("success"):
        return data.get("session_ids", [])

    # Fallback to the heavier endpoint
    data = api_get("/api/agent/sessions?limit=10")
    if data and data.get("success"):
        return [s["session_id"] for s in data.get("sessions", [])]

    return []


def save_telemetry_json(session_id: str, raw: dict, run_index: int) -> Path:
    """Save raw telemetry in live_confirm format (same as human exports)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    mouse = raw.get("mouse", [])
    clicks = raw.get("clicks", [])
    keystrokes = raw.get("keystrokes", [])
    scroll = raw.get("scroll", [])

    consolidated = {
        "sessionId": session_id,
        "label": 0,
        "exportedAt": datetime.now(timezone.utc).isoformat(),
        "source": "live_confirm",
        "bot_type": "llm",
        "segments": [{
            "mouse": mouse,
            "clicks": clicks,
            "keystrokes": keystrokes,
            "scroll": scroll,
        }],
    }

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-%f")
    filename = f"session_{session_id[:13]}_{ts}.json"
    out_path = DATA_DIR / filename

    with open(out_path, "w") as f:
        json.dump(consolidated, f, indent=2)

    return out_path


async def _inject_event_script(browser) -> bool:
    """Inject DOM event generation script via CDP addScriptToEvaluateOnNewDocument.

    Returns True if injection succeeded.
    """
    try:
        cdp_client = getattr(browser, "_cdp_client_root", None)
        if cdp_client is None:
            print("  (no CDP client available for injection)")
            return False

        await cdp_client.send.Page.addScriptToEvaluateOnNewDocument(
            params={"source": INJECT_EVENTS_JS}
        )
        print("  Event injection script installed (will activate on page load)")
        return True
    except Exception as e:
        print(f"  WARNING: Could not inject event script: {e}")
        return False


async def _get_accessibility_tree(browser, max_chars: int = 4000) -> str | None:
    """Extract the accessibility tree from the current page via CDP.

    Returns a formatted text representation of the accessibility tree,
    truncated to max_chars. Returns None if extraction fails.
    """
    try:
        cdp_session = getattr(browser, "agent_focus", None)
        if cdp_session is None:
            return None

        cdp_client = getattr(cdp_session, "cdp_client", None)
        if cdp_client is None:
            return None

        result = await cdp_client.send.Accessibility.getFullAXTree(
            params={},
            session_id=cdp_session.session_id,
        )

        nodes = result.get("nodes", [])
        if not nodes:
            return None

        lines = []
        # Build a lookup for parent-child relationships
        node_map = {}
        for node in nodes:
            node_id = node.get("nodeId", "")
            node_map[node_id] = node

        # Determine depth of each node via parentId chain
        def get_depth(n):
            depth = 0
            visited = set()
            current = n
            while True:
                pid = current.get("parentId")
                if not pid or pid in visited:
                    break
                visited.add(pid)
                parent = node_map.get(pid)
                if not parent:
                    break
                current = parent
                depth += 1
            return depth

        for node in nodes:
            role = node.get("role", {}).get("value", "")
            name = node.get("name", {}).get("value", "")
            # Skip generic/invisible nodes
            if role in ("none", "generic", "InlineTextBox", "StaticText") and not name:
                continue

            depth = get_depth(node)
            indent = "  " * min(depth, 10)
            props = []
            for prop in node.get("properties", []):
                prop_name = prop.get("name", "")
                prop_val = prop.get("value", {}).get("value", "")
                if prop_name in ("focusable", "editable", "required", "disabled", "checked"):
                    if prop_val:
                        props.append(prop_name)

            line = f"{indent}[{role}]"
            if name:
                line += f' "{name}"'
            if props:
                line += f" ({', '.join(props)})"
            lines.append(line)

        tree_text = "\n".join(lines)
        if len(tree_text) > max_chars:
            tree_text = tree_text[:max_chars] + "\n... (truncated)"

        return tree_text

    except Exception as e:
        print(f"  WARNING: Could not extract accessibility tree: {e}")
        return None


async def _read_session_id_from_browser(browser) -> str | None:
    """Try to read tm_session_id via CDP Runtime.evaluate on the active page."""
    try:
        # browser-use v0.11+ uses CDP directly (no Playwright)
        # agent_focus is the CDPSession for the currently active tab
        cdp_session = getattr(browser, "agent_focus", None)
        if cdp_session is None:
            return None

        cdp_client = getattr(cdp_session, "cdp_client", None)
        if cdp_client is None:
            return None

        result = await cdp_client.send.Runtime.evaluate(
            params={
                "expression": "window.sessionStorage.getItem('tm_session_id')",
                "returnByValue": True,
            },
            session_id=cdp_session.session_id,
        )
        value = result.get("result", {}).get("value")
        if value and value != "null":
            # Also set the bot flag so Confirmation.jsx won't auto-confirm as human
            try:
                await cdp_client.send.Runtime.evaluate(
                    params={
                        "expression": "window.sessionStorage.setItem('tm_is_bot', '1')",
                    },
                    session_id=cdp_session.session_id,
                )
            except Exception:
                pass
            return value
    except Exception as e:
        print(f"  (CDP session read failed: {e})")
    return None


async def extract_and_save(browser, run_index: int, known_session_ids: list[str]) -> str | None:
    """Extract telemetry from Flask backend and save to data/bot/.

    Uses three strategies to find the session ID:
    1. Read sessionStorage from browser pages on localhost
    2. Diff recent session IDs from backend to find the new one
    3. Last resort: use the most recent session ID

    Returns the session_id if successful, None otherwise.
    """
    # Wait for tracking.js to flush (it flushes every 5s)
    print("  Waiting for telemetry flush...")
    await asyncio.sleep(8)

    session_id = None

    # Strategy 1: read sessionStorage from the browser
    session_id = await _read_session_id_from_browser(browser)
    if session_id:
        print(f"  Session ID (from browser): {session_id}")

    # Strategy 2: find new session by diffing the sessions list
    if not session_id:
        current_ids = get_recent_session_ids()
        if not current_ids:
            print("  WARNING: Could not fetch session IDs from backend")
        else:
            new_ids = [sid for sid in current_ids if sid not in known_session_ids]
            if new_ids:
                session_id = new_ids[0]
                print(f"  Session ID (from backend diff): {session_id}")
            else:
                # Last resort: just use the most recent session
                session_id = current_ids[0]
                print(f"  Session ID (most recent): {session_id}")

    if not session_id:
        print("  ERROR: Could not determine session ID")
        return None

    # Pull raw telemetry
    raw = api_get(f"/api/session/raw/{session_id}")
    if not raw or not raw.get("success"):
        print("  ERROR: Could not pull raw telemetry from backend")
        return None

    mouse_count = len(raw.get("mouse", []))
    click_count = len(raw.get("clicks", []))
    key_count = len(raw.get("keystrokes", []))
    scroll_count = len(raw.get("scroll", []))
    total = mouse_count + click_count + key_count + scroll_count

    if total == 0:
        print("  WARNING: Session exists but has 0 events")
        return None

    print(f"  Events: {mouse_count} mouse, {click_count} clicks, "
          f"{key_count} keystrokes, {scroll_count} scroll")

    out_path = save_telemetry_json(session_id, raw, run_index)
    print(f"  Saved: {out_path.name} ({total} events)")
    return session_id


def confirm_bot(session_id: str) -> None:
    """Tell the RL agent this session was a bot so it can learn."""
    print("  Confirming bot label + triggering online RL update (this may take a minute)...")
    result = api_post("/api/agent/confirm", {
        "session_id": session_id,
        "true_label": 0,  # 0 = bot
    }, timeout=120)  # PPO update on large sessions can take >10s
    if result and result.get("success"):
        updated = result.get("updated", False)
        if updated:
            metrics = result.get("metrics", {})
            print(f"  RL agent updated! (loss: {metrics.get('policy_loss', '?')}, steps: {result.get('steps', '?')})")
        else:
            print(f"  RL agent confirmed (no update: {result.get('reason', '?')})")
    else:
        print("  WARNING: Could not confirm bot label with RL agent")


def _build_task_prompt(skip_honeypot: bool = False) -> str:
    """Build the default task prompt for the LLM bot."""
    task = f"""
Go to {SITE_URL} and complete the full ticket booking flow:
1. Browse the available concerts on the home page
2. Click on a concert's "Tickets" button to view available sections
3. Select a section, then click "Continue to Checkout"
4. Fill in the checkout form with realistic fake data:
   - Card number: 4111111111111111
   - Expiry: 12/28
   - CVV: 123
   - Full name: any realistic name
   - Billing address: any realistic address
   - City, State, Zip code
5. Click "Purchase" to submit

Act naturally - take your time browsing, move around the page,
scroll through options before selecting.
"""
    if skip_honeypot:
        task += """
IMPORTANT: Only fill in the form fields listed above (card number,
expiry, CVV, name, address, city, state, zip). If you see any other
input fields that are not in this list, DO NOT fill them in — skip
them completely. They may be hidden traps.
"""
    return task


async def run_llm_bot(provider: str = "anthropic", task: str | None = None,
                      browser=None, inject_events: bool = False,
                      mode: str = "screenshot", skip_honeypot: bool = False):
    """Run a single LLM bot session.

    Args:
        provider: LLM provider ("anthropic" or "openai")
        task: Custom task prompt (None for default)
        browser: Existing Browser instance to reuse
        inject_events: Whether to inject DOM event generation script
        mode: Perception mode — "screenshot", "dom", or "accessibility"
        skip_honeypot: If True, task prompt tells LLM to skip unknown fields
    """
    try:
        from browser_use import Agent, Browser
    except ImportError:
        print("Error: browser-use not installed.")
        print("Install with: pip install browser-use")
        sys.exit(1)

    if task is None:
        task = _build_task_prompt(skip_honeypot=skip_honeypot)

    if provider == "anthropic":
        from browser_use import ChatAnthropic
        llm = ChatAnthropic(model="claude-sonnet-4-20250514")
    elif provider == "openai":
        from browser_use import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o")
    else:
        print(f"Unknown provider: {provider}")
        sys.exit(1)

    if browser is None:
        browser = Browser(
            headless=False,
            disable_security=True,
            args=["--no-first-run"],
        )

    # Inject DOM event generation script if requested
    if inject_events:
        await _inject_event_script(browser)

    # Configure agent based on perception mode
    use_vision = (mode == "screenshot")
    extend_msg = None

    if mode == "accessibility":
        # Extract accessibility tree and add it as context
        ax_tree = await _get_accessibility_tree(browser)
        if ax_tree:
            extend_msg = (
                "You are navigating a website using its accessibility tree. "
                "Below is the current page's accessibility structure:\n\n"
                f"```\n{ax_tree}\n```\n\n"
                "Use this information along with the DOM element descriptions "
                "to understand the page layout and interact with elements."
            )
            print(f"  Accessibility tree extracted ({len(ax_tree)} chars)")
        else:
            print("  WARNING: Could not extract accessibility tree, falling back to DOM-only")

    agent_kwargs = dict(task=task, llm=llm, browser=browser, use_vision=use_vision)
    if extend_msg:
        agent_kwargs["extend_system_message"] = extend_msg

    agent = Agent(**agent_kwargs)
    result = await agent.run()
    print(f"Agent result: {result}")
    return browser


async def run_multiple(provider: str, runs: int, task: str | None,
                       pause: float, inject_events: bool = False,
                       mode: str = "mixed", skip_honeypot: bool = False):
    """Run multiple LLM bot sessions."""
    try:
        from browser_use import Browser
    except ImportError:
        print("Error: browser-use not installed.")
        sys.exit(1)

    browser = Browser(
        headless=False,
        disable_security=True,
        args=["--no-first-run"],
    )

    for i in range(runs):
        print(f"\n{'='*50}")

        # Determine perception mode for this run
        if mode == "mixed":
            current_mode = MODES[i % len(MODES)]
        else:
            current_mode = mode

        # Alternate injection: even runs get injection, odd runs don't
        use_injection = inject_events and (i % 2 == 0)
        inject_label = " + event injection" if use_injection else ""
        print(f"LLM Bot Run {i + 1}/{runs} [mode: {current_mode}{inject_label}]")
        print(f"{'='*50}")

        # Snapshot current session IDs so we can find the new one after
        known_ids = get_recent_session_ids()

        try:
            await run_llm_bot(
                provider=provider,
                task=task,
                browser=browser,
                inject_events=use_injection,
                mode=current_mode,
                skip_honeypot=skip_honeypot,
            )
        except Exception as e:
            print(f"  Run {i + 1} error: {e}")

        # Extract telemetry and save to data/bot/
        print("Extracting telemetry...")
        session_id = await extract_and_save(browser, i + 1, known_ids)

        # Confirm this was a bot so the RL agent can learn
        if session_id:
            confirm_bot(session_id)

        if i < runs - 1:
            print(f"Waiting {pause}s before next run...")
            await asyncio.sleep(pause)

    print(f"\n{'='*50}")
    print("All runs complete!")
    print(f"Telemetry saved to: {DATA_DIR}")
    print("="*50)

    if sys.stdin.isatty():
        input("Press Enter to close the browser...")
    await browser.stop()


def main():
    parser = argparse.ArgumentParser(description="Run LLM-powered bot")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    parser.add_argument("--pause-between", type=float, default=3.0)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--inject-events", action="store_true",
                        help="Inject DOM events so tracking.js captures full telemetry. "
                             "Alternates: even runs get injection, odd runs don't.")
    parser.add_argument("--mode", choices=["screenshot", "dom", "accessibility", "mixed"],
                        default="mixed",
                        help="Perception mode: screenshot (vision), dom (text-only), "
                             "accessibility (accessibility tree context), or mixed (cycles all three)")
    parser.add_argument("--skip-honeypot", action="store_true",
                        help="Tell LLM to skip unknown/hidden form fields (avoids honeypot traps)")
    args = parser.parse_args()

    mode_label = f"mode: {args.mode}"
    inject_label = " + alternating injection" if args.inject_events else ""
    honeypot_label = " + skip-honeypot" if args.skip_honeypot else ""
    print(f"LLM Bot ({args.provider}) - {args.runs} runs [{mode_label}{inject_label}{honeypot_label}]")
    print(f"Target: {SITE_URL}")
    print(f"Output: {DATA_DIR}")
    print()
    print("Make sure backend (python app.py) and frontend (npm run dev) are running!")
    print()

    asyncio.run(run_multiple(
        provider=args.provider,
        runs=args.runs,
        task=args.task,
        pause=args.pause_between,
        inject_events=args.inject_events,
        mode=args.mode,
        skip_honeypot=args.skip_honeypot,
    ))


if __name__ == "__main__":
    main()
