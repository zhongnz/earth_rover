import os
import time
import curses
import requests

DEFAULT_BASE_URL = os.getenv("SDK_URL", "http://127.0.0.1:8000")
CONTROL_URL = f"{DEFAULT_BASE_URL}/control"
SDK_URL = f"{DEFAULT_BASE_URL}/sdk"
SDK_STATUS_URL = f"{DEFAULT_BASE_URL}/sdk-status"
KEY_HOLD_TIMEOUT_S = 0.35

MOVEMENT_KEYMAP = {
    ord("w"): "w",
    ord("W"): "w",
    curses.KEY_UP: "w",
    ord("s"): "s",
    ord("S"): "s",
    curses.KEY_DOWN: "s",
    ord("a"): "a",
    ord("A"): "a",
    curses.KEY_LEFT: "a",
    ord("d"): "d",
    ord("D"): "d",
    curses.KEY_RIGHT: "d",
}

STOP_KEYS = {
    ord(" "),
    curses.KEY_BACKSPACE,
    127,
    8,
}

QUIT_KEYS = {
    ord("q"),
    ord("Q"),
}


def clamp(value: float, min_value: float = -1.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


def initialize_sdk_session() -> None:
    try:
        # Warm the backend Playwright bridge before the first control command.
        requests.get(SDK_STATUS_URL, timeout=60)
    except Exception:
        try:
            requests.get(SDK_URL, timeout=5)
        except Exception:
            # Non-fatal if this fails; /control path will also lazy-init
            pass


def calculate_target_from_keys(pressed_keys: set[str]) -> tuple[float, float]:
    linear = 0.0
    angular = 0.0
    if "w" in pressed_keys:
        linear += 1.0
    if "s" in pressed_keys:
        linear -= 1.0
    if "a" in pressed_keys:
        angular += 1.0
    if "d" in pressed_keys:
        angular -= 1.0

    if linear != 0.0 and angular != 0.0:
        factor = 2 ** -0.5
        linear *= factor
        angular *= factor

    return linear, angular


def lerp(start: float, end: float, t: float) -> float:
    return start + (end - start) * t


def send_command(linear: float, angular: float) -> tuple[bool, str]:
    try:
        payload = {"command": {"linear": float(linear), "angular": float(angular)}}
        response = requests.post(CONTROL_URL, json=payload, timeout=2)
        if response.ok:
            return True, "ok"
        return False, f"http {response.status_code}"
    except Exception as exc:
        return False, str(exc)


def send_stop_command() -> tuple[bool, str]:
    return send_command(0.0, 0.0)


def draw_ui(stdscr, status: dict) -> None:
    stdscr.erase()
    stdscr.addstr(0, 0, "Earth Rovers SDK - Keyboard Control (Arrows or W/A/S/D to drive, Space to stop, +/- speed, Q to quit)")
    stdscr.addstr(2, 0, f"Base URL: {DEFAULT_BASE_URL}")
    stdscr.addstr(3, 0, f"Tick: {status['tick_ms']} ms, Smoothing: {int(status['smoothing']*100)}%  MaxSpeed: {status['max_speed']:.2f}  Input: {status['input_mode']}")
    stdscr.addstr(5, 0, f"Target linear: {status['target_linear']:+.2f}   Target angular: {status['target_angular']:+.2f}")
    stdscr.addstr(6, 0, f"Actual linear: {status['current_linear']:+.2f}   Actual angular: {status['current_angular']:+.2f}")
    stdscr.addstr(8, 0, f"Last send: {status['last_send_result']} at {status['last_send_time']:.2f}s")
    stdscr.addstr(10, 0, f"Active keys: {' '.join(sorted(status['pressed'])) or '-'}")
    stdscr.refresh()


def main(stdscr) -> None:
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(10)
    stdscr.keypad(True)

    initialize_sdk_session()

    update_interval_s = 0.05  # 20 Hz
    smoothing = 0.5  # 0..1
    max_speed = 0.5  # scales target output
    # Terminal-native curses input relies on key repeat, so we keep
    # each movement key active briefly after its last event.

    pressed_keys: set[str] = set()
    held_until = {key: 0.0 for key in ("w", "a", "s", "d")}
    current_linear = 0.0
    current_angular = 0.0
    target_linear = 0.0
    target_angular = 0.0
    last_send_time = 0.0
    last_send_result = "-"
    start_time = time.time()

    while True:
        loop_now = time.monotonic()
        quit_requested = False
        force_stop = False

        key = stdscr.getch()
        while key != -1:
            if key in QUIT_KEYS:
                quit_requested = True
            elif key in STOP_KEYS:
                for name in held_until:
                    held_until[name] = 0.0
                force_stop = True
            elif key in (ord("+"), ord("=")):
                max_speed = clamp(max_speed + 0.05, 0.05, 1.0)
            elif key in (ord("-"), ord("_")):
                max_speed = clamp(max_speed - 0.05, 0.05, 1.0)
            else:
                mapped = MOVEMENT_KEYMAP.get(key)
                if mapped is not None:
                    held_until[mapped] = loop_now + KEY_HOLD_TIMEOUT_S
            key = stdscr.getch()

        if quit_requested:
            send_stop_command()
            return

        pressed_keys = {
            name for name, until in held_until.items() if until > loop_now
        }

        if force_stop:
            pressed_keys.clear()
            target_linear = 0.0
            target_angular = 0.0

        # Derive targets from active keys
        t_linear, t_angular = calculate_target_from_keys(pressed_keys)
        target_linear = t_linear * max_speed
        target_angular = t_angular * max_speed

        # Smooth towards targets
        current_linear = lerp(current_linear, target_linear, smoothing)
        current_angular = lerp(current_angular, target_angular, smoothing)

        # Send only if significant or periodically
        should_send = abs(current_linear) > 0.01 or abs(current_angular) > 0.01
        now = time.time() - start_time
        if should_send or (now - last_send_time) > 1.0:
            ok, msg = send_command(clamp(current_linear), clamp(current_angular))
            last_send_result = "ok" if ok else f"err: {msg}"
            last_send_time = now

        status = {
            "tick_ms": int(update_interval_s * 1000),
            "smoothing": smoothing,
            "max_speed": max_speed,
            "target_linear": target_linear,
            "target_angular": target_angular,
            "current_linear": current_linear,
            "current_angular": current_angular,
            "last_send_result": last_send_result,
            "last_send_time": last_send_time,
            "pressed": pressed_keys,
            "input_mode": "curses",
        }

        draw_ui(stdscr, status)
        time.sleep(update_interval_s)


if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        send_stop_command()
