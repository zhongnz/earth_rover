import asyncio
import os
import time

from dotenv import load_dotenv
from playwright.async_api import TimeoutError as PlaywrightTimeoutError, async_playwright

load_dotenv()

# Configuration from environment variables with defaults
FORMAT = os.getenv("IMAGE_FORMAT", "png")
QUALITY = float(os.getenv("IMAGE_QUALITY", "1.0"))
HAS_REAR_CAMERA = os.getenv("HAS_REAR_CAMERA", "False").lower() == "true"
BROWSER_ENGINE = os.getenv("BROWSER_ENGINE", "webkit").lower()
BROWSER_HEADLESS = os.getenv("BROWSER_HEADLESS", "false").lower() == "true"
SDK_BASE_URL = os.getenv("SDK_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

if FORMAT not in ["png", "jpeg", "webp"]:
    raise ValueError("Invalid image format. Supported formats: png, jpeg, webp")

if QUALITY < 0 or QUALITY > 1:
    raise ValueError("Invalid image quality. Quality should be between 0 and 1")


class BrowserService:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.page = None
        self._init_lock = asyncio.Lock()
        self.default_viewport = {"width": 3840, "height": 2160}
        self.last_runtime_status = {
            "browser_initialized": False,
            "joined_rtc": False,
            "remote_video_published": False,
            "front_uid": None,
            "rear_uid": None,
            "remote_video_uids": [],
            "severity": "secondary",
            "message": "Browser not initialized.",
            "last_error": None,
        }

    def _browser_is_live(self) -> bool:
        if not self.browser:
            return False
        try:
            return self.browser.is_connected()
        except Exception:
            return False

    def _page_is_live(self) -> bool:
        if not self.page:
            return False
        try:
            return not self.page.is_closed()
        except Exception:
            return False

    def _session_is_live(self) -> bool:
        return self._browser_is_live() and self._page_is_live()

    async def _invalidate_browser(self, message: str, exc: Exception | None = None):
        suffix = f": {exc}" if exc else ""
        print(f"{message}{suffix}")
        await self.close_browser(
            status_message=f"{message}{suffix}",
            status_severity="danger",
            last_error=str(exc) if exc else None,
        )

    async def initialize_browser(self):
        async with self._init_lock:
            if self._session_is_live():
                return

            if self.browser or self.page or self.playwright:
                await self.close_browser(status_message=None)

            try:
                self.playwright = await async_playwright().start()

                browser_factory = getattr(self.playwright, BROWSER_ENGINE, None)
                if browser_factory is None:
                    raise ValueError(
                        f"Unsupported BROWSER_ENGINE={BROWSER_ENGINE}. "
                        "Use one of: webkit, chromium, firefox."
                    )

                launch_kwargs = {"headless": BROWSER_HEADLESS}
                executable_path = os.getenv("CHROME_EXECUTABLE_PATH", "").strip()
                if BROWSER_ENGINE == "chromium" and executable_path:
                    launch_kwargs["executable_path"] = executable_path

                self.browser = await browser_factory.launch(**launch_kwargs)
                self.page = await self.browser.new_page(
                    viewport=self.default_viewport,
                    extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
                )

                await self.page.goto(f"{SDK_BASE_URL}/sdk", wait_until="networkidle")
                await self.page.click("#join")
                await self.page.wait_for_selector("#map", timeout=30000)
                await self.page.wait_for_function(
                    """() => (
                        typeof window.initializeImageParams === \"function\" &&
                        typeof window.getRemoteVideoStatus === \"function\"
                    )""",
                    timeout=10000,
                )
                await self.page.set_viewport_size(self.default_viewport)
                await self.page.wait_for_timeout(3000)
                await self.page.evaluate(
                    f"""() => {{
                        window.initializeImageParams({{
                            imageFormat: \"{FORMAT}\",
                            imageQuality: {QUALITY}
                        }});
                    }}"""
                )
                await self._update_runtime_status()
            except Exception as exc:
                await self._invalidate_browser("Browser initialization failed", exc)
                raise

    async def _ensure_page(self) -> bool:
        try:
            await self.initialize_browser()
        except Exception:
            return False
        return self._session_is_live()

    async def _update_runtime_status(self) -> dict:
        if not self._session_is_live():
            return dict(self.last_runtime_status)
        try:
            page_status = await self.page.evaluate(
                """() => {
                    if (!window.getRemoteVideoStatus) {
                        return null;
                    }
                    return window.getRemoteVideoStatus();
                }"""
            )
        except Exception as exc:
            await self._invalidate_browser("Failed to read SDK runtime status", exc)
            return dict(self.last_runtime_status)

        if page_status:
            self.last_runtime_status = {
                "browser_initialized": self._session_is_live(),
                "joined_rtc": bool(page_status.get("joinedRtc")),
                "remote_video_published": bool(page_status.get("remoteVideoPublished")),
                "front_uid": page_status.get("frontUid"),
                "rear_uid": page_status.get("rearUid"),
                "remote_video_uids": page_status.get("remoteVideoUids") or [],
                "severity": page_status.get("severity", "info"),
                "message": page_status.get("message", ""),
                "last_error": page_status.get("lastError"),
            }
        return dict(self.last_runtime_status)

    async def status(self) -> dict:
        if not await self._ensure_page():
            return dict(self.last_runtime_status)
        return await self._update_runtime_status()

    def _selector_for_view(self, status: dict, view: str) -> str:
        if view == "rear":
            uid = status.get("rear_uid")
            if uid is None:
                uid = 1001
        else:
            uid = status.get("front_uid")
            if uid is None:
                uid = 1000
        return f"#player-{uid}"

    async def _wait_for_view(self, view: str, timeout_ms: int = 15000) -> bool:
        if not await self._ensure_page():
            return False
        status = await self._update_runtime_status()
        uid_key = "rear_uid" if view == "rear" else "front_uid"
        if status.get(uid_key) is not None:
            return True
        try:
            await self.page.wait_for_function(
                """(view) => {
                    if (!window.getRemoteVideoStatus) {
                        return false;
                    }
                    const status = window.getRemoteVideoStatus();
                    if (view === "rear") {
                        return status.rearUid !== null && status.rearUid !== undefined;
                    }
                    return status.frontUid !== null && status.frontUid !== undefined;
                }""",
                arg=view,
                timeout=timeout_ms,
            )
            await self._update_runtime_status()
            return True
        except PlaywrightTimeoutError:
            status = await self._update_runtime_status()
            if not status.get("message"):
                label = "rear" if view == "rear" else "front"
                self.last_runtime_status = {
                    **status,
                    "severity": "warning",
                    "message": (
                        f"Joined RTC channel, but no remote {label} video has been "
                        "published yet."
                    ),
                }
            return False
        except Exception as exc:
            await self._invalidate_browser(
                f"Lost browser session while waiting for {view} view", exc
            )
            return False

    async def take_screenshot(self, video_output_folder: str, elements: list):
        if not await self._ensure_page():
            return {}
        status = await self._update_runtime_status()
        try:
            dimensions = await self.page.evaluate(
                """() => {
                return {
                    width: Math.max(document.documentElement.scrollWidth, window.innerWidth),
                    height: Math.max(document.documentElement.scrollHeight, window.innerHeight),
                }
            }"""
            )

            if (
                dimensions["width"] > self.default_viewport["width"]
                or dimensions["height"] > self.default_viewport["height"]
            ):
                await self.page.set_viewport_size(dimensions)

            element_map = {
                "front": self._selector_for_view(status, "front"),
                "rear": self._selector_for_view(status, "rear"),
                "map": "#map",
            }

            screenshots = {}
            for name in elements:
                if name in element_map:
                    element_id = element_map[name]
                    output_path = f"{video_output_folder}/{name}.png"
                    element = await self.page.query_selector(element_id)
                    if element:
                        start_time = time.time()
                        await element.screenshot(path=output_path)
                        end_time = time.time()
                        elapsed_time = (end_time - start_time) * 1000
                        print(f"Screenshot for {name} took {elapsed_time:.2f} ms")
                        screenshots[name] = output_path
                    else:
                        print(f"Element {element_id} not found")
                else:
                    print(f"Invalid element name: {name}")

            return screenshots
        except Exception as exc:
            await self._invalidate_browser("Failed to capture screenshot", exc)
            return {}

    async def data(self) -> dict:
        if not await self._ensure_page():
            return None
        await self._update_runtime_status()
        try:
            bot_data = await self.page.evaluate(
                """() => {
            return window.rtm_data;
            }"""
            )
            return bot_data
        except Exception as exc:
            await self._invalidate_browser("Failed to read telemetry from browser", exc)
            return None

    async def front(self) -> str:
        if not await self._wait_for_view("front"):
            return None
        try:
            front_frame = await self.page.evaluate(
                """async () => {
                    const frame = await getLastBase64FrameForView("front");
                    return frame || null;
                }"""
            )
            return front_frame
        except Exception as exc:
            await self._invalidate_browser("Error capturing front frame", exc)
            return None

    async def rear(self) -> str:
        if not await self._wait_for_view("rear"):
            return None
        try:
            rear_frame = await self.page.evaluate(
                """async () => {
                    const frame = await getLastBase64FrameForView("rear");
                    return frame || null;
                }"""
            )
            return rear_frame
        except Exception as exc:
            await self._invalidate_browser("Error capturing rear frame", exc)
            return None

    async def send_message(self, message: dict):
        if not await self._ensure_page():
            raise RuntimeError(self.last_runtime_status.get("message", "Browser unavailable"))
        try:
            await self.page.evaluate(
                """(message) => {
                    window.sendMessage(message);
                }""",
                message,
            )
        except Exception as exc:
            await self._invalidate_browser("Failed to send message via browser", exc)
            raise

    async def close_browser(
        self,
        status_message: str | None = "Browser closed.",
        status_severity: str = "secondary",
        last_error: str | None = None,
    ):
        page = self.page
        browser = self.browser
        playwright = self.playwright
        self.page = None
        self.browser = None
        self.playwright = None

        if page:
            try:
                await page.close()
            except Exception as exc:
                print(f"Ignoring page close error: {exc}")
        if browser:
            try:
                await browser.close()
            except Exception as exc:
                print(f"Ignoring browser close error: {exc}")
        if playwright:
            try:
                await playwright.stop()
            except Exception as exc:
                print(f"Ignoring playwright stop error: {exc}")

        if status_message is not None:
            self.last_runtime_status = {
                **self.last_runtime_status,
                "browser_initialized": False,
                "joined_rtc": False,
                "remote_video_published": False,
                "front_uid": None,
                "rear_uid": None,
                "remote_video_uids": [],
                "severity": status_severity,
                "message": status_message,
                "last_error": last_error,
            }
