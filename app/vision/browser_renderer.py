from playwright.async_api import async_playwright
from app.config.config import settings
from app.utils.logger import logger
import os
import shutil
import uuid
import sys
from typing import Optional

class BrowserRenderer:
    def __init__(self):
        self.headless = settings.playwright_headless
        self.temp_dir_base = os.path.join(os.getcwd(), "scratch", "previews")
        os.makedirs(self.temp_dir_base, exist_ok=True)

    @staticmethod
    def _extract_content(file_data) -> str:
        """
        Student file entries are expected to look like {"content": "..."}, but the API accepts
        loosely-typed JSON, so tolerate a bare string value too instead of crashing.
        """
        if isinstance(file_data, dict):
            return file_data.get("content", "") or ""
        if isinstance(file_data, str):
            return file_data
        return "" if file_data is None else str(file_data)

    @staticmethod
    def _safe_target_path(base_dir: str, filename: str) -> Optional[str]:
        """
        Resolve `filename` (attacker-controlled, from student submissions) against `base_dir`
        and make sure the result cannot escape `base_dir` via path traversal (e.g.
        "../../../etc/passwd"), a rooted/absolute path, or a Windows drive/UNC path.
        Returns None if the filename is unsafe and should be rejected.
        """
        if not filename or not isinstance(filename, str):
            return None

        # Reject absolute paths and drive letters/UNC paths outright.
        normalized_input = filename.replace("\\", "/")
        if normalized_input.startswith("/") or ":" in normalized_input.split("/")[0]:
            return None

        base_dir_real = os.path.realpath(base_dir)
        candidate = os.path.realpath(os.path.join(base_dir_real, normalized_input))

        # Ensure the resolved candidate path is still inside base_dir.
        if candidate != base_dir_real and not candidate.startswith(base_dir_real + os.sep):
            return None

        return candidate

    async def capture_screenshot(self, files: dict, viewport_width: int = 1280, viewport_height: int = 800) -> bytes:
        """
        Write the code files temporarily to local folders and take a full-page screenshot
        """
        if sys.platform == "win32" and sys.version_info >= (3, 13):
            logger.warning("Playwright preview rendering is disabled on Windows Python 3.13 due to subprocess compatibility issues.")
            return None

        session_id = str(uuid.uuid4())
        temp_session_dir = os.path.join(self.temp_dir_base, session_id)
        os.makedirs(temp_session_dir, exist_ok=True)

        logger.info(f"Preparing temporary preview files in: {temp_session_dir}")

        index_file_path = None
        for filename, file_data in (files or {}).items():
            content = self._extract_content(file_data)
            target_path = self._safe_target_path(temp_session_dir, filename)
            if target_path is None:
                logger.warning(f"Rejected unsafe/out-of-bounds student file path: {filename!r}")
                continue

            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            with open(target_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            if filename.lower() == "index.html" or filename.lower().endswith("index.html"):
                index_file_path = target_path

        if not index_file_path:
            for filename in files.keys():
                if filename.lower().endswith(".html"):
                    safe_path = self._safe_target_path(temp_session_dir, filename)
                    if safe_path and os.path.isfile(safe_path):
                        index_file_path = safe_path
                        break

        if not index_file_path:
            logger.warning("No entry point HTML found. Creating default page placeholder.")
            index_file_path = os.path.join(temp_session_dir, "index.html")
            with open(index_file_path, "w", encoding="utf-8") as f:
                f.write("<h1>DevBattles Preview</h1><p>Source files present, but no entry index.html was detected.</p>")

        screenshot_bytes = None
        try:
            logger.info("Launching Playwright chromium headless session...")
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=self.headless)
                context = await browser.new_context(
                    viewport={"width": viewport_width, "height": viewport_height}
                )
                page = await context.new_page()
                
                # Navigate using file:// protocol
                file_url = f"file:///{os.path.abspath(index_file_path).replace(os.sep, '/')}"
                logger.info(f"Opening rendering target URL: {file_url}")
                
                await page.goto(file_url, wait_until="networkidle", timeout=15000)
                await page.wait_for_timeout(1000)  # brief wait for rendering stability
                
                screenshot_bytes = await page.screenshot(full_page=True)
                logger.info("Headless page screenshot captured successfully.")
                
                await browser.close()
        except Exception as e:
            logger.error(f"Playwright visual rendering pipeline failed: {e}")
        finally:
            try:
                shutil.rmtree(temp_session_dir, ignore_errors=True)
                logger.info(f"Successfully cleaned session files for preview ID: {session_id}")
            except Exception as e:
                logger.warning(f"Could not delete temporary folders: {e}")

        return screenshot_bytes
