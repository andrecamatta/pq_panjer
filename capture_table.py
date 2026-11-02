import sys
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def capture(html_path, png_path, width=900):
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(f"--window-size={width},600")
    options.add_argument("--force-device-scale-factor=2")
    options.add_argument("--hide-scrollbar")
    driver = webdriver.Chrome(options=options)
    try:
        driver.get("file://" + html_path)
        time.sleep(0.3)
        content_height = driver.execute_script(
            "return document.body.scrollHeight"
        )
        driver.set_window_size(width, max(600, content_height + 250))
        time.sleep(0.3)
        el = driver.find_element("id", "capture")
        el.screenshot(png_path)
    finally:
        driver.quit()

if __name__ == "__main__":
    html_path, png_path = sys.argv[1], sys.argv[2]
    width = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    capture(html_path, png_path, width)
    print("saved", png_path)
