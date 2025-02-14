import webbrowser
import subprocess
import sys

def launch_web_broswer():
    def set_chrome_as_default_browser():
        try:
            subprocess.Popen(["reg", "add", "HKCU\Software\Classes\http\shell\open\command", "/t", "REG_SZ", "/d", "\"C:\Program Files\Google\Chrome\Application\chrome.exe\" -- %1", "/f"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.Popen(["reg", "add", "HKCU\Software\Classes\https\shell\open\command", "/t", "REG_SZ", "/d", "\"C:\Program Files\Google\Chrome\Application\chrome.exe\" -- %1", "/f"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            print(f"Error setting default browser: {e}")

    # Set Chrome as the default browser
    set_chrome_as_default_browser()

    # Open the specified URL in the default browser
    url = "http://127.0.0.1:8050/"
    webbrowser.open(url)