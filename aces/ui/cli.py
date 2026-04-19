"""Entry point for the `aces-ui` command."""
import os
import sys


def main() -> None:
    import streamlit.web.cli as stcli
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    sys.argv = ["streamlit", "run", app_path, "--server.headless", "false"]
    stcli.main()
