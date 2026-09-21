"""Flask Web 层：单页展示 6 大板块恐贪情绪折线。"""
from __future__ import annotations

import os
import webbrowser
import threading

from flask import Flask, render_template

from . import config as C
from . import store

TPL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "templates")


def create_app(panels: dict | None = None) -> Flask:
    app = Flask(__name__, template_folder=TPL_DIR)
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.jinja_env.auto_reload = True
    _state = {"panels": panels, "payload": None, "summary": None}

    def ensure():
        if _state["panels"] is None:
            _state["panels"] = store.build_and_cache()
        if _state["payload"] is None:
            _state["payload"] = store.to_json(_state["panels"])
            _state["summary"] = store.summary(_state["panels"])
        return _state

    @app.route("/")
    def index():
        s = ensure()
        return render_template(
            "index.html",
            boards=[{"key": b, "name": C.BOARDS[b]["name"], "desc": C.BOARDS[b]["desc"]}
                    for b in C.BOARD_ORDER],
            payload=s["payload"],
            summary=s["summary"],
            meta=store.meta(),
            hold=store.H,
            tol=int(store.TOL * 100),
        )

    @app.route("/api/data")
    def api_data():
        s = ensure()
        return {"boards": s["payload"], "summary": s["summary"], "meta": store.meta()}

    return app


def serve(port: int | None = None, open_browser: bool = True, panels: dict | None = None):
    port = port or C.WEB_PORT
    app = create_app(panels)
    if open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{port}/")).start()
    print(f"情绪指标 v4 已启动: http://127.0.0.1:{port}/")
    app.run(host="127.0.0.1", port=port, debug=False)
