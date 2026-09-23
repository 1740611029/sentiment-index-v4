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
    _state = {"panels": panels, "payload": None, "summary": None, "stamp": None}

    def ensure():
        """面板 + 序列化结果按 meta.built_at 失效重算。

        为什么不能只算一次：`run.py refresh` 是在**另一个进程**里跑完的，
        常驻的服务进程如果一直抱着内存里的旧 payload，用户跑完更新、刷新页面
        还是看到旧数据（还要手动重启服务）。这里用 meta.json 的 built_at 当版本号。
        """
        if panels is not None:                      # 注入面板（测试用），只算一次
            if _state["payload"] is None:
                _state["payload"] = store.to_json(panels)
                _state["summary"] = store.summary(panels)
            return _state
        m = store.meta()
        stamp = (m.get("built_at"), m.get("last_date"))
        if _state["stamp"] != stamp:
            store.clear_caches()                    # 小波段面板有进程内缓存，必须清
            p = store.build_and_cache()
            _state.update({"panels": p, "payload": store.to_json(p),
                           "summary": store.summary(p), "stamp": stamp})
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
