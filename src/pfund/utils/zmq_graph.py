from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import graphviz

    from pfund.engines.trade_engine import TradeEngine


# FIXME: currently only a draft
def show_zmq_graph(
    engine: TradeEngine,
    *,
    name: str = "zmq_graph",
    directory: str | None = None,
    fmt: str = "svg",
    view: bool = True,
    cleanup: bool = True,
) -> graphviz.Digraph:
    """Renders the live ZeroMQ topology of a trade engine as a directed graph.

    Dev-only helper (graphviz is a dev dependency). Call it AFTER `engine._setup()`
    has run, so the sockets are bound and `settings.zmq_urls/zmq_ports` are filled;
    before that the registry is empty and there is nothing to draw.

    The graph is reconstructed from the finalized registry plus the engine's
    component tree — it mirrors the wiring rules in `_setup_proxy` / `_setup_worker`
    / `DataBoy._subscribe`, so an edge means "this sender feeds this receiver":

    - data_engine XPUB  -> proxy SUB / component data SUB   (market data)
    - component data PUSH -> engine worker PULL             (orders)
    - proxy XPUB        -> component data SUB               (order/position/balance updates)
    - component logger PUB -> proxy SUB                     (logs, Ray only)
    - child signals PUB -> parent signals SUB              (signals up the component tree)

    pfeed only surfaces as the single `data_engine` boundary node — its plane-1
    fabric (faucet ROUTER, Ray worker DEALER/PUSH) lives in pfeed and isn't created
    until `run()`, so it can't be drawn from here.

    Args:
        engine: the trade engine to introspect (already `_setup()`-ed).
        name: base filename for the rendered file.
        directory: output directory; defaults to graphviz's cwd.
        fmt: output format passed to graphviz (e.g. "svg", "png", "pdf").
        view: if True, render to a file and open it in the OS default viewer.
        cleanup: if True, delete the intermediate .dot source after rendering.

    Returns:
        The `graphviz.Digraph` object (so it can be further tweaked or re-rendered).
    """
    try:
        import graphviz
    except ImportError:
        raise ImportError(
            "graphviz is a dev dependency and is not installed; "
            + "install it with `pixi add --feature dev graphviz` "
            + "(also needs the system `dot` binary, e.g. `brew install graphviz`)"
        )
    import re

    # message-type -> edge color, doubles as the legend
    COLORS = {
        "market data": "#1f77b4",
        "orders": "#d62728",
        "updates": "#2ca02c",
        "logs": "#7f7f7f",
        "signals": "#9467bd",
    }

    ports: dict[str, int] = dict(engine.settings.zmq_ports.items())
    urls: dict[str, str] = dict(engine.settings.zmq_urls.items())
    engine_name: str = engine.name

    def _nid(*parts: object) -> str:
        return re.sub(r"\W", "_", "__".join(str(p) for p in parts))

    def _addr(url: str | None, port: int | None) -> str:
        if url is None and port is None:
            return "(connects)"
        if port is None:
            return str(url)
        return f"{url or 'tcp://localhost'}:{port}"

    # ----- discover the actors from the finalized registry ---------------------
    # every component registers a "<name>_data" bind port; that's the reliable roster
    components: list[str] = sorted(
        k[: -len("_data")] for k in ports if k.endswith("_data")
    )
    loggers: set[str] = {k[: -len("_logger")] for k in ports if k.endswith("_logger")}

    # data_engine: prefer live socket ports off the object, fall back to the registry
    data_engine = getattr(engine, "_data_engine", None)
    de_xpub_port: int | None = ports.get("data_engine")
    de_pull_port: int | None = None
    de_url: str = urls.get("data_engine", "tcp://localhost")
    de_msg_queue = getattr(data_engine, "_msg_queue", None) if data_engine else None
    if de_msg_queue is not None:
        try:
            de_xpub_port = de_msg_queue.get_ports_in_use(de_msg_queue.sender)[0]
            de_pull_port = de_msg_queue.get_ports_in_use(de_msg_queue.receiver)[0]
        except Exception:
            pass
    has_data_engine: bool = "data_engine" in ports or de_msg_queue is not None

    # signals hierarchy (child -> parent) from the component tree
    signal_edges: list[tuple[str, str]] = []
    stack = list(getattr(engine, "_strategies", {}).values())
    while stack:
        component = stack.pop()
        for child in component.get_components():
            signal_edges.append((child.name, component.name))
            stack.append(child)

    # ----- build the graph -----------------------------------------------------
    dot = graphviz.Digraph(name=name, format=fmt)
    dot.attr(rankdir="LR", compound="true", labelloc="t", fontname="Helvetica")
    dot.attr(
        label=f"ZeroMQ topology — {engine_name} ({engine.env})",
        fontsize="18",
    )
    dot.attr("node", fontname="Helvetica", fontsize="10", style="filled")
    dot.attr("edge", fontname="Helvetica", fontsize="9")

    sender_style = {"shape": "box", "fillcolor": "#dbe9ff"}
    receiver_style = {"shape": "box", "fillcolor": "#ffffff"}

    # data engine (pfeed boundary)
    de_xpub = _nid("data_engine", "xpub")
    if has_data_engine:
        with dot.subgraph(name="cluster_data_engine") as c:
            c.attr(label="pfeed: DataEngine", style="dashed,rounded", color="#888888")
            c.node(
                de_xpub,
                f"XPUB · market data out\n{_addr(de_url, de_xpub_port)}",
                **sender_style,
            )
            c.node(
                _nid("data_engine", "pull"),
                f"PULL · from Ray workers\n{_addr(de_url, de_pull_port)}",
                **receiver_style,
            )

    # trade engine (proxy + worker)
    proxy_xpub = _nid("engine", "proxy_xpub")
    proxy_sub = _nid("engine", "proxy_sub")
    worker_pull = _nid("engine", "worker_pull")
    with dot.subgraph(name="cluster_engine") as c:
        c.attr(label=f"TradeEngine: {engine_name}", style="rounded", color="#333333")
        c.node(
            proxy_xpub,
            f"proxy XPUB · updates out\n{_addr(urls.get(engine_name), ports.get(engine_name))}",
            **sender_style,
        )
        c.node(proxy_sub, "proxy SUB · data + logs in\n(connects)", **receiver_style)
        c.node(worker_pull, "worker PULL · orders in\n(connects)", **receiver_style)

    # components (strategies + sub-components)
    for comp in components:
        c_url = urls.get(comp, "tcp://localhost")
        push = _nid(comp, "push")
        dsub = _nid(comp, "dsub")
        spub = _nid(comp, "spub")
        ssub = _nid(comp, "ssub")
        with dot.subgraph(name=f"cluster_{_nid(comp)}") as c:
            c.attr(label=comp, style="rounded", color="#555555")
            c.node(
                push,
                f"data PUSH · orders out\n{_addr(c_url, ports.get(comp + '_data'))}",
                **sender_style,
            )
            c.node(
                dsub,
                "data SUB · market data + updates in\n(connects)",
                **receiver_style,
            )
            c.node(
                spub,
                f"signals PUB · signals out\n{_addr(c_url, ports.get(comp))}",
                **sender_style,
            )
            c.node(
                ssub, "signals SUB · children signals in\n(connects)", **receiver_style
            )
            if comp in loggers:
                c.node(
                    _nid(comp, "log"),
                    f"logger PUB · logs out\n{_addr(c_url, ports.get(comp + '_logger'))}",
                    **sender_style,
                )

    # ----- edges (sender -> receiver) ------------------------------------------
    def _edge(src: str, dst: str, kind: str) -> None:
        dot.edge(src, dst, color=COLORS[kind], fontcolor=COLORS[kind])

    for comp in components:
        if has_data_engine:
            _edge(de_xpub, _nid(comp, "dsub"), "market data")
        _edge(_nid(comp, "push"), worker_pull, "orders")
        _edge(proxy_xpub, _nid(comp, "dsub"), "updates")
        if comp in loggers:
            _edge(_nid(comp, "log"), proxy_sub, "logs")
    if has_data_engine:
        _edge(de_xpub, proxy_sub, "market data")
    for child, parent in signal_edges:
        if child in components and parent in components:
            _edge(_nid(child, "spub"), _nid(parent, "ssub"), "signals")

    # legend
    with dot.subgraph(name="cluster_legend") as c:
        c.attr(label="legend", style="rounded", color="#aaaaaa", fontsize="9")
        prev: str | None = None
        for kind, color in COLORS.items():
            nid = _nid("legend", kind)
            c.node(nid, kind, shape="plaintext", style="", fontcolor=color)
            if prev is not None:
                c.edge(prev, nid, style="invis")
            prev = nid

    if view:
        dot.render(filename=name, directory=directory, view=True, cleanup=cleanup)
    return dot
