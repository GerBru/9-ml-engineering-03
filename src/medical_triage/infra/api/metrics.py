import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from prometheus_client import Counter, Histogram

REQUESTS = Counter(
    "http_requests_total",
    "Total de requisições HTTP",
    ["method", "path", "status"],
)
LATENCY = Histogram(
    "http_request_duration_seconds",
    "Duração das requisições HTTP em segundos",
    ["path"],
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

async def metrics_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    start = time.perf_counter()  # relógio monotônico, só serve para medir intervalos
    status_code = 500  # padrão: se a rota explodir, contamos como 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        # roda sempre, com sucesso ou exceção
        duration = time.perf_counter() - start
        route = request.scope.get("route")  # só existe depois do roteamento
        path = route.path if route else "unmatched"  # modelo da rota, não a URL crua
        REQUESTS.labels(method=request.method, path=path, status=str(status_code)).inc()
        LATENCY.labels(path=path).observe(duration)
