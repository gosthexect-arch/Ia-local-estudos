"""Diagnostico do modelo local servido pelo llama-server.

Verifica o arquivo GGUF, a saude do servidor, o contexto configurado e se o
modelo consegue devolver um alerta em JSON valido dentro do prazo aceitavel.
Nada aqui inicia, para ou reconfigura o servidor.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from cripto_monitor.config import AppConfig
from cripto_monitor.diagnostics.base import CheckResult, Status
from cripto_monitor.net.http import HttpError, get_json, request
from cripto_monitor.schema import EXAMPLE_ALERT, SCHEMA_VERSION, validate_alert
from cripto_monitor.term import human_bytes, human_ms

PROMPT_SISTEMA = (
    "Voce e um analista tecnico. Responda SOMENTE com um objeto JSON valido, "
    "sem texto antes ou depois, sem markdown. Nao invente dados."
)

PROMPT_TESTE = (
    "Teste de contrato do schema " + SCHEMA_VERSION + ". Nao ha dados de mercado nesta "
    "chamada, portanto a unica resposta correta e um alerta neutro.\n"
    "Responda exatamente com estes campos: " + json.dumps(EXAMPLE_ALERT, ensure_ascii=False)
)


def extract_json_object(text: str) -> Any:
    """Extrai o primeiro objeto JSON do texto, tolerando cercas de markdown."""
    limpo = text.strip()
    cerca = re.match(r"^```(?:json)?\s*(.*?)\s*```$", limpo, re.DOTALL)
    if cerca:
        limpo = cerca.group(1).strip()
    try:
        return json.loads(limpo)
    except json.JSONDecodeError:
        pass
    inicio = limpo.find("{")
    if inicio == -1:
        raise ValueError("nenhum objeto JSON encontrado na resposta")
    profundidade = 0
    em_string = False
    escapado = False
    for indice in range(inicio, len(limpo)):
        char = limpo[indice]
        if em_string:
            if escapado:
                escapado = False
            elif char == "\\":
                escapado = True
            elif char == '"':
                em_string = False
            continue
        if char == '"':
            em_string = True
        elif char == "{":
            profundidade += 1
        elif char == "}":
            profundidade -= 1
            if profundidade == 0:
                return json.loads(limpo[inicio : indice + 1])
    raise ValueError("objeto JSON incompleto na resposta")


def check_model_file(config: AppConfig) -> CheckResult:
    caminho = Path(config.llm.model_path).expanduser()
    if not caminho.exists():
        return CheckResult(
            name="arquivo_modelo",
            status=Status.WARN,
            summary=f"nao encontrado: {caminho}",
            details={"caminho": str(caminho)},
            hint="Ajuste llm.model_path se o GGUF estiver em outro diretorio.",
        )
    tamanho = caminho.stat().st_size
    legivel = caminho.is_file() and Path(caminho).stat().st_size > 0
    return CheckResult(
        name="arquivo_modelo",
        status=Status.OK if legivel else Status.WARN,
        summary=f"{caminho.name} ({human_bytes(tamanho)})",
        details={"caminho": str(caminho), "tamanho_bytes": tamanho},
    )


def check_server_health(config: AppConfig) -> CheckResult:
    url = f"{config.llm.server_root}/health"
    try:
        payload, resp = get_json(url, timeout=config.network.connect_timeout_s)
    except HttpError as exc:
        return CheckResult(
            name="llama_health",
            status=Status.FAIL,
            summary=str(exc),
            details={"url": url},
            hint=(
                "Suba o llama-server com a configuracao de referencia do projeto. "
                "Sem ele o sistema continua coletando, mas nao emite alerta direcional."
            ),
        )
    estado = payload.get("status", "desconhecido") if isinstance(payload, dict) else "?"
    ok = estado == "ok"
    return CheckResult(
        name="llama_health",
        status=Status.OK if ok else Status.WARN,
        summary=f"status={estado} em {human_ms(resp.elapsed_ms)}",
        details={"url": url, "resposta": payload, "latencia_ms": round(resp.elapsed_ms, 1)},
        hint=None if ok else "Servidor respondeu, mas ainda nao esta pronto para inferencia.",
    )


def check_server_props(config: AppConfig) -> CheckResult:
    url = f"{config.llm.server_root}/props"
    try:
        payload, _ = get_json(url, timeout=config.network.connect_timeout_s)
    except HttpError as exc:
        return CheckResult("llama_props", Status.SKIP, f"nao consultado: {exc}", {"url": url})
    if not isinstance(payload, dict):
        return CheckResult("llama_props", Status.WARN, "resposta inesperada", {"url": url})
    default = payload.get("default_generation_settings") or {}
    n_ctx = payload.get("n_ctx") or default.get("n_ctx")
    modelo = payload.get("model_path") or default.get("model") or payload.get("model")
    return CheckResult(
        name="llama_props",
        status=Status.OK,
        summary=f"contexto n_ctx={n_ctx}; modelo carregado: {Path(str(modelo)).name if modelo else '?'}",
        details={"n_ctx": n_ctx, "modelo": modelo, "chat_template": bool(payload.get("chat_template"))},
        hint="Contexto grande nao deve ser preenchido artificialmente (spec, secao 5).",
    )


def check_models_endpoint(config: AppConfig) -> CheckResult:
    url = f"{config.llm.base_url}/models"
    try:
        payload, resp = get_json(url, timeout=config.network.connect_timeout_s)
    except HttpError as exc:
        return CheckResult(
            name="llama_models",
            status=Status.FAIL,
            summary=str(exc),
            details={"url": url},
            hint="A API compativel com OpenAI e a interface usada pelo llm_client.",
        )
    ids = [m.get("id") for m in payload.get("data", [])] if isinstance(payload, dict) else []
    return CheckResult(
        name="llama_models",
        status=Status.OK if ids else Status.WARN,
        summary=f"{len(ids)} modelo(s) expostos: {', '.join(str(i) for i in ids[:3]) or 'nenhum'}",
        details={"url": url, "ids": ids, "latencia_ms": round(resp.elapsed_ms, 1)},
    )


def check_json_capability(config: AppConfig) -> CheckResult:
    """Pergunta ao modelo um alerta neutro e valida a resposta contra o schema."""
    url = f"{config.llm.base_url}/chat/completions"
    corpo = {
        "model": "local",
        "temperature": 0.0,
        "max_tokens": 512,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": PROMPT_SISTEMA},
            {"role": "user", "content": PROMPT_TESTE},
        ],
    }
    try:
        resp = request(
            url, method="POST", payload=corpo, timeout=config.llm.request_timeout_s
        )
    except HttpError as exc:
        return CheckResult(
            name="llama_json",
            status=Status.FAIL,
            summary=str(exc),
            details={"url": url},
            hint="Sem JSON estruturado o alert_engine deve permanecer em MODEL_UNAVAILABLE.",
        )
    try:
        conteudo = resp.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        return CheckResult(
            name="llama_json",
            status=Status.FAIL,
            summary=f"resposta fora do formato OpenAI: {exc}",
            details={"url": url, "corpo": resp.text[:400]},
        )
    detalhes: dict[str, Any] = {
        "url": url,
        "latencia_ms": round(resp.elapsed_ms, 1),
        "resposta_bruta": conteudo[:800],
    }
    try:
        objeto = extract_json_object(conteudo)
    except (ValueError, json.JSONDecodeError) as exc:
        return CheckResult(
            name="llama_json",
            status=Status.FAIL,
            summary=f"modelo nao devolveu JSON: {exc}",
            details=detalhes,
            hint="Avaliar grammar/GBNF ou json_schema no llama-server antes da Fase 4.",
        )
    erros = validate_alert(objeto)
    detalhes["erros_schema"] = erros
    detalhes["resposta_normalizada"] = objeto
    lento = resp.elapsed_ms > config.llm.max_latency_s * 1000
    if erros:
        status = Status.WARN
        resumo = f"JSON parseado, {len(erros)} erro(s) de schema em {human_ms(resp.elapsed_ms)}"
        dica = "O prompt da Fase 4 precisara de grammar ou few-shot para fechar o contrato."
    elif lento:
        status = Status.WARN
        resumo = f"alerta valido, porem em {human_ms(resp.elapsed_ms)}"
        dica = f"Acima do limite de {config.llm.max_latency_s:.0f}s entre fechamento e alerta."
    else:
        status = Status.OK
        resumo = f"alerta valido ({SCHEMA_VERSION}) em {human_ms(resp.elapsed_ms)}"
        dica = None
    return CheckResult("llama_json", status, resumo, detalhes, hint=dica)


def build_checks(config: AppConfig) -> list[tuple[str, object]]:
    return [
        ("arquivo_modelo", lambda: check_model_file(config)),
        ("llama_health", lambda: check_server_health(config)),
        ("llama_props", lambda: check_server_props(config)),
        ("llama_models", lambda: check_models_endpoint(config)),
        ("llama_json", lambda: check_json_capability(config)),
    ]
