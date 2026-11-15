"""通用 Breeze TTS 兼容端点（AstrBot 插件）。

AstrBot 的「新建模型提供商」列表是内置固定的，插件无法往里加新卡片。
所以本插件在 AstrBot 进程内起一个 OpenAI 形状的 TTS 端点，
再用列表里现成的「OpenAI TTS(API)」卡片指向它（同一个容器，127.0.0.1 可达）。

- 音色不预设：由提供商表单里的 voice 决定（填 `voc_` 开头的 Breeze 音色 ID）
- 配置落盘到 runtime_config.json，插件重载后旧端点仍读最新配置
- 端口被占用时先复用已在跑的端点，否则自动顺延一个端口并打印实际值
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from astrbot.api import logger
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star

VERSION = "3.7.8"
SERVER_TAG = "breeze-tts-compat"
DEFAULT_BASE = "https://api.breeze.blue"
DEFAULT_PORT = 8788
RUNTIME_FILE = Path(__file__).parent / "runtime_config.json"

_CTX: dict = {"context": None}

DEFAULTS: dict = {
    "port": DEFAULT_PORT,
    "api_key": "",
    "base_url": DEFAULT_BASE,
    "voice_id": "",
    "default_voice": "",
    "model": "breeze-tts-2",
    "instructions": "",
    "guidance_scale": 4.0,
    "speed": 1.0,
    "timeout": 60,
    "max_chars": 500,
    "proxy": "",
    "voice_cache": "",
    "voice_fallback_only": False,
}

CFG: dict = dict(DEFAULTS)
_CACHE: dict = {"mtime": -1.0, "cfg": dict(DEFAULTS)}
_STATE: dict = {"server": None, "port": None, "error": "", "reused": False}


def current_cfg() -> dict:
    """读运行配置：优先 runtime_config.json（重载后旧实例也能拿到新值）。"""
    try:
        st = RUNTIME_FILE.stat()
        if st.st_mtime != _CACHE["mtime"]:
            data = json.loads(RUNTIME_FILE.read_text(encoding="utf-8"))
            merged = dict(DEFAULTS)
            merged.update({k: v for k, v in data.items() if v not in (None, "")})
            _CACHE["cfg"] = merged
            _CACHE["mtime"] = st.st_mtime
    except Exception:  # noqa: BLE001
        pass
    return _CACHE["cfg"]


# ---------------- 核心配置「语音回复」下方的音色子项 -------------------------

VOICE_KEY = "provider_tts_settings.breeze_voice_id"
ANCHOR_KEY = "provider_tts_settings.enable"
VOICE_ITEM = {
    "description": "语音回复音色 ID",
    "hint": "Breeze 音色 ID（voc_ 开头）。填了就先用它，优先于提供商里的 voice。",
    "type": "string",
}


def _insert_voice_item(items: dict) -> bool:
    """把音色项插在 provider_tts_settings.* 这组的最末尾。

    插在内置项中间会打乱后面几项（「默认文本转语音模型」「TTS 触发概率」）
    在配置页上的显示，所以只接在这一组的尾部。
    """
    if VOICE_KEY in items:
        return True
    if ANCHOR_KEY not in items:
        return False
    last_tts_key = None
    for key in items:
        if key.startswith("provider_tts_settings."):
            last_tts_key = key
    if last_tts_key is None:
        return False
    rebuilt: dict = {}
    for key, value in items.items():
        rebuilt[key] = value
        if key == last_tts_key:
            rebuilt[VOICE_KEY] = dict(VOICE_ITEM)
    items.clear()
    items.update(rebuilt)
    return True


def _restore_voice_text(meta) -> None:
    """把被 i18n 换成键名的音色子项文字换回中文。

    convert_to_i18n_keys 会把每一项的 description/hint 无条件改成
    `ai_group.ai.<字段>.description` 这样的键名，而词条只覆盖内置字段。
    这里只还原插件自己那一项，其它字段保持 AstrBot 原样。
    """
    if isinstance(meta, dict):
        for key, value in meta.items():
            if key == VOICE_KEY and isinstance(value, dict):
                value["description"] = VOICE_ITEM["description"]
                value["hint"] = VOICE_ITEM["hint"]
            _restore_voice_text(value)
    elif isinstance(meta, list):
        for item in meta:
            _restore_voice_text(item)


def patch_i18n() -> bool:
    """让配置页元数据转换时保留插件那一项的中文文字。"""
    try:
        from astrbot.core.config import i18n_utils

        cls = i18n_utils.ConfigMetadataI18n
        if getattr(cls, "_breeze_patched", False):
            return True
        original = cls.convert_to_i18n_keys

        def convert(metadata):
            result = original(metadata)
            try:
                _restore_voice_text(result)
            except Exception:  # noqa: BLE001
                pass
            return result

        cls.convert_to_i18n_keys = staticmethod(convert)
        cls._breeze_patched = True
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning(f"breeze_tts: 中文标签补丁未生效（不影响功能）：{e}")
        return False


def inject_voice_item(schema: dict) -> bool:
    """递归找到含「语音回复」的那一层，把音色项接进 provider_tts_settings 组。"""
    if not isinstance(schema, dict):
        return False
    hit = False

    def walk(node, depth: int = 0) -> None:
        nonlocal hit
        if not isinstance(node, dict) or depth > 10:
            return
        if ANCHOR_KEY in node:
            hit = _insert_voice_item(node) or hit
        for key, value in list(node.items()):
            # condition 里也有 provider_tts_settings.enable，走进去会自我递归
            if key in ("condition", VOICE_KEY):
                continue
            if isinstance(value, dict):
                walk(value, depth + 1)

    walk(schema)
    return hit


def _collect_schemas(context=None) -> list[dict]:
    found: list[dict] = []
    try:
        from astrbot.core.config.default import CONFIG_METADATA_3

        found.append(CONFIG_METADATA_3)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"breeze_tts: 读取 CONFIG_METADATA_3 失败：{e}")
    try:
        ctx = context or _CTX.get("context")
        cfg = ctx.get_config() if ctx else None
        schema = getattr(cfg, "schema", None)
        if isinstance(schema, dict):
            found.append(schema)
    except Exception:  # noqa: BLE001
        pass
    return found


def inject_default_config() -> bool:
    """把 breeze_voice_id 也补进 DEFAULT_CONFIG。

    AstrBotConfig.check_config_integrity 以 DEFAULT_CONFIG 为参照，
    参照里没有的键会在启动/保存时被删掉（日志 "Config key removed: ..."），
    所以「显示用的 schema」和「参照默认值」两处都必须有，光补 schema 存不住。
    """
    try:
        from astrbot.core.config.default import DEFAULT_CONFIG
    except Exception as e:  # noqa: BLE001
        logger.warning(f"breeze_tts: 读取 DEFAULT_CONFIG 失败：{e}")
        return False
    settings = DEFAULT_CONFIG.get("provider_tts_settings")
    if not isinstance(settings, dict):
        settings = {}
        DEFAULT_CONFIG["provider_tts_settings"] = settings
    key = VOICE_KEY.split(".", 1)[1]
    if key not in settings:
        settings[key] = ""
    return True


def push_default_voice_value(value: str) -> None:
    """把音色值写进 DEFAULT_CONFIG 的同一项。

    配置页拿到的展示用 config 可能就是 DEFAULT_CONFIG（
    `ConfigProfileService.get_profile_schema()` 直接返回 `"config": DEFAULT_CONFIG`），
    只往里塞空字符串的话，界面上永远显示为空。
    """
    try:
        from astrbot.core.config.default import DEFAULT_CONFIG
    except Exception:  # noqa: BLE001
        return
    try:
        settings = DEFAULT_CONFIG.get("provider_tts_settings")
        if isinstance(settings, dict):
            settings[VOICE_KEY.split(".", 1)[1]] = value
    except Exception:  # noqa: BLE001
        pass


def ensure_live_voice_key(context=None) -> bool:
    """让运行中的配置对象里也存在该键，页面才会把用户输入带进保存请求。"""
    ctx = context or _CTX.get("context")
    if ctx is None:
        return False
    try:
        cfg = ctx.get_config()
        settings = cfg.get("provider_tts_settings")
        if not isinstance(settings, dict):
            return False
        key = VOICE_KEY.split(".", 1)[1]
        if key not in settings:
            settings[key] = ""
        return True
    except Exception:  # noqa: BLE001
        return False


def hook_config_save() -> None:
    """在 AstrBot 保存配置的瞬间截获 AI 页填的音色。

    dashboard 的保存最终走 AstrBotConfig.save_config(post_config)，提交体里可能带着
    用户输入的 breeze_voice_id；但重启时该键会被 check_config_integrity 当成未知键删掉。
    所以在这一刻就把它固化进 runtime_config.json：音色立刻生效，重启也不丢；
    同时把「提交体里到底有没有这个字段」写进日志，避免再靠猜。
    """
    try:
        from astrbot.core.config.astrbot_config import AstrBotConfig
    except Exception as e:  # noqa: BLE001
        logger.warning(f"breeze_tts: 挂配置保存钩子失败：{e}")
        return
    if getattr(AstrBotConfig, "_breeze_save_hooked", False):
        return

    key = VOICE_KEY.split(".", 1)[1]

    def capture(data) -> None:
        try:
            settings = (data or {}).get("provider_tts_settings")
            if not isinstance(settings, dict):
                return
            if key not in settings:
                logger.info(
                    "breeze_tts: 保存配置时光标里没有「语音回复音色 ID」——配置页没提交该字段"
                )
                return
            value = str(settings.get(key) or "").strip()
            logger.info(f"breeze_tts: 保存配置时截获音色 {value or '（空）'}")
            if value and value != str(CFG.get("voice_cache") or "").strip():
                CFG["voice_cache"] = value
                RUNTIME_FILE.write_text(
                    json.dumps(CFG, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                _CACHE.update({"mtime": -1.0, "cfg": dict(CFG)})
                push_default_voice_value(value)
                logger.info(f"breeze_tts: AI 页音色 {value} 已立即生效（并写入缓存）")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"breeze_tts: 截获音色失败：{e}")

    orig = AstrBotConfig.save_config

    def save_config(self, replace_config=None, *, indent=2):
        capture(replace_config if isinstance(replace_config, dict) else dict(self))
        return orig(self, replace_config, indent=indent)

    AstrBotConfig.save_config = save_config
    AstrBotConfig._breeze_save_hooked = True
    logger.info("breeze_tts: 已挂上配置保存钩子（在配置页保存音色会立刻记住）")


def try_inject_schema(context=None) -> bool:
    """往候选 schema 里注入音色项；成功后再挂上中文还原补丁。"""
    inject_default_config()
    ensure_live_voice_key(context)
    hook_config_save()
    done = False
    for schema in _collect_schemas(context):
        try:
            done = inject_voice_item(schema) or done
        except Exception as e:  # noqa: BLE001
            logger.warning(f"breeze_tts: 注入配置项失败：{e}")
    if done:
        patch_i18n()
    else:
        logger.warning(
            "breeze_tts: 未能注入「语音回复音色 ID」，音色请填插件配置里的 voice_id"
        )
    return done


def plugin_voice_id() -> str:
    """读插件配置面板里的「语音回复音色 ID」。"""
    return str(current_cfg().get("voice_id") or "").strip()


def core_voice_id() -> str:
    """读核心配置（AI → 语音回复）里的音色，读不到就用插件缓存的值。"""
    value = ""
    ctx = _CTX.get("context")
    if ctx is not None:
        try:
            cfg = ctx.get_config()
            settings = cfg.get("provider_tts_settings") or {}
            value = str(settings.get("breeze_voice_id") or "").strip()
        except Exception:  # noqa: BLE001
            value = ""
    if value:
        return value
    return str(current_cfg().get("voice_cache") or "").strip()


def effective_voice_id() -> str:
    """配置栏音色：AI 页那一项优先，其次插件配置面板那一项。"""
    return core_voice_id() or plugin_voice_id()


def remember_ai_voice() -> None:
    """把 AI 页填的音色固化到 runtime_config.json。

    AstrBot 重启时会把不在参照 schema 里的配置键清掉（插件注入的
    `provider_tts_settings.breeze_voice_id` 就属于这类），所以读到值就自己存一份，
    重启后仍能从缓存里取回来。
    """
    value = ""
    ctx = _CTX.get("context")
    if ctx is None:
        return
    try:
        settings = ctx.get_config().get("provider_tts_settings") or {}
        value = str(settings.get(VOICE_KEY.split(".", 1)[1]) or "").strip()
    except Exception:  # noqa: BLE001
        return
    if not value or value == str(CFG.get("voice_cache") or "").strip():
        return
    CFG["voice_cache"] = value
    try:
        RUNTIME_FILE.write_text(
            json.dumps(CFG, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _CACHE.update({"mtime": -1.0, "cfg": dict(CFG)})
        push_default_voice_value(value)
        logger.info(f"breeze_tts: 已记住 AI 页音色 {value}（重启后仍生效）")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"breeze_tts: 缓存 AI 页音色失败：{e}")


def pick_voice(
    core_voice: str,
    request_voice: str,
    default_voice: str,
    fallback_only: bool = False,
) -> str:
    """优先级：配置栏音色 > 提供商里的 voice > 插件默认音色。

    「配置栏音色」填了就一律用它（不校验 voc_ 前缀，写错了让上游报错，
    也不能静默忽略用户填的值）；提供商 voice 只认 voc_ 开头的，
    因为内置模板默认是 alloy 这种 OpenAI 音色名。

    fallback_only=True（插件配置「按角色保留音色」）时反过来：请求里明确带了
    voc_ 开头的 voice，就按请求走，AI 页那一项只在该请求没带 voc_ 时兜底。
    这样多个角色各自挂一个提供商、各填各的 voice 时不会互相覆盖。
    """
    request = (request_voice or "").strip()
    if fallback_only and request.startswith("voc_"):
        return request
    value = (core_voice or "").strip()
    if value:
        return value
    if request.startswith("voc_"):
        return request
    return (default_voice or "").strip()


def resolve_voice(request_voice: str, default_voice: str) -> tuple[str, str]:
    """返回（实际使用的音色, 来源标签），用于日志与自检。"""
    core = effective_voice_id()
    fallback_only = bool(current_cfg().get("voice_fallback_only"))
    voice = pick_voice(core, request_voice, default_voice, fallback_only)
    if fallback_only and voice == (request_voice or "").strip() and voice.startswith("voc_"):
        return voice, "提供商voice（按角色，AI页仅兜底）"
    if core and voice == core.strip():
        ai = core_voice_id()
        source = "AI页音色ID" if ai and ai.strip() == voice else "插件配置voice_id"
    elif voice and voice == (request_voice or "").strip():
        source = "提供商voice"
    elif voice:
        source = "插件default_voice"
    else:
        source = "无"
    return voice, source


# ---------------- 文本与请求 ----------------------------


def split_text(text: str, limit: int = 500) -> list[str]:
    """按标点切成不超过 limit 的片段，避免被 Breeze 直接拒绝。"""
    text = (text or "").strip()
    if limit <= 0:
        limit = 500
    if len(text) <= limit:
        return [text] if text else []

    marks = "。！？!?；;…\n"
    chunks: list[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if len(buf) >= limit:
            cut = -1
            for i in range(len(buf) - 1, max(-1, len(buf) - 60), -1):
                if buf[i] in marks:
                    cut = i
                    break
            if cut >= 0:
                chunks.append(buf[: cut + 1].strip())
                buf = buf[cut + 1:]
            else:
                chunks.append(buf.strip())
                buf = ""
        elif ch in marks and len(buf) >= limit * 0.6:
            chunks.append(buf.strip())
            buf = ""
    if buf.strip():
        chunks.append(buf.strip())
    return [c for c in chunks if c]


def build_payload(text: str, model: str, instructions: str, guidance: float, speed: float) -> dict:
    body: dict = {"text": text, "language_code": "zh"}
    if model:
        body["model_id"] = model
    if instructions:
        body["instructions"] = instructions
    vs: dict = {}
    if guidance and guidance > 0:
        vs["guidance_scale"] = guidance
    if speed and speed > 0 and speed != 1.0:
        vs["speed"] = speed
    if vs:
        body["voice_settings"] = vs
    return body


def _open(req, timeout: int):
    """单独抽出来，方便测试替换。"""
    cfg = current_cfg()
    proxy = str(cfg.get("proxy") or "")
    if proxy:
        handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        opener = urllib.request.build_opener(handler)
        return opener.open(req, timeout=timeout)
    return urllib.request.urlopen(req, timeout=timeout)


def synthesize(text: str, voice: str, out_format: str = "mp3", api_key: str = "") -> bytes:
    """调 Breeze 合成，返回音频字节。失败抛 RuntimeError。"""
    cfg = current_cfg()
    key = (api_key or str(cfg.get("api_key") or "") or os.getenv("BREEZE_API_KEY", "")).strip()
    if not key:
        raise RuntimeError("未配置 Breeze API Key")
    if not voice:
        raise RuntimeError("未配置音色（voice 需为 voc_ 开头的音色 ID）")

    base = str(cfg.get("base_url") or DEFAULT_BASE).rstrip("/")
    pieces = split_text(text, int(cfg.get("max_chars") or 500))
    audio = bytearray()
    timeout = int(cfg.get("timeout") or 60)

    for piece in pieces:
        url = f"{base}/v1/text-to-speech/{voice}?output_format={out_format or 'mp3'}"
        payload = json.dumps(
            build_payload(
                piece,
                str(cfg.get("model") or ""),
                str(cfg.get("instructions") or ""),
                float(cfg.get("guidance_scale") or 0),
                float(cfg.get("speed") or 0),
            )
        ).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"xi-api-key": key, "content-type": "application/json"},
            method="POST",
        )

        last = "未知错误"
        for attempt in range(3):
            try:
                with _open(req, timeout) as resp:
                    data = resp.read()
                if not data:
                    raise RuntimeError("上游返回空音频")
                audio.extend(data)
                break
            except urllib.error.HTTPError as e:
                detail = (e.read() or b"")[:200].decode("utf-8", "replace")
                last = f"{e.code} {detail}"
                if e.code == 422 and (out_format or "").lower() != "mp3":
                    # 该格式不被接受时退回 mp3
                    return synthesize(text, voice, "mp3", key)
                if e.code not in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"Breeze 合成失败：{last}")
                logger.warning(f"breeze_tts: {last}，第 {attempt + 1} 次失败")
                time.sleep(1.0 + attempt)
            except RuntimeError:
                raise
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__}: {e}"
                logger.warning(f"breeze_tts: {last}")
                time.sleep(1.0 + attempt)
        else:
            raise RuntimeError(f"Breeze 合成失败：{last}")

    return bytes(audio)


# ---------------- HTTP 端点 ----------------------------


class _Handler(BaseHTTPRequestHandler):
    server_version = f"BreezeTTSCompat/{VERSION}"

    def log_message(self, fmt, *args):  # 静默，避免刷屏
        return

    def _json(self, status: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0].rstrip("/")
        cfg = current_cfg()
        if path == "/health":
            self._json(
                200,
                {
                    "ok": True,
                    "server": SERVER_TAG,
                    "version": VERSION,
                    "model": cfg.get("model"),
                    "default_voice": cfg.get("default_voice"),
                    "key_present": bool(cfg.get("api_key")),
                },
            )
            return
        if path == "/v1/models":
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": cfg.get("model") or "breeze-tts-2",
                            "object": "model",
                            "owned_by": "breeze",
                        }
                    ],
                },
            )
            return
        self._json(404, {"error": {"message": "Not Found", "code": "not_found"}})

    def do_POST(self):  # noqa: N802
        path = self.path.split("?")[0].rstrip("/")
        # 先把请求体读干净，避免未读 body 导致连接被重置（WinError 10053 / RST）
        try:
            length = int(self.headers.get("content-length") or 0)
            raw = self.rfile.read(length) if length > 0 else b""
        except Exception:  # noqa: BLE001
            raw = b""

        if path != "/v1/audio/speech":
            self._json(404, {"error": {"message": "Not Found", "code": "not_found"}})
            return

        try:
            body = json.loads(raw or b"{}")
        except Exception:  # noqa: BLE001
            self._json(400, {"error": {"message": "请求体不是合法 JSON", "code": "bad_json"}})
            return

        remember_ai_voice()
        cfg = current_cfg()
        text = str(body.get("input") or body.get("text") or "").strip()
        if not text:
            self._json(400, {"error": {"message": "缺少 input", "code": "missing_input"}})
            return

        voice, source = resolve_voice(
            str(body.get("voice") or ""),
            str(cfg.get("default_voice") or ""),
        )
        if not voice:
            self._json(
                400,
                {
                    "error": {
                        "message": "没有可用音色：请在配置「语音回复音色 ID」或提供商 voice 里填音色 ID（形如 voc_xxx）",
                        "code": "missing_voice",
                    }
                },
            )
            return

        out_format = str(body.get("response_format") or "mp3").strip().lower() or "mp3"

        auth = self.headers.get("authorization") or ""
        key = auth.split(" ", 1)[1].strip() if auth.lower().startswith("bearer ") else ""

        try:
            t0 = time.monotonic()
            audio = synthesize(text, voice, out_format, key)
        except Exception as e:  # noqa: BLE001
            logger.error(f"breeze_tts: {e}")
            self._json(502, {"error": {"message": str(e), "code": "upstream_error"}})
            return
        logger.info(
            f"breeze_tts: 合成成功 voice={voice}（来源：{source}）字数={len(text)} "
            f"音频={len(audio)}B 耗时={time.monotonic() - t0:.1f}s"
        )

        ctype = "audio/mpeg" if out_format == "mp3" else f"audio/{out_format}"
        self.send_response(200)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(audio)))
        self.end_headers()
        self.wfile.write(audio)


def _probe(port: int) -> bool:
    """判断该端口上跑的是不是本插件自己的端点。"""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
            data = json.loads(r.read())
        return data.get("server") == SERVER_TAG
    except Exception:  # noqa: BLE001
        return False


def start_server(port: int) -> dict:
    """起端点；已存在则复用，端口被外部占用则顺延。返回状态字典。"""
    if _STATE["server"] is not None:
        return {"port": _STATE["port"], "reused": False, "error": ""}

    if _probe(port):
        _STATE.update({"port": port, "reused": True, "error": ""})
        logger.info(f"breeze_tts v{VERSION}: 端口 {port} 上已有本插件端点，直接复用")
        return {"port": port, "reused": True, "error": ""}

    tried = []
    for candidate in range(port, port + 6):
        if candidate != port and _probe(candidate):
            _STATE.update({"port": candidate, "reused": True, "error": ""})
            logger.info(f"breeze_tts v{VERSION}: 复用端口 {candidate} 上已有的端点")
            return {"port": candidate, "reused": True, "error": ""}
        try:
            httpd = ThreadingHTTPServer(("127.0.0.1", candidate), _Handler)
        except OSError as e:
            tried.append(f"{candidate}: {e}")
            continue

        threading.Thread(
            target=httpd.serve_forever, name=f"breeze-tts-http-{candidate}", daemon=True
        ).start()
        _STATE.update({"server": httpd, "port": candidate, "reused": False, "error": ""})
        logger.info(f"breeze_tts v{VERSION}: 端点已启动 http://127.0.0.1:{candidate}/v1")
        if candidate != port:
            logger.warning(
                f"breeze_tts v{VERSION}: 原端口 {port} 被占用，已改用 {candidate}；"
                f"请把提供商的 API Base URL 改成 http://127.0.0.1:{candidate}/v1"
            )
        return {"port": candidate, "reused": False, "error": ""}

    _STATE["error"] = "端口全部不可用：" + "; ".join(tried)
    return {"port": None, "reused": False, "error": _STATE["error"]}


# ---------------- 插件入口 -----------------------------


class Main(Star):
    def __init__(self, context: Context, config: dict | None = None) -> None:
        super().__init__(context)
        self.context = context
        self.config = config or {}
        _CTX["context"] = context
        self._apply_config()
        injected = try_inject_schema(context)
        self._sync_voice_setting()

        state = start_server(int(CFG["port"]))
        if state["error"]:
            logger.error(f"breeze_tts v{VERSION}: {state['error']}")
        logger.info(
            f"breeze_tts v{VERSION} 已加载 | 端口={state['port']} 模型={CFG['model']} "
            f"key={'已配置' if CFG['api_key'] else '未配置'} "
            f"音色={'已配置' if effective_voice_id() else '未配置（用提供商里的 voice）'} "
            f"AI页音色项={'已注入' if injected else '未注入（用插件配置的 voice_id）'} "
            f"AI页音色ID={core_voice_id() or '空'}"
        )

    def _apply_config(self) -> None:
        merged = dict(DEFAULTS)
        try:
            if RUNTIME_FILE.is_file():
                old = json.loads(RUNTIME_FILE.read_text(encoding="utf-8"))
                if str(old.get("voice_cache") or "").strip():
                    merged["voice_cache"] = str(old["voice_cache"]).strip()
        except Exception:  # noqa: BLE001
            pass
        for key in (
            "api_key",
            "base_url",
            "voice_id",
            "default_voice",
            "model",
            "instructions",
            "proxy",
        ):
            v = self.config.get(key)
            if v:
                merged[key] = str(v).strip()
        v = self.config.get("voice_fallback_only")
        if v is not None:
            merged["voice_fallback_only"] = bool(v)
        for key, caster, fallback in (
            ("port", int, DEFAULT_PORT),
            ("guidance_scale", float, 4.0),
            ("speed", float, 1.0),
            ("timeout", int, 60),
            ("max_chars", int, 500),
        ):
            try:
                v = self.config.get(key)
                merged[key] = caster(v) if v not in (None, "") else fallback
            except (TypeError, ValueError):
                merged[key] = fallback
        CFG.update(merged)
        _CACHE.update({"mtime": -1.0, "cfg": merged})
        try:
            RUNTIME_FILE.write_text(
                json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"breeze_tts: 写入运行配置失败：{e}")

    def _sync_voice_setting(self) -> None:
        """把 AI 页那一项与插件侧缓存对齐（双向）。

        AstrBot 只保存它 schema 里认识的键，插件注入的 `breeze_voice_id` 在重启时会被
        当成未知配置清掉（与 astrbot_plugin_zhipu_search 的结论一致）。
        所以：读到值就缓存进 runtime_config.json；读不到就用缓存回填，
        保证页面上那个输入框重启后仍显示上次填的音色。
        """
        try:
            cfg = self.context.get_config()
            settings = cfg.get("provider_tts_settings")
            if not isinstance(settings, dict):
                return
            key = VOICE_KEY.split(".", 1)[1]
            value = str(settings.get(key) or "").strip()
            cached = str(CFG.get("voice_cache") or "").strip()
            if value and value != cached:
                CFG["voice_cache"] = value
                RUNTIME_FILE.write_text(
                    json.dumps(CFG, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                _CACHE.update({"mtime": -1.0, "cfg": dict(CFG)})
            elif not value and cached:
                settings[key] = cached
                push_default_voice_value(cached)
                try:
                    cfg.save_config()
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"breeze_tts: 回填音色到配置失败：{e}")
                logger.info(
                    f"breeze_tts: 已把缓存音色 {cached} 回填到 AI 页「语音回复音色 ID」"
                )
            elif value:
                push_default_voice_value(value)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"breeze_tts: 同步音色设置失败：{e}")

    def _core_voice_probe(self) -> tuple[bool, str]:
        """（运行中配置里是否存在该键, 读到的值）——用于自检定位。"""
        ctx = _CTX.get("context")
        if ctx is None:
            return False, ""
        try:
            settings = ctx.get_config().get("provider_tts_settings") or {}
        except Exception:  # noqa: BLE001
            return False, ""
        key = VOICE_KEY.split(".", 1)[1]
        if key not in settings:
            return False, ""
        return True, str(settings.get(key) or "").strip()

    @filter.command("tts状态")
    async def status(self, event: AstrMessageEvent):
        """查看端点状态与版本。"""
        cfg = current_cfg()
        running = _STATE["server"] is not None or _STATE["reused"]
        core_voice = core_voice_id()
        plugin_voice = plugin_voice_id()
        core_present, core_raw = self._core_voice_probe()
        lines = [
            f"Breeze TTS v{VERSION}",
            f"端点：{'http://127.0.0.1:' + str(_STATE['port']) + '/v1' if running else '未启动'}",
            f"模型：{cfg.get('model')}",
            f"AI 页音色（语音回复音色 ID）：{core_raw if core_present else '配置里没有该键'}"
            + ("" if core_present else "（未填过，或保存时被 AstrBot 清理）"),
            f"插件配置音色（voice_id）：{plugin_voice or '未填'}",
            f"实际生效音色：{effective_voice_id() or '无（回落到提供商 voice / 兜底音色）'}",
            f"兜底音色：{cfg.get('default_voice') or '未配置（由提供商表单的 voice 决定）'}",
            f"Key：{'已配置' if cfg.get('api_key') else '未配置（也可在提供商里填）'}",
        ]
        if _STATE["error"]:
            lines.append(f"错误：{_STATE['error']}")
        if running:
            lines.append(
                f"提供商 Base URL 应为 http://127.0.0.1:{_STATE['port']}/v1"
                "（服务提供商 → 新增 → OpenAI TTS(API)）"
            )
        yield event.plain_result("\n".join(lines))
