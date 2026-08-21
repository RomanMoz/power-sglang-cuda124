# Granite + Open WebUI (POWER9 / Tesla V100)

Стек: **IBM POWER9 (ppc64le)** + **2× NVIDIA Tesla V100-SXM2 16GB (sm70)** + **CUDA 12.4**.

Инференс идёт через форк [RomanMoz/power-sglang-cuda124](https://github.com/RomanMoz/power-sglang-cuda124): SGLang с патчем sm70 и колесом `power-sgl-kernel-cuda124`. Upstream SGLang на Volta не стартует (отрезает ниже sm75).

Open WebUI подключается только к **SGLang** (`runSglang.py`, OpenAI API на порту **30000**).

`runServer.py` (порт 8000, `POST /generate`) для WebUI не подходит.

Активная модель: `ibm-granite/granite-4.1-8b` (fp16, `--tp 2`, контекст до 8192). `granite-4.1-3b` остаётся в HF-кэше как запасной вариант на одну карту.

## Железо и runtime

| | |
| --- | --- |
| CPU | IBM POWER9, `ppc64le` |
| GPU | 2× Tesla V100-SXM2 16GB, compute **7.0 (sm70)** |
| CUDA | 12.4 |
| Python | 3.11 |
| Attention | `--attention-backend triton` (FA3/Flash на sm70 нет) |
| Dtype | `float16` (bf16/fp8 на Volta нет) |
| Kernel | `power-sgl-kernel-cuda124` 1.0.1, gencode sm70 |

Флаги по умолчанию в `runSglang.py` / `.env`:

```
--attention-backend triton --dtype float16 --tp 2 --mem-fraction-static 0.85 --max-model-len 8192
```

Колесо: [v1.0.1](https://github.com/RomanMoz/power-sglang-cuda124/releases/tag/v1.0.1)

```
pip install https://github.com/RomanMoz/power-sglang-cuda124/releases/download/v1.0.1/power_sgl_kernel_cuda124-1.0.1-cp310-abi3-linux_ppc64le.whl
```

## 4.1-3b vs 4.1-8b

| | `granite-4.1-3b` | `granite-4.1-8b` (сейчас) |
| --- | --- | --- |
| Параметры | 3B | 8B |
| Веса fp16 | ~6 ГБ | ~16 ГБ |
| На 2×V100 16GB | одна карта, больше KV | `--tp 2`, контекст ~4–8K |
| Архитектура | dense transformer (GQA, RoPE, SwiGLU) | то же |
| Лицензия | Apache 2.0 | Apache 2.0 |

Откат на 3b: в `.env` вернуть `MODEL_HUB_ID` / `MODEL_PATH` на `ibm-granite/granite-4.1-3b` и `--tp 1`.

## Hugging Face CLI: удалить старый кэш

```
export HF_HUB_CACHE=/home/powerai/Granite/models
python3 - <<'PY'
from huggingface_hub import scan_cache_dir
cm = scan_cache_dir("/home/powerai/Granite/models")
hashes = []
for repo in cm.repos:
    if repo.repo_id == "ibm-granite/granite-4.0-h-micro":
        hashes.extend(r.commit_hash for r in repo.revisions)
if hashes:
    cm.delete_revisions(*hashes).execute()
PY
```

## 1. Запустить Granite (SGLang)

На GPU-сервере остановите Transformers-сервис, если он занял видеокарты:

```
systemctl stop granite-server.service
```

Затем:

```
cd /home/powerai/Programming/AfpsGranite
python3 runSglang.py
```

Проверка:

```
curl http://127.0.0.1:30000/health
curl http://127.0.0.1:30000/v1/models
```

Нужен ответ со списком моделей. Этот URL и есть backend для Open WebUI: `http://HOST:30000/v1`.

## 2. Установить Open WebUI (Docker)

Нужен Docker. Образ: `ghcr.io/open-webui/open-webui:main` (то же самое: `openwebui/open-webui:main`).

Документация: https://docs.openwebui.com/getting-started/quick-start/

### Тот же Linux-хост, что и SGLang

```
docker pull ghcr.io/open-webui/open-webui:main

docker run -d \
  --name open-webui \
  --restart unless-stopped \
  -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -e ENABLE_OLLAMA_API=false \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:30000/v1 \
  -e OPENAI_API_KEY=sk-local \
  -v open-webui:/app/backend/data \
  ghcr.io/open-webui/open-webui:main
```

UI: http://localhost:3000  
Первый зарегистрированный пользователь становится admin.

### Open WebUI на другой машине

В `OPENAI_API_BASE_URL` поставьте IP GPU-сервера (порт 30000 должен быть доступен по сети):

```
docker run -d \
  --name open-webui \
  --restart unless-stopped \
  -p 3000:8080 \
  -e ENABLE_OLLAMA_API=false \
  -e OPENAI_API_BASE_URL=http://GPU_SERVER_IP:30000/v1 \
  -e OPENAI_API_KEY=sk-local \
  -v open-webui:/app/backend/data \
  ghcr.io/open-webui/open-webui:main
```

Ключ `sk-local` для локального SGLang можно любой; поле в Open WebUI обязательно.

### Подключить вручную в UI

Admin → Settings → Connections → OpenAI:

- API Base URL: `http://HOST:30000/v1`
- API Key: `sk-local`
- Save, затем выбрать модель из `/v1/models`

## 3. Установить Open WebUI (pip)

Python **3.11** или **3.12** (3.13 не поддерживается).

```
pip install open-webui
```

Запуск:

```
ENABLE_OLLAMA_API=false \
OPENAI_API_BASE_URL=http://127.0.0.1:30000/v1 \
OPENAI_API_KEY=sk-local \
open-webui serve
```

По умолчанию UI на http://localhost:8080.

Обновление:

```
pip install -U open-webui
```

## 4. Полезные команды Docker

```
docker logs -f open-webui
docker restart open-webui
docker stop open-webui
docker rm open-webui
```

Данные чатов в volume `open-webui`. Удаление контейнера volume не стирает.

## Порты

| Сервис | Порт | API |
| --- | --- | --- |
| Open WebUI (Docker) | 3000 | браузер |
| Open WebUI (pip) | 8080 | браузер |
| SGLang / `runSglang.py` | 30000 | `/v1/chat/completions` |
| Transformers / `runServer.py` | 8000 | не для WebUI |

Не запускайте `runServer.py` и `runSglang.py` одновременно на одних GPU.
