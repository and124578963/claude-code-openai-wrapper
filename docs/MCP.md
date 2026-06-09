# MCP через wrapper — агентский режим

Wrapper поддерживает проброс MCP-серверов **на уровне запроса** (per-request).
Серверы и их инструменты подключаются к агенту Claude для конкретного
чата/сессии через поле `mcp_servers` в теле `/v1/chat/completions`.

## Почему `claude mcp add` НЕ работает

Wrapper намеренно изолирует файловые настройки CLI (`setting_sources=[]`),
чтобы локальная персона Claude Code, скиллы, хуки и CLAUDE.md не «протекали»
в OpenAI-совместимый API. Побочный эффект: серверы из `~/.claude.json` и
`.mcp.json` (то, что добавляет `claude mcp add`) **не загружаются**.

Единственный поддерживаемый путь — поле `mcp_servers` в запросе. Оно
передаётся в Claude Agent SDK программно (`ClaudeAgentOptions.mcp_servers`)
вместе с флагом `--strict-mcp-config`, так что используется ровно то, что
указано в запросе — ничего из файловых конфигов.

## Поля запроса

| Поле | Тип | Описание |
|---|---|---|
| `mcp_servers` | `object` | Карта `имя → конфиг сервера` (формат SDK, см. ниже). Имя: только `[a-zA-Z0-9_-]` — оно становится частью имён тулзов `mcp__<server>__<tool>`. |
| `mcp_tools` | `string[]` | Опциональный whitelist тулзов: `mcp__<server>__<tool>` или `mcp__<server>` (все тулзы сервера). По умолчанию разрешены **все** тулзы всех перечисленных серверов. |
| `enable_tools` | `bool` | Отдельный флаг для встроенных тулзов (Read, Glob, Grep, Bash, Write, Edit). Независим от MCP. |

Комбинации:

- `mcp_servers` **без** `enable_tools` → агент видит **только** MCP-тулзы.
- `mcp_servers` + `enable_tools: true` → MCP-тулзы + встроенные.
- ни того, ни другого → обычный OpenAI-совместимый режим без тулзов.

Агент работает итеративно (до `max_turns=10`): вызывает тулзы, читает
результаты, продолжает — это и есть агентский режим.

## Форматы конфига сервера

### Remote (http / sse) — рекомендуется для Docker

```json
{
  "notion": {
    "type": "http",
    "url": "https://mcp.notion.com/mcp",
    "headers": { "Authorization": "Bearer <token>" }
  }
}
```

### stdio (локальный процесс)

```json
{
  "fetch": {
    "command": "python",
    "args": ["-m", "mcp_server_fetch"],
    "env": { "MY_VAR": "value" }
  }
}
```

⚠️ `command` должен существовать **там, где запущен wrapper**. В Docker-образе
нет Node.js, поэтому `npx`-серверы внутри контейнера не запустятся —
используйте `http`/`sse`, python-серверы (установив пакет в образ) или
добавьте Node в Dockerfile.

## Примеры

### curl

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-6",
    "messages": [{"role": "user", "content": "Получи список задач из трекера"}],
    "mcp_servers": {
      "tracker": {"type": "http", "url": "https://mcp.example.com/mcp"}
    }
  }'
```

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": "Что в моём Notion-дашборде?"}],
    extra_body={
        "mcp_servers": {
            "notion": {
                "type": "http",
                "url": "https://mcp.notion.com/mcp",
                "headers": {"Authorization": "Bearer ..."},
            }
        },
        # необязательно: ограничить набор тулзов
        "mcp_tools": ["mcp__notion__search", "mcp__notion__fetch"],
        # необязательно: добавить встроенные Read/Bash/Edit и т.д.
        "enable_tools": True,
    },
)
print(response.choices[0].message.content)
```

### Сессия (многошаговый чат)

Отдельного endpoint'а «создать сессию» нет — сессия создаётся первым запросом
с `session_id`. Поле `mcp_servers` передаётся **с каждым запросом** (опции
агента собираются заново на каждый вызов), поэтому просто шлите его вместе с
тем же `session_id`:

```python
common = {
    "session_id": "support-chat-42",
    "mcp_servers": {"crm": {"type": "http", "url": "https://crm.example.com/mcp"}},
}

client.chat.completions.create(
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": "Найди клиента Иванова"}],
    extra_body=common,
)
client.chat.completions.create(
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": "А какие у него открытые тикеты?"}],
    extra_body=common,  # тот же session_id → контекст сохраняется
)
```

### LibreChat / OpenWebUI и другие фронтенды

Если фронтенд не умеет слать произвольные поля в теле запроса, варианты:

1. Прокси/middleware, добавляющий `mcp_servers` в тело.
2. Форк дефолтов: захардкодить набор серверов на стороне wrapper
   (env-переменная — пока не реализовано, можно добавить).

## Отладка

- Логи wrapper'а: при подключении видно
  `MCP servers attached: ['notion']` и `Tools enabled by user request: ['mcp__notion']`.
- `POST /v1/compatibility` — покажет, что `mcp_servers`/`mcp_tools` поддерживаются.
- Ошибки валидации конфига возвращаются как HTTP 422 с описанием поля.
- Если сервер подключился, но агент «не видит» тулзы — проверьте `mcp_tools`:
  имена должны начинаться с `mcp__` и совпадать с именем сервера.

## Ограничения

- Эндпоинты `/v1/mcp/*` (register/connect/stats) — это интроспекция со стороны
  wrapper-процесса, к агенту Claude они **не** подключают серверы. Используйте
  `mcp_servers` в чат-запросе.
- stdio-серверы запускаются от пользователя wrapper-процесса (в Docker —
  uid 1000) — права на файлы/сеть соответствующие.
- Время первого запроса растёт на время старта/handshake MCP-сервера.
