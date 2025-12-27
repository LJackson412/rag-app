# Azure OpenAI Service – Chat Completions API  
**API Version:** `2024-02-01`

Diese README beschreibt die **Azure OpenAI Chat Completions API** inklusive Request-/Response-Strukturen, Parametern und Fehlerbehandlung.  
Sie ist als kompakte, entwicklerfreundliche Referenz gedacht.

---

## Inhaltsverzeichnis

- [Überblick](#überblick)
- [Endpoint](#endpoint)
- [Request](#request)
  - [Pfadparameter](#pfadparameter)
  - [Request Body](#request-body)
  - [Wichtige Parameter](#wichtige-parameter)
  - [Beispiel Request](#beispiel-request)
- [Response](#response)
  - [Erfolgreiche Response (200)](#erfolgreiche-response-200)
  - [Fehler-Responses](#fehler-responses)
- [Content Filtering](#content-filtering)
- [Typdefinitionen (Auszug)](#typdefinitionen-auszug)

---

## Überblick

Die **Chat Completions API** erzeugt Antworten auf Chat-Nachrichten (ähnlich zu ChatGPT).  
Sie unterstützt u. a.:

- Konversationsverläufe (`messages`)
- Streaming (`stream`)
- Funktions- & Tool-Calling
- JSON-Mode
- Content-Filtering
- Deterministische Antworten über `seed`

---

## Endpoint

```http
POST https://genaihub-gateway.genai-prod.comp.db.de/openai/deployments/{deployment-id}/chat/completions?api-version={api-version}
````

---

## Request

### Pfadparameter

| Name            | Typ    | Pflicht | Beschreibung                    |
| --------------- | ------ | ------- | ------------------------------- |
| `deployment-id` | string | ✅       | ID des Azure OpenAI Deployments |
| `api-version`   | string | ✅       | z. B. `2024-02-01`              |

---

### Request Body

`Content-Type: application/json`

#### Pflichtfelder

| Feld       | Typ                              | Beschreibung         |
| ---------- | -------------------------------- | -------------------- |
| `messages` | `chatCompletionRequestMessage[]` | Konversationsverlauf |

---

### Wichtige Parameter

| Parameter               | Typ                      | Beschreibung                                    |
| ----------------------- | ------------------------ | ----------------------------------------------- |
| `temperature`           | number (0–2)             | Kreativität der Antwort                         |
| `top_p`                 | number                   | Nucleus Sampling (Alternative zu `temperature`) |
| `max_completion_tokens` | integer                  | Maximale Anzahl generierter Tokens              |
| `stream`                | boolean                  | Streaming-Antworten (SSE)                       |
| `stop`                  | string | string[]        | Stop-Sequenzen (max. 4)                         |
| `presence_penalty`      | number (-2–2)            | Fördert neue Themen                             |
| `frequency_penalty`     | number (-2–2)            | Reduziert Wiederholungen                        |
| `logit_bias`            | object                   | Beeinflussung einzelner Tokens                  |
| `seed`                  | integer                  | Bestmögliche Deterministik                      |
| `user`                  | string                   | Endnutzer-ID (Abuse Detection)                  |
| `response_format.type`  | `text` | `json_object`   | JSON Mode                                       |
| `tools`                 | `chatCompletionTool[]`   | Tool-/Function-Calling                          |
| `tool_choice`           | `none` | `auto` | Objekt | Steuerung der Tool-Auswahl                      |

> 💡 **Hinweis:** `max_tokens` ist **deprecated** – bitte `max_completion_tokens` verwenden.

---

### Beispiel Request

```json
{
  "messages": [
    {
      "role": "user",
      "content": "say this is a test"
    }
  ],
  "temperature": 0.7
}
```

---

## Response

### Erfolgreiche Response (200)

```json
{
  "id": "string",
  "object": "chat.completion",
  "created": 0,
  "model": "string",
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  },
  "system_fingerprint": "string",
  "choices": [
    {
      "index": 0,
      "finish_reason": "string",
      "message": {
        "role": "assistant",
        "content": "string"
      }
    }
  ]
}
```

#### Response Header

| Name              | Beschreibung                   |
| ----------------- | ------------------------------ |
| `apim-request-id` | Request-ID für Troubleshooting |

---

### Fehler-Responses

#### 400 – Bad Request

#### 500 – Internal Server Error

```json
{
  "error": {
    "code": "string",
    "message": "string",
    "param": "string",
    "type": "string",
    "inner_error": {
      "code": "ResponsibleAIPolicyViolation",
      "content_filter_results": {
        "sexual": { "filtered": true, "severity": "safe" },
        "violence": { "filtered": true, "severity": "safe" },
        "hate": { "filtered": true, "severity": "safe" },
        "self_harm": { "filtered": true, "severity": "safe" }
      }
    }
  }
}
```

---

## Content Filtering

Die API liefert detaillierte **Content-Filter-Ergebnisse** für:

* `sexual`
* `violence`
* `hate`
* `self_harm`
* `profanity`
* `jailbreak`
* `protected_material_text`
* `protected_material_code`

Jede Kategorie enthält Informationen zu:

* `filtered` (boolean)
* `detected` (boolean)
* `severity` (`very_low` → `high`)

---

## Typdefinitionen (Auszug)

### `chatCompletionRequestMessage`

| Feld      | Typ                                               | Beschreibung     |
| --------- | ------------------------------------------------- | ---------------- |
| `role`    | `system \| user \| assistant \| tool \| function` | Rolle des Autors |
| `content` | string                                            | Textinhalt       |

---

### `chatCompletionTool`

| Feld                  | Typ         | Beschreibung       |
| --------------------- | ----------- | ------------------ |
| `type`                | `function`  | Tool-Typ           |
| `function.name`       | string      | Funktionsname      |
| `function.parameters` | JSON Schema | Funktionsparameter |

---

### `completionUsage`

| Feld                | Typ     | Beschreibung          |
| ------------------- | ------- | --------------------- |
| `prompt_tokens`     | integer | Tokens im Prompt      |
| `completion_tokens` | integer | Tokens in der Antwort |
| `total_tokens`      | integer | Gesamtverbrauch       |

---

## Hinweise

* Determinismus ist **nicht garantiert**, selbst bei identischem `seed`
* Streaming-Antworten enden mit:

  ```text
  data: [DONE]
  ```

---

📌 **Status:** Aktuell & produktionsreif
📅 **API-Version:** `2024-02-01`

```
```
