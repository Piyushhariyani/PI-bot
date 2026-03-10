# Chatbot Web App (HTML + Flask + Redis)

## Environment
Create/update `.env`:

```env
GOOGLE_API_KEY=your_google_api_key
REDIS_URL=redis://localhost:6379/0
CACHE_TTL_SECONDS=3600
MODEL_NAME=gemini-2.5-flash
RESPONSE_WORD_LIMIT=100
HOST=0.0.0.0
PORT=8000
```

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Open: `http://localhost:8000`

## Run with Docker + Redis

```bash
docker compose up --build
```

Open: `http://localhost:8000`

## Features
- Dedicated HTML chatbot frontend (`templates/chat.html`)
- Previous user chats shown in a side panel (stored in browser localStorage)
- Redis response caching with TTL
- Configurable word limit enforced on every response
