# Frontend For MTG Server

The server opens a websocket connection to the backend
and sends images (would be better as WebRTC...). The
server then runs detection and embedding and then searches
against qdrant to get results. Finally tracked objects are
sent back over websocket to the frontend and displayed.

1. run the server

```bash
cd mtg-vision
fastapi dev mtgvision/server.py --host 0.0.0.0
```

2. run the dev server

```bash
cd mtg-vision/www
pnpm start
```

3. run cloudflare tunnel to expose as https for mobile devices

```bash
cloudflared tunnel --url http://localhost:8000
```
