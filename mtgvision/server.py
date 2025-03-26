from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles


app = FastAPI()

# Serve static files (HTML/JS) from a 'static' directory
app.mount(
    "/",
    StaticFiles(directory=Path(__file__).parent.parent / "www", html=False),
    name="static",
)


if __name__ == "__main__":
    print("run with: uvicorn server:app")
