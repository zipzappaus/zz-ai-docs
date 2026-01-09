from fastapi import FastAPI
from app.core.config import settings
from app.api.routes import router

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION
)

app.include_router(router, prefix="/api/v1")

@app.get("/")
def root():
    return {"message": "Welcome to Flexible Document Search API", "docs": "/docs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
