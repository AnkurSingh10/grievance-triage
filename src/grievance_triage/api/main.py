import uvicorn

from .app import app


def main():
    uvicorn.run("grievance_triage.api.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
