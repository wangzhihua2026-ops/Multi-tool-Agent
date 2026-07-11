FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOME=/app

WORKDIR ${APP_HOME}

RUN addgroup --system app && adduser --system --ingroup app app

COPY pyproject.toml README.md ./
COPY app ./app
COPY config ./config
COPY alembic.ini ./
COPY migrations ./migrations
COPY tests ./tests

ARG INSTALL_EXTRAS=""
RUN python -m pip install --upgrade pip \
    && if [ -n "$INSTALL_EXTRAS" ]; then \
        python -m pip install -e ".[${INSTALL_EXTRAS}]"; \
    else \
        python -m pip install -e .; \
    fi

RUN mkdir -p data && chown -R app:app ${APP_HOME}

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3).read()"

CMD ["uvicorn", "app.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
