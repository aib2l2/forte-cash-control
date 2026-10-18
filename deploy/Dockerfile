FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false STORAGE_DIR=/var/lib/forte
WORKDIR /app
COPY deploy/requirements-runtime.txt /tmp/requirements-runtime.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 torchvision==0.29.1 \
    && pip install -r /tmp/requirements-runtime.txt \
    && useradd --uid 10001 --create-home forte \
    && mkdir -p /var/lib/forte && chown forte:forte /var/lib/forte
COPY backend/ backend/
COPY cash_ml/ cash_ml/
COPY models/cash_r3d/r3d18_kinetics400.pth models/cash_r3d/r3d18_kinetics400.pth
COPY models/cash_r3d/typed_classifier.npz models/cash_r3d/typed_classifier.npz
COPY models/cash_r3d/typed_report.json models/cash_r3d/typed_report.json
COPY deploy/third_party/ deploy/third_party/
USER forte
EXPOSE 8000 8501
CMD ["python", "-m", "uvicorn", "cash_ml.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
