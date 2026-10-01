FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MPLCONFIGDIR=/tmp/matplotlib \
    DRSTAGE_CHECKPOINT=/app/model/best_fp16.pt \
    DRSTAGE_DEVICE=cpu \
    DRSTAGE_TTA=0 \
    DRSTAGE_SPLIT_CSV=/app/samples/samples.csv \
    DRSTAGE_FRONTEND_DIST=/app/frontend/dist

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.14.0 torchvision==0.29.0 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps grad-cam==1.5.7 ttach==0.0.3

COPY src ./src
COPY configs ./configs
COPY frontend/dist ./frontend/dist
COPY samples ./samples
COPY model ./model

RUN useradd --create-home --uid 1000 user && chown -R user /app
USER user

EXPOSE 7860
CMD ["uvicorn", "drstage.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "7860"]
