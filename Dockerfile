FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
RUN pip install --no-cache-dir numpy==2.2.6 scipy==1.15.3
WORKDIR /bot
COPY bot.py world.py ./
COPY data ./data
ENTRYPOINT ["python3", "-u", "/bot/bot.py"]
