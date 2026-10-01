FROM python:3.12-slim

RUN useradd -m -u 1000 user && mkdir /app && chown user:user /app
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /app

COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user main.py train.csv ./

# Train the model during the build, so it always matches the installed scikit-learn
RUN python -c "import main; main.train_and_save()"

EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
