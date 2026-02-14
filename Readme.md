# MLFlow Template Repo
## Small template repo to get started with mlflow locally & immediately.

### Source
I followed [this](https://ruhyadi.github.io/blog/mlflow-docker/) tutorial by Didi Ruhyadi. Go check him out!

### Setup
1. Start the server(s)
   ```bash
   docker-compose up
   ```
2. Run the pipeline with either:
   ```bash
   uv run src/pipeline.py
   ```
   or if that stalls longer than expected:
   ```bash
   source .venv/bin/activate
   python src/pipeline.py
   ```
3. Enjoy mlflow at `http://localhost:5001`, find your model there & follow the training cycle & metrics!

### Configuration

All credentials and settings can be overridden via environment variables or a `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `MINIO_ROOT_USER` | mlflow | MinIO access key |
| `MINIO_ROOT_PASSWORD` | mlflow123 | MinIO secret key |
| `POSTGRES_USER` | mlflow | PostgreSQL username |
| `POSTGRES_PASSWORD` | mlflow123 | PostgreSQL password |
| `POSTGRES_DB` | mlflow | PostgreSQL database name |
| `MLFLOW_PORT` | 5001 | MLflow UI port |
| `MINIO_PORT` | 9000 | MinIO API port |
| `MINIO_CONSOLE_PORT` | 8900 | MinIO console port |
| `POSTGRES_PORT` | 5432 | PostgreSQL port |
