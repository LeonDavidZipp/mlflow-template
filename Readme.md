# MLFlow Template Repo
## Small template repo to get started with mlflow locally & immediately.

### Source
I followed [this](https://ruhyadi.github.io/blog/mlflow-docker/) tutorial by Didi Ruhyadi. Go check him out!

### Setup
1. Start the server(s)
   ```bash
   docker-compose up
   ```
2. Create bucket for artifact store: Go to `http://localhost:8900` using the default credentials from the docker compose minio service and create a bucket called `mlflow`.
3. Run the pipeline with either:
   ```bash
   uv run src/pipeline.py
   ```
   or if that stalls longer than expected:
   ```bash
   source .venv/bin/activate
   python src/pipeline.py
   ```
4. Enjoy mlflow at `http://localhost:5001`, find your model there & follow the training cycle & metrics!
