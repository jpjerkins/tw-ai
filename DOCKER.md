# Docker Deployment Guide

This guide explains how to deploy the TiddlyWiki RAG system using Docker Compose.

## Architecture

The Docker Compose setup includes:
- **PostgreSQL Database**: Custom image with Apache AGE and pgvector extensions
- **Python Application** (optional): Containerized Python app for running the TiddlyWiki API

## Prerequisites

- Docker Engine 20.10+
- Docker Compose V2
- Your TiddlyWiki instance accessible from the Docker network

## Quick Start

### 1. Setup Secrets

Create the secret files from templates:

```bash
# On Linux/Mac
cp secrets/postgres_password.txt.example secrets/postgres_password.txt
cp secrets/openai_api_key.txt.example secrets/openai_api_key.txt

# Edit the files with your actual secrets
nano secrets/postgres_password.txt
nano secrets/openai_api_key.txt
```

```powershell
# On Windows PowerShell
Copy-Item secrets\postgres_password.txt.example secrets\postgres_password.txt
Copy-Item secrets\openai_api_key.txt.example secrets\openai_api_key.txt

# Edit the files with your actual secrets
notepad secrets\postgres_password.txt
notepad secrets\openai_api_key.txt
```

**Important**: Ensure files contain only the secret value with no trailing newlines.

### 2. Configure Environment

Copy and edit the environment file:

```bash
cp .env.example .env
```

Edit `.env` to match your setup:
```env
POSTGRES_DB=tiddlywiki_rag
POSTGRES_USER=twrag
POSTGRES_PORT=5432
```

### 3. Start the Services

```bash
# Build and start the database
docker compose up -d

# View logs
docker compose logs -f postgres

# Check status
docker compose ps
```

### 4. Verify Database

Check that the extensions are loaded:

```bash
docker compose exec postgres psql -U twrag -d tiddlywiki_rag -c "SELECT * FROM pg_extension;"
```

You should see `vector` and `age` extensions listed.

## Usage

### Fetch and Store Tiddlers

From your host machine (with Python environment):

```bash
# Activate your virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Run the script
python tiddlywiki_api.py localhost:8080
```

### Search for Tiddlers

```bash
python tiddlywiki_api.py search "your search query" 5
```

### Using the Containerized App (Optional)

To run the Python application in a container:

1. Uncomment the `app` service in `docker-compose.yml`

2. Start both services:
```bash
docker compose up -d
```

3. Execute commands:
```bash
# Fetch tiddlers
docker compose exec app python tiddlywiki_api.py <tiddlywiki-host>:8080

# Search
docker compose exec app python tiddlywiki_api.py search "query" 5
```

## Docker Compose Commands

### Basic Operations

```bash
# Start services
docker compose up -d

# Stop services
docker compose down

# View logs
docker compose logs -f [service-name]

# Restart a service
docker compose restart [service-name]

# Rebuild and restart
docker compose up -d --build
```

### Database Management

```bash
# Access PostgreSQL CLI
docker compose exec postgres psql -U twrag -d tiddlywiki_rag

# Backup database
docker compose exec postgres pg_dump -U twrag tiddlywiki_rag > backup.sql

# Restore database
docker compose exec -T postgres psql -U twrag -d tiddlywiki_rag < backup.sql

# View database size
docker compose exec postgres psql -U twrag -d tiddlywiki_rag -c "\l+"
```

### Maintenance

```bash
# Remove all containers and volumes (DESTRUCTIVE)
docker compose down -v

# Remove only containers (preserves data)
docker compose down

# View resource usage
docker compose stats

# Clean up unused resources
docker system prune -a
```

## Volumes

Data is persisted in named volumes:
- `postgres_data`: PostgreSQL database files

To backup the volume:
```bash
docker run --rm -v tiddlywiki_postgres_data:/data -v $(pwd):/backup ubuntu tar czf /backup/postgres_backup.tar.gz -C /data .
```

To restore the volume:
```bash
docker run --rm -v tiddlywiki_postgres_data:/data -v $(pwd):/backup ubuntu tar xzf /backup/postgres_backup.tar.gz -C /data
```

## Networking

The services communicate over a custom bridge network `tiddlywiki_network`.

To connect your TiddlyWiki instance:
1. Add it to the same network in docker-compose.yml
2. Or use `host.docker.internal` to reference the host machine

## Security Best Practices

1. **Secrets Management**
   - Never commit `secrets/*.txt` files to version control
   - Use strong, unique passwords
   - Rotate secrets regularly

2. **Network Security**
   - Don't expose PostgreSQL port to the internet
   - Use `127.0.0.1:5432:5432` instead of `5432:5432` if only local access needed

3. **Production Deployment**
   - Use Docker Swarm secrets or Kubernetes secrets
   - Enable PostgreSQL SSL/TLS
   - Implement proper backup strategies
   - Use container security scanning

## Troubleshooting

### Database Connection Issues

```bash
# Check if database is ready
docker compose exec postgres pg_isready -U twrag

# Check database logs
docker compose logs postgres

# Test connection
docker compose exec postgres psql -U twrag -d tiddlywiki_rag -c "SELECT 1;"
```

### Extension Problems

```bash
# Verify extensions are installed
docker compose exec postgres psql -U twrag -d tiddlywiki_rag -c "\dx"

# Manually enable extensions (if needed)
docker compose exec postgres psql -U twrag -d tiddlywiki_rag -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Secret File Issues

If authentication fails, check for trailing newlines:

```bash
# Linux/Mac
hexdump -C secrets/postgres_password.txt

# Windows PowerShell
Format-Hex secrets\postgres_password.txt
```

Remove trailing newlines (see `secrets/README.md` for instructions).

### Container Won't Start

```bash
# View detailed logs
docker compose logs --tail=100 postgres

# Check container status
docker compose ps -a

# Inspect container
docker inspect tiddlywiki_postgres
```

## Environment Variables

### PostgreSQL Container

- `POSTGRES_DB`: Database name
- `POSTGRES_USER`: Database user
- `POSTGRES_PASSWORD_FILE`: Path to password secret file (set by Docker)

### Application Container (optional)

- `POSTGRES_HOST`: Database hostname (use service name: `postgres`)
- `POSTGRES_PORT`: Database port (default: 5432)
- `POSTGRES_DB`: Database name
- `POSTGRES_USER`: Database user
- `POSTGRES_PASSWORD_FILE`: Path to password secret file
- `OPENAI_API_KEY_FILE`: Path to OpenAI API key secret file

## Customization

### Change Database Port

Edit `docker-compose.yml`:
```yaml
ports:
  - "5433:5432"  # External:Internal
```

Update `.env`:
```env
POSTGRES_PORT=5433
```

### Add Custom Init Scripts

Place SQL files in `init/` directory. They will be executed in alphabetical order when the database is first created.

### Resource Limits

Add resource constraints in `docker-compose.yml`:
```yaml
services:
  postgres:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

## Monitoring

### Health Checks

View service health:
```bash
docker compose ps
```

### Resource Usage

```bash
# Real-time stats
docker compose stats

# Database size
docker compose exec postgres psql -U twrag -d tiddlywiki_rag -c "SELECT pg_size_pretty(pg_database_size('tiddlywiki_rag'));"

# Table sizes
docker compose exec postgres psql -U twrag -d tiddlywiki_rag -c "SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) FROM pg_tables WHERE schemaname = 'public' ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;"
```

## Further Reading

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [PostgreSQL Docker Image](https://hub.docker.com/_/postgres)
- [Docker Secrets](https://docs.docker.com/engine/swarm/secrets/)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
