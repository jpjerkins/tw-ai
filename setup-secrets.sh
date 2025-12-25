#!/bin/bash
# Setup script for Docker secrets

set -e

echo "========================================"
echo "TiddlyWiki RAG - Docker Secrets Setup"
echo "========================================"
echo ""

# Check if secrets directory exists
if [ ! -d "secrets" ]; then
    echo "Creating secrets directory..."
    mkdir -p secrets
fi

# Setup PostgreSQL password
if [ -f "secrets/postgres_password.txt" ]; then
    echo "✓ PostgreSQL password file already exists"
    read -p "Do you want to overwrite it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -sp "Enter PostgreSQL password: " POSTGRES_PASS
        echo
        printf '%s' "$POSTGRES_PASS" > secrets/postgres_password.txt
        echo "✓ PostgreSQL password updated"
    fi
else
    read -sp "Enter PostgreSQL password: " POSTGRES_PASS
    echo
    printf '%s' "$POSTGRES_PASS" > secrets/postgres_password.txt
    echo "✓ PostgreSQL password saved"
fi

# Setup OpenAI API key
if [ -f "secrets/openai_api_key.txt" ]; then
    echo "✓ OpenAI API key file already exists"
    read -p "Do you want to overwrite it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -sp "Enter OpenAI API key: " OPENAI_KEY
        echo
        printf '%s' "$OPENAI_KEY" > secrets/openai_api_key.txt
        echo "✓ OpenAI API key updated"
    fi
else
    read -sp "Enter OpenAI API key: " OPENAI_KEY
    echo
    printf '%s' "$OPENAI_KEY" > secrets/openai_api_key.txt
    echo "✓ OpenAI API key saved"
fi

# Set proper permissions
chmod 600 secrets/*.txt 2>/dev/null || true

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Review .env file and adjust settings if needed"
echo "2. Start the services: docker compose up -d"
echo "3. Check logs: docker compose logs -f"
echo ""
echo "For more information, see DOCKER.md"
