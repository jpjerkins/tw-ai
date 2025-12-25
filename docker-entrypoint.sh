#!/bin/bash
set -e

# Read secrets from Docker secret files and export as environment variables
if [ -f "$POSTGRES_PASSWORD_FILE" ]; then
    export POSTGRES_PASSWORD=$(cat "$POSTGRES_PASSWORD_FILE")
fi

if [ -f "$OPENAI_API_KEY_FILE" ]; then
    export OPENAI_API_KEY=$(cat "$OPENAI_API_KEY_FILE")
fi

# Execute the main command
exec "$@"
