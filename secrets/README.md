# Docker Secrets

This directory contains secret files used by Docker Compose.

## Setup Instructions

1. Copy the example files and remove the `.example` extension:
   ```bash
   cp postgres_password.txt.example postgres_password.txt
   cp openai_api_key.txt.example openai_api_key.txt
   ```

2. Edit each file and replace the placeholder text with your actual secrets:
   - `postgres_password.txt`: Your PostgreSQL database password
   - `openai_api_key.txt`: Your OpenAI API key

3. **Important**: These files should contain ONLY the secret value with no trailing newlines or spaces.

## Security Notes

- The actual secret files (`*.txt`) are gitignored and will not be committed
- Keep these files secure and never commit them to version control
- Use strong, unique passwords for production environments
- Consider using Docker Swarm secrets or external secret management for production

## File Format

Each secret file should contain only the secret value:
```
your_secret_value
```

Do NOT include:
- Quotes around the value
- Trailing newlines (some editors add these automatically)
- Any other formatting

## Removing Trailing Newlines

If your secrets aren't working, they might have trailing newlines. Remove them:

### Linux/Mac:
```bash
# Remove trailing newline from postgres password
printf '%s' "$(cat postgres_password.txt)" > postgres_password.txt

# Remove trailing newline from OpenAI API key
printf '%s' "$(cat openai_api_key.txt)" > openai_api_key.txt
```

### Windows PowerShell:
```powershell
# Remove trailing newline from postgres password
[System.IO.File]::WriteAllText("postgres_password.txt", (Get-Content -Raw postgres_password.txt).TrimEnd())

# Remove trailing newline from OpenAI API key
[System.IO.File]::WriteAllText("openai_api_key.txt", (Get-Content -Raw openai_api_key.txt).TrimEnd())
```
