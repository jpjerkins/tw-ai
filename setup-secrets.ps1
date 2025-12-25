# Setup script for Docker secrets (Windows PowerShell)

Write-Host "========================================"
Write-Host "TiddlyWiki RAG - Docker Secrets Setup"
Write-Host "========================================"
Write-Host ""

# Check if secrets directory exists
if (-not (Test-Path "secrets")) {
    Write-Host "Creating secrets directory..."
    New-Item -ItemType Directory -Path "secrets" | Out-Null
}

# Setup PostgreSQL password
if (Test-Path "secrets\postgres_password.txt") {
    Write-Host "✓ PostgreSQL password file already exists"
    $overwrite = Read-Host "Do you want to overwrite it? (y/N)"
    if ($overwrite -eq "y" -or $overwrite -eq "Y") {
        $postgresPass = Read-Host "Enter PostgreSQL password" -AsSecureString
        $plainPass = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($postgresPass))
        [System.IO.File]::WriteAllText("secrets\postgres_password.txt", $plainPass)
        Write-Host "✓ PostgreSQL password updated"
    }
} else {
    $postgresPass = Read-Host "Enter PostgreSQL password" -AsSecureString
    $plainPass = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($postgresPass))
    [System.IO.File]::WriteAllText("secrets\postgres_password.txt", $plainPass)
    Write-Host "✓ PostgreSQL password saved"
}

# Setup OpenAI API key
if (Test-Path "secrets\openai_api_key.txt") {
    Write-Host "✓ OpenAI API key file already exists"
    $overwrite = Read-Host "Do you want to overwrite it? (y/N)"
    if ($overwrite -eq "y" -or $overwrite -eq "Y") {
        $openaiKey = Read-Host "Enter OpenAI API key" -AsSecureString
        $plainKey = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($openaiKey))
        [System.IO.File]::WriteAllText("secrets\openai_api_key.txt", $plainKey)
        Write-Host "✓ OpenAI API key updated"
    }
} else {
    $openaiKey = Read-Host "Enter OpenAI API key" -AsSecureString
    $plainKey = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($openaiKey))
    [System.IO.File]::WriteAllText("secrets\openai_api_key.txt", $plainKey)
    Write-Host "✓ OpenAI API key saved"
}

Write-Host ""
Write-Host "========================================"
Write-Host "Setup complete!"
Write-Host "========================================"
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Review .env file and adjust settings if needed"
Write-Host "2. Start the services: docker compose up -d"
Write-Host "3. Check logs: docker compose logs -f"
Write-Host ""
Write-Host "For more information, see DOCKER.md"
