import sys
import io
import os
import time
import uuid
import hashlib
import json
import requests

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.settings import settings
from rich.console import Console

console = Console(force_terminal=True)

def test_bitunix_auth():
    api_key = settings.BITUNIX_API_KEY
    api_secret = settings.BITUNIX_API_SECRET
    base_url = settings.BITUNIX_REST_URL

    if not api_key or not api_secret:
        console.print("[bold red]❌ Error:[/bold red] API Key o Secret Key no encontradas en .env")
        return

    console.print(f"[cyan]Probando autenticación con Bitunix API (Key: {api_key[:8]}...)...[/cyan]")

    url = f"{base_url}/api/v1/futures/account"
    nonce = uuid.uuid4().hex
    timestamp = str(int(time.time() * 1000))
    query_str = ""

    # Firma Doble SHA-256
    raw_string = f"{nonce}{timestamp}{api_key}{query_str}"
    first_digest = hashlib.sha256(raw_string.encode('utf-8')).hexdigest()
    final_sign = hashlib.sha256((first_digest + api_secret).encode('utf-8')).hexdigest()

    headers = {
        "api-key": api_key,
        "nonce": nonce,
        "timestamp": timestamp,
        "sign": final_sign,
        "Content-Type": "application/json"
    }

    try:
        res = requests.get(url, headers=headers, timeout=10)
        console.print(f"Respuesta HTTP Status: {res.status_code}")
        console.print(f"Contenido: {res.text}")
        if res.status_code == 200:
            data = res.json()
            if data.get("code") == 0:
                console.print("[bold green]✔ Autenticación con Bitunix EXITOSA al 100%.[/bold green]")
                balance_data = data.get("data", {})
                console.print(f"[green]Datos de cuenta:[/green] {balance_data}")
            else:
                console.print(f"[yellow]Bitunix respondió:[/yellow] {data}")
    except Exception as e:
        console.print(f"[bold red]Excepción al conectar con Bitunix:[/bold red] {e}")

if __name__ == "__main__":
    test_bitunix_auth()
