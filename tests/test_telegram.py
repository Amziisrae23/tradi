import sys
import io
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if sys.platform.startswith("win"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.copilot.telegram_notifier import TelegramNotifier
from rich.console import Console

console = Console(force_terminal=True)

def test_telegram_connection():
    notifier = TelegramNotifier()
    if not notifier.is_configured():
        console.print("[bold red]❌ Error:[/bold red] TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID no están configurados en .env.")
        console.print("[yellow]Por favor, edita tu archivo .env y añade tus credenciales.[/yellow]")
        return

    console.print("[cyan]Enviando mensaje de prueba a Telegram...[/cyan]")
    success = notifier.send_message("🚀 [TRADI COPILOT]: ¡Conexión con Telegram establecida exitosamente!\n\nListo para recibir señales en tiempo real.")
    
    if success:
        console.print("[bold green]✔ Mensaje de prueba enviado con éxito. Revisa tu Telegram.[/bold green]")
        
        # Probar envío de gráfico si existe alguno en output/charts
        charts_dir = os.path.join(os.getcwd(), "output", "charts")
        if os.path.exists(charts_dir):
            files = [f for f in os.listdir(charts_dir) if f.endswith(".png")]
            if files:
                sample_chart = os.path.join(charts_dir, files[-1])
                console.print(f"[cyan]Enviando gráfico de prueba: {files[-1]}...[/cyan]")
                notifier.send_photo(sample_chart, caption="📊 Gráfico de prueba TradingView renderizado por Tradi Copilot.")
    else:
        console.print("[bold red]❌ No se pudo enviar el mensaje a Telegram. Verifica que el Token y Chat ID sean correctos.[/bold red]")

if __name__ == "__main__":
    test_telegram_connection()
