# Tradi - Estado del Proyecto y Tareas

Última actualización: 2026-09-15

## 📌 Resumen del Estado Actual
- **Cerebro Inteligente Multi-Timeframe (Machine Learning):** Motor adaptativo cuantitativo que extrae un vector continuo de 24+ características institucionales (Volatilidad normalizada ATR, Volatilidad de Parkinson, RVOL 20, Volume Delta por barra, Pendiente OBV 14, Distancia VWAP rodante, ADX/DI delta, RSI slope, Bollinger Bandwidth, Z-Score, MACD normalizado, Geometría de velas y Sesgo Macro 4H). Evalúa la probabilidad matemática ($P \ge 75\%$) con filtro de Meta-Labeling antes de emitir cualquier señal.
- **Escaneo Multi-Timeframe (4H + 15m):** Descarga simultánea de 4H para dirección y estructura macro de mercado junto con 15m para Puntos de Interés (POI), descartando estrictamente señales que contradigan la tendencia 4H (prohibido contra-tendencia).
- **Gestión de Riesgo Institucional y Calor de Cartera:**
  - Límite estricto de apalancamiento: Máximo 5.0x Notional Exposure ($2,500 USD máximo sobre $500 USD de capital).
  - Control de calor de cartera (Portfolio Heat): Máximo 2 operaciones concurrentes simultáneas (4% de riesgo total = $20 USD máx).
- **Simulador Cuantitativo de Backtesting y Monte Carlo Bootstrap (100,000 Caminos):** Motor de simulación barra por barra con deducción exacta de comisiones (Maker 0.02%, Taker 0.06%) y slippage (0.02%) en los 10 pares líquidos de Bitunix, generando dashboards visuales de 4 paneles en `output/charts/simulacion_500_usd.png`.
- **Ejecución 1-Clic en Telegram:** Señales institucionales enriquecidas con botones interactivos para ejecutar la orden en Bitunix con $10 USD de riesgo (2%) o descartarla.
- **Trader Bitunix (Firma Doble SHA-256):** Módulo de órdenes de futuros con SL estructural inviolable y gestión dinámica de posiciones. Cuenta con Modo Simulación activo hasta configurar las API keys.
- **Despliegue 24/7 en la Nube:** Operando en Render (`https://tradi-wcic.onrender.com`) y monitoreado sin pausas por UptimeRobot.

## 🚀 Próximos Pasos (Al llegar a Casa)
- [ ] Entrar a Bitunix desde la PC de casa, generar la API Key y Secret Key (Permisos: Read + Futures Trading; SIN retiros).
- [ ] Agregar `BITUNIX_API_KEY` y `BITUNIX_API_SECRET` en las variables de entorno de Render para activar la ejecución real de órdenes (o dejarlas vacías para seguir en modo simulación virtual).
- [ ] En la laptop de casa, clonar el repositorio con `git pull` o `git clone git@github.com:Amziisrae23/tradi.git` y probar el comando *"Continuemos"*.

## ✅ Tareas Completadas
- [x] Configuración inicial de Git, SSH y sincronización entre computadoras.
- [x] Conector en vivo con Bitunix Futures (REST y WebSockets).
- [x] Motor de Smart Money Concepts (Order Blocks, FVGs, Sweeps, Swings).
- [x] Vector continuo de 24+ características institucionales (`src/intelligence/features.py`).
- [x] Cerebro adaptativo de Machine Learning con Meta-Labeling (`src/intelligence/ml_model.py`).
- [x] Escaneo Multi-Timeframe (4H + 15m) en `src/patterns/smc_engine.py` y `run_copilot.py`.
- [x] Límite institucional de apalancamiento (Max 5.0x Notional) y control de calor de cartera (`src/exchanges/bitunix/trader.py`).
- [x] Motor de Backtesting Histórico cuantitativo con comisiones reales (`src/simulation/historical_backtest.py`).
- [x] Simulador de Monte Carlo Bootstrap de 100,000 iteraciones con dashboard de 4 paneles (`src/simulation/capital_simulation.py` & `src/simulation/monte_carlo.py`).
- [x] Botones interactivos de 1-Clic en Telegram (`src/copilot/telegram_notifier.py`).
- [x] Despliegue en la nube en Render con UptimeRobot 24/7.
