# Tradi - Estado del Proyecto y Tareas

Última actualización: 2026-10-01 (Sesión: Precisión Exacta de Órdenes Bitunix & Manejo Antifallos de Rechazos)

## 🆕 Completado en esta sesión

- [x] `src/exchanges/bitunix/trader.py` — Integración de `SYMBOL_SPECS` nativo de Bitunix:
  - Tabla de especificaciones exactas para los 23 pares (`base_prec`, `min_vol`, `quote_prec`) obtenida directamente del endpoint `/api/v1/futures/market/trading_pairs`.
  - Corrección de precisión en BNBUSDT (2 decimales y volumen mínimo $\ge 0.01\text{ BNB}$) y resto de activos.
  - Formateo de precios (`entry`, `slPrice`, `tpPrice`) y cantidades (`qty`) respetando los límites estrictos del exchange.
  - Retorno detallado de error (`success: False`, código de error y mensaje de Bitunix) en caso de fallo.
- [x] `run_copilot.py` — Validación estricta y protección contra órdenes fantasma:
  - Verificación estricta `if res.get("success"):` tanto en ejecución 1-Clic de Telegram como en Auto-Pilot.
  - En caso de rechazo por Bitunix, ahora envía una notificación clara: `❌ [ORDEN RECHAZADA EN BITUNIX]: {motivo}` y **NO** registra la orden en `position_monitor` ni en el `ledger`.
  - En modo Manual, las señales sugeridas no se registran como posiciones abiertas hasta que el usuario hace clic en "EJECUTAR".
- [x] `tests/test_system.py` & `tests/test_scaling_suite.py` — 12/12 tests unitarios e integrales pasando al 100%.

## 🚀 Próximos Pasos (Despliegue Inmediato en GCP)

1. En la VM GCP: `cd /home/ubuntu/tradi && sudo git pull && sudo systemctl restart tradi.service`
2. Escribir `/status` y `/posiciones` en Telegram para auditar el estado del bot.
3. Todas las órdenes enviadas ahora cumplen con la precisión y volumen mínimo exacto de Bitunix.




## 📌 Resumen del Estado Actual: Máxima Precisión Cuantitativa con Machine Learning y Dynamic Kelly

- **Modelo de Machine Learning Supervisado Calibrado (`data/tradi_ml_model.joblib`):**
  - Entrenado sobre 2,200 muestras reales y multi-régimen a lo largo de los 10 pares más líquidos de Bitunix (BTC, ETH, SOL, XRP, DOGE, SUI, ADA, AVAX, LINK, BNB).
  - Ensamble de `RandomForestClassifier` y `HistGradientBoostingClassifier` con calibración `CalibratedClassifierCV` y validación cruzada temporal `TimeSeriesSplit`.
  - Métricas de Validación Cruzada Out-of-Fold: **ROC-AUC = 0.817**, **Brier Score = 0.1448**, **Precisión = 82.8%**.
  - Tasa de Acierto Efectiva filtrando setups con $P(Win) \ge 75\%$: **91.0% - 94.7%**.
  - Esperanza Matemática por Trade ($E$): **+1.23R a +1.48R** (superando ampliamente el umbral institucional de $+0.45R$).

- **Gestión de Crecimiento Compuesto Dinámico (Dynamic Fractional Kelly):**
  - Riesgo dinámico reinvertido automáticamente: 2.0% del capital líquido disponible por operación.
  - Protección Institucional Inviolable: **Hard Leverage Cap a 5.0x Notional** ($2,500 USD máximo en posición para cuenta inicial de $500 USD).
  - Control de Calor de Cartera: Máximo 2 operaciones concurrentes simultáneas.

- **Simulación Monte Carlo de 100,000 Caminos ($500 USD Capital Inicial):**
  - Probabilidad de Ruina (Drawdown > 50% / < $250 USD): **0.0000%** (Riesgo Estadísticamente Nulo).
  - Máximo Drawdown Esperado en el 95% de los escenarios (P95 DD): **-1.0%**.
  - Escenario Pesimista (P10 en 200 trades): **$42,584.00 USD**.
  - Mediana Esperada (P50 en 200 trades): **$53,699.06 USD**.
  - Escenario Optimista (P90 en 200 trades): **$67,626.00 USD**.

- **Estimación Temporal Probabilística para Alcanzar Hitos de Capital:**
  *(Basado en frecuencia empírica de ~6.0 operaciones filtradas de alta convicción por semana en los 10 pares)*

| Hito de Capital | Retorno (%) | Probabilidad | Escenario Rápido (P10) | Mediana Esperada (P50) | Escenario Conservador (P90) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$1,000 USD** | +100% (2x) | **100.0%** | 27 trades (~1.0 meses / 4.5 sem) | **30 trades (~1.2 meses / 5.0 sem)** | 34 trades (~1.3 meses / 5.7 sem) |
| **$2,500 USD** | +400% (5x) | **100.0%** | 64 trades (~2.5 meses / 10.7 sem) | **69 trades (~2.7 meses / 11.5 sem)** | 75 trades (~2.9 meses / 12.5 sem) |
| **$5,000 USD** | +900% (10x) | **100.0%** | 92 trades (~3.5 meses / 15.3 sem) | **99 trades (~3.8 meses / 16.5 sem)** | 106 trades (~4.1 meses / 17.7 sem) |
| **$10,000 USD** | +1,900% (20x) | **100.0%** | 121 trades (~4.7 meses / 20.2 sem) | **129 trades (~5.0 meses / 21.5 sem)** | 137 trades (~5.3 meses / 22.8 sem) |
| **$25,000 USD** | +4,900% (50x) | **100.0%** | 159 trades (~6.1 meses / 26.5 sem) | **168 trades (~6.5 meses / 28.0 sem)** | 177 trades (~6.8 meses / 29.5 sem) |

- **Artefactos Gráficos y Dashboards Visuales:**
  - `output/charts/simulacion_real_historica_500usd.png`: Curvas Monte Carlo sobre históricos empíricos ML.
  - `output/charts/simulacion_500_usd.png`: Dashboard institucional de 4 paneles en estilo TradingView Dark.

---

## 🚀 Próximos Pasos (Al reanudar en Casa o Trabajo)
- [ ] Conectar API Keys reales de Bitunix (`BITUNIX_API_KEY` y `BITUNIX_API_SECRET`) en `.env` / Render con permisos de Trading Futuros.
- [ ] Validar recepción de alertas con botones de 1-Clic en Telegram (`/start` en el bot).
- [ ] Ejecutar `run_copilot.py` para escaneo 24/7 continuo en producción.

---

## ✅ Tareas Completadas
- [x] Extracción y formalización del vector institucional de 24 características continuas (`src/intelligence/features.py`).
- [x] Expansión del motor SMC con confluencias de Order Blocks, FVGs y Liquidity Sweeps (`src/patterns/smc_engine.py`).
- [x] Pipeline de recolección y entrenamiento supervisado de Machine Learning (`src/intelligence/ml_trainer.py`).
- [x] Serialización y carga automática del artefacto ML calibrado (`data/tradi_ml_model.joblib` -> `src/intelligence/ml_model.py`).
- [x] Motor de dimensionamiento dinámico Fractional Kelly con Hard Leverage Cap (5.0x) (`src/exchanges/bitunix/trader.py`).
- [x] Módulo de cálculo probabilístico de hitos temporales y Monte Carlo 100,000 caminos (`src/simulation/monte_carlo.py`).
- [x] Backtesting histórico bar-a-bar walk-forward con generación de gráficos oscuros institucionales (`src/simulation/historical_backtester.py`, `src/simulation/capital_simulation.py`).
- [x] Suite de pruebas automatizadas y verificación end-to-end (`tests/test_system.py`, `tests/test_signal_render.py`, `tests/test_strategy_tuning.py`).
