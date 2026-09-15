from typing import Dict, Any, Optional
from src.patterns.types import TradeSetup
from src.simulation.monte_carlo import run_monte_carlo_simulation

class SignalGenerator:
    """Genera señales enriquecidas con justificación técnica y métricas de riesgo."""

    @staticmethod
    def format_signal_text(setup: TradeSetup, mc_metrics: Optional[Dict[str, Any]] = None) -> str:
        """
        Formatea el mensaje de la señal con estructura institucional lista para Telegram/Discord/Terminal.
        """
        direction_emoji = "🟢" if setup.direction == "LONG" else "🔴"
        action_text = "LONG (COMPRA)" if setup.direction == "LONG" else "SHORT (VENTA)"

        reasons_formatted = "\n".join([f"  • {r}" for r in setup.reasons]) if setup.reasons else "  • Confluencia técnica institucional en zona de liquidez."

        mc_info = ""
        if mc_metrics:
            mc_info = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎲 𝐒𝐈𝐌𝐔𝐋𝐀𝐂𝐈Ó𝐍 𝐌𝐎𝐍𝐓𝐄 𝐂𝐀𝐑𝐋𝐎 ({mc_metrics.get('n_simulations', 25000):,} Caminos):
  🛡️ Probabilidad de Ruina: {mc_metrics.get('prob_of_ruin', 0.0) * 100:.2f}%
  📉 Max Drawdown Esperado (P95): {mc_metrics.get('max_drawdown_p95', 0.0) * 100:.2f}%
  💼 Tamaño de Posición Sugerido (Half-Kelly): {mc_metrics.get('recommended_position_pct', 1.5)}% del capital"""

        msg = f"""🚨 𝗧𝗥𝗔𝗗𝗜 𝗖𝗢𝗣𝗜𝗟𝗢𝗧 | 𝗦𝗘Ñ𝗔𝗟 𝗘𝗡 𝗧𝗜𝗘𝗠𝗣𝗢 𝗥𝗘𝗔𝗟 🚨

⚡ 𝐏𝐀𝐑: {setup.symbol} (Bitunix Futures)
{direction_emoji} 𝐃𝐈𝐑𝐄𝐂𝐂𝐈Ó𝐍: {action_text}
📍 𝐏𝐑𝐄𝐂𝐈𝐎 𝐃𝐄 𝐄𝐍𝐓𝐑𝐀𝐃𝐀: ${setup.entry_price:,.2f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 𝐎𝐁𝐉𝐄𝐓𝐈𝐕𝐎𝐒 𝐃𝐄 𝐆𝐀𝐍𝐀𝐍𝐂𝐈𝐀 (𝐓𝐀𝐊𝐄 𝐏𝐑𝐎𝐅𝐈𝐓):
  🥇 TP 1: ${setup.tp1:,.2f} (R:R 1:{setup.rr_tp1}) ➔ [Cerrar 40% + Mover SL a Breakeven]
  🥈 TP 2: ${setup.tp2:,.2f} (R:R 1:{setup.rr_tp2}) ➔ [Cerrar 40% Target Principal]
  🚀 TP 3: ${setup.tp3:,.2f} (R:R 1:{setup.rr_tp3}) ➔ [Runner 20% / Trailing Stop]

🛑 𝐒𝐓𝐎𝐏 𝐋𝐎𝐒𝐒 (𝐈𝐍𝐕𝐀𝐋𝐈𝐃𝐀𝐂𝐈Ó𝐍):
  ❌ SL: ${setup.stop_loss:,.2f} (Distancia de riesgo: ${setup.risk_distance:,.2f})

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 𝐌É𝐓𝐑𝐈𝐂𝐀𝐒 & 𝐆𝐄𝐒𝐓𝐈Ó𝐍 𝐃𝐄 𝐑𝐈𝐄𝐒𝐆𝐎:
  ⚖️ Ratio Riesgo/Beneficio (R:R): 1 : {setup.rr_tp2}
  🤖 Confianza del Modelo: {setup.confidence_score:.1f}%
  🛡️ Riesgo recomendado: 1.0% - 2.0% de tu cuenta
{mc_info}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🧠 𝐉𝐔𝐒𝐓𝐈𝐅𝐈𝐂𝐀𝐂𝐈Ó𝐍 𝐓É𝐂𝐍𝐈𝐂𝐀 (𝗥𝗘𝗔𝗦𝗢𝗡𝗜𝗡𝗚):
{reasons_formatted}

🖼️ Gráfico renderizado: {setup.chart_path or 'No disponible'}
"""
        return msg
