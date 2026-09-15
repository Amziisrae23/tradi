import os
from typing import Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from src.patterns.types import TradeSetup
from config.settings import settings

class ChartRenderer:
    """Renderiza gráficos con estilo TradingView Dark con zonas de TP/SL y Entry."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or settings.CHARTS_DIR
        os.makedirs(self.output_dir, exist_ok=True)

    def render_trade_setup(self, df: pd.DataFrame, setup: TradeSetup, lookback: int = 50) -> str:
        """
        Genera y guarda una imagen PNG del setup con estilo profesional dark mode.
        Retorna la ruta absoluta del archivo generado.
        """
        plot_df = df.tail(lookback).copy().reset_index(drop=True)
        n = len(plot_df)

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
        fig.patch.set_facecolor('#131722')
        ax.set_facecolor('#131722')

        # Dibujar Velas
        width = 0.6

        for i, row in plot_df.iterrows():
            o, h, l, c = row['open'], row['high'], row['low'], row['close']
            color = '#089981' if c >= o else '#F23645'  # Verde / Rojo TradingView
            
            # Mecha
            ax.plot([i, i], [l, h], color=color, linewidth=1.0)
            # Cuerpo
            ax.bar(i, abs(c - o), width, bottom=min(o, c), color=color, edgecolor=color)

        # Dibujar Medias Móviles si existen
        if 'ema50' in plot_df.columns:
            ax.plot(plot_df.index, plot_df['ema50'], color='#2962FF', linewidth=1.2, label='EMA 50', alpha=0.8)
        if 'ema200' in plot_df.columns:
            ax.plot(plot_df.index, plot_df['ema200'], color='#FF9800', linewidth=1.2, label='EMA 200', alpha=0.8)

        # Dibujar Caja de Posición (Estilo TradingView Long/Short Tool)
        start_box = max(0, n - 14)
        end_box = n + 8

        entry = setup.entry_price
        sl = setup.stop_loss
        tp2 = setup.tp2

        if setup.direction == "LONG":
            # Zona Verde (Ganancia TP)
            ax.fill_between([start_box, end_box], entry, tp2, color='#089981', alpha=0.25, edgecolor='#089981', linewidth=1.5)
            # Zona Roja (Pérdida SL)
            ax.fill_between([start_box, end_box], sl, entry, color='#F23645', alpha=0.25, edgecolor='#F23645', linewidth=1.5)
        else:
            # Short: Zona Verde abajo (TP)
            ax.fill_between([start_box, end_box], tp2, entry, color='#089981', alpha=0.25, edgecolor='#089981', linewidth=1.5)
            # Zona Roja arriba (SL)
            ax.fill_between([start_box, end_box], entry, sl, color='#F23645', alpha=0.25, edgecolor='#F23645', linewidth=1.5)

        # Líneas de Niveles y Etiquetas
        ax.axhline(entry, color='#D1D4DC', linestyle='--', linewidth=1.2)
        ax.axhline(tp2, color='#089981', linestyle='-', linewidth=1.5)
        ax.axhline(sl, color='#F23645', linestyle='-', linewidth=1.5)

        # Textos de Precios
        ax.text(end_box, entry, f"  Entry: {entry:.2f}", color='#D1D4DC', verticalalignment='center', fontweight='bold', fontsize=9)
        ax.text(end_box, tp2, f"  TP: {tp2:.2f} (R:R 1:{setup.rr_tp2})", color='#089981', verticalalignment='center', fontweight='bold', fontsize=9)
        ax.text(end_box, sl, f"  SL: {sl:.2f}", color='#F23645', verticalalignment='center', fontweight='bold', fontsize=9)

        # Formato de Ejes y Título
        dir_label = "[LONG]" if setup.direction == "LONG" else "[SHORT]"
        ax.set_title(f"TRADI COPILOT | {setup.symbol} {dir_label} | R:R 1:{setup.rr_tp2} | Confianza: {setup.confidence_score:.1f}%", 
                     fontsize=12, fontweight='bold', color='#E0E3EB', pad=15)
        
        ax.grid(True, linestyle=':', alpha=0.2, color='#787B86')
        ax.set_xlim(-1, end_box + 6)
        
        # Ajustar límites de Y con margen
        y_min = min(plot_df['low'].min(), sl, tp2) * 0.995
        y_max = max(plot_df['high'].max(), sl, tp2) * 1.005
        ax.set_ylim(y_min, y_max)

        # Guardar gráfico con nombre de archivo limpio
        clean_symbol = setup.symbol.replace("/", "_").replace("-", "_").upper()
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"signal_{clean_symbol}_{setup.direction}_{timestamp_str}.png"
        filepath = os.path.join(self.output_dir, filename)
        
        plt.tight_layout()
        plt.savefig(filepath, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        plt.close(fig)

        setup.chart_path = filepath
        return filepath
