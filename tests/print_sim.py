import sys, os
sys.path.insert(0, ".")
from src.simulation.capital_simulation import simulate_portfolio
for n, risk in [('Conservador (1%)', 0.01), ('Moderado (2%)', 0.02), ('Agresivo (5%)', 0.05)]:
    r = simulate_portfolio(500.0, risk_per_trade_pct=risk)
    print(f"{n} -> Mediana: ${r['median_final']:,.2f} | P10 (Pesimista): ${r['p10_final']:,.2f} | P90 (Optimista): ${r['p90_final']:,.2f} | MaxDD: -{r['max_dd_p95']*100:.1f}% | Ruina: {r['prob_of_ruin']*100:.2f}%")
