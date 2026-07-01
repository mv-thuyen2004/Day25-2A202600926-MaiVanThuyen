"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing

DAYS = 30


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = 0.0
    recs = []
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        tier = pricing.recommend_tier(hpd, interruptible)
        if tier == "spot":
            sim = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)
            opt_cost = sim["spot_cost"]
        elif tier == "reserved":
            opt_cost = gpu_hours * num(c["reserved_3yr_hr"])
        else:
            opt_cost = on_demand_cost

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": tier,
                     "on_demand": round(on_demand_cost), "optimized": round(opt_cost)})

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0

    # Extension 5: Carbon-aware Scheduling for interruptible training workloads
    from finops.sustainability import carbon_g, energy_cost_usd
    carbon_report = []
    total_co2_us = 0.0
    total_co2_eu = 0.0
    for j in jobs:
        interruptible = bool(int(num(j["interruptible"])))
        if interruptible:
            gtype = j["gpu_type"]
            ngpu = int(num(j["num_gpus"]))
            hpd = num(j["hours_per_day"])
            gpu_hours = hpd * DAYS * ngpu
            watts = num(cat[gtype]["watts"])
            
            energy_wh = gpu_hours * watts
            co2_us = carbon_g(energy_wh, "us-east-1")
            co2_eu = carbon_g(energy_wh, "europe-north1")
            cost_us = energy_cost_usd(energy_wh, "us-east-1")
            cost_eu = energy_cost_eu = energy_cost_usd(energy_wh, "europe-north1")
            
            total_co2_us += co2_us
            total_co2_eu += co2_eu
            
            carbon_report.append({
                "job_id": j["job_id"],
                "gpu_type": gtype,
                "energy_kwh": round(energy_wh / 1000, 1),
                "co2_us_kg": round(co2_us / 1000, 1),
                "co2_eu_kg": round(co2_eu / 1000, 1),
                "cost_us": round(cost_us, 2),
                "cost_eu": round(cost_eu, 2),
            })

    carbon_saved_pct = (1.0 - total_co2_eu / total_co2_us) * 100.0 if total_co2_us > 0 else 0.0

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'tier':11}{'on-demand':>12}{'optimized':>12}")
        for r in recs:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['tier']:11}${r['on_demand']:>11,}${r['optimized']:>11,}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")
        
        print("\n== Extension 5: Carbon-aware Scheduling (Migrating us-east-1 -> europe-north1) ==")
        print(f"{'job':18}{'gpu':7}{'energy(kWh)':>12}{'CO2 US(kg)':>12}{'CO2 EU(kg)':>12}{'Cost US':>10}{'Cost EU':>10}")
        for c in carbon_report:
            print(f"{c['job_id']:18}{c['gpu_type']:7}{c['energy_kwh']:>12,}{c['co2_us_kg']:>12,}{c['co2_eu_kg']:>12,}${c['cost_us']:>9,}${c['cost_eu']:>9,}")
        print(f"Total Carbon Reduction: {total_co2_us - total_co2_eu:,.1f} gCO2e ({carbon_saved_pct:.1f}% reduction)")

    return {"recommendations": recs, "on_demand_monthly": round(on_demand_monthly),
            "optimized_monthly": round(optimized_monthly), "savings_pct": round(savings_pct, 1),
            "carbon_report": carbon_report, "carbon_savings_g": round(total_co2_us - total_co2_eu, 1)}


if __name__ == "__main__":
    run()
