"""Compare two saved analyses."""
from __future__ import annotations

import math

from app.application.services.formatting import signed as _f


# --------------------------------------------------------------------------
# Comparing two analyses
# --------------------------------------------------------------------------
def compare_summaries(base: dict, latest: dict) -> dict:
    bs, ls = base["stats"], latest["stats"]

    def pair(b: float, l: float, nd: int = 3) -> dict:
        return {"base": b, "latest": l, "delta": round(l - b, nd)}

    out: dict = {
        "avg": pair(bs["mean"], ls["mean"]),
        "net": pair(base["net_sentiment"], latest["net_sentiment"], 1),
        "pos_pct": pair(base["label_pct"]["Positive"], latest["label_pct"]["Positive"], 1),
        "neg_pct": pair(base["label_pct"]["Negative"], latest["label_pct"]["Negative"], 1),
        "rows": pair(base["scored"], latest["scored"], 0),
    }
    se = math.sqrt(bs["std"] ** 2 / base["scored"] + ls["std"] ** 2 / latest["scored"])
    z = (ls["mean"] - bs["mean"]) / se if se > 0 else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))  # two-sided
    out["significance"] = {"z": round(z, 2), "p": round(p, 4), "meaningful": p < 0.05}

    bg, lg = base.get("by_group"), latest.get("by_group")
    out["groups"] = None
    if bg and lg and bg["column"] == lg["column"]:
        bm, lm = {i["name"]: i for i in bg["items"]}, {i["name"]: i for i in lg["items"]}
        both = [
            {
                "name": k,
                "base_avg": bm[k]["avg"],
                "latest_avg": lm[k]["avg"],
                "delta": round(lm[k]["avg"] - bm[k]["avg"], 3),
                "base_rows": bm[k]["rows"],
                "latest_rows": lm[k]["rows"],
            }
            for k in bm
            if k in lm and bm[k]["rows"] >= 3 and lm[k]["rows"] >= 3
        ]
        both.sort(key=lambda g: g["delta"], reverse=True)
        out["groups"] = {
            "column": bg["column"],
            "items": both,
            "new": [k for k in lm if k not in bm],
            "dropped": [k for k in bm if k not in lm],
        }

    ins: list[dict] = []
    d = out["avg"]["delta"]
    sig = out["significance"]
    if sig["meaningful"] and abs(d) >= 0.05:
        ins.append({"level": "good" if d > 0 else "warn", "text": f"Average sentiment {'improved' if d > 0 else 'declined'} from {_f(bs['mean'])} to {_f(ls['mean'])} ({_f(d)}). That is larger than normal sample-to-sample variation (p = {sig['p']:.3f}), so it is likely a real change."})
    else:
        ins.append({"level": "info", "text": f"Average sentiment is about the same ({_f(bs['mean'])} to {_f(ls['mean'])}). The difference is within normal variation (p = {sig['p']:.2f}), so avoid reading it as a real change."})
    dn = out["neg_pct"]["delta"]
    if abs(dn) >= 5:
        ins.append({"level": "warn" if dn > 0 else "good", "text": f"The share of negative rows {'rose' if dn > 0 else 'fell'} by {abs(dn):.1f} points ({out['neg_pct']['base']:.0f}% to {out['neg_pct']['latest']:.0f}%)."})
    g = out["groups"]
    if g and g["items"]:
        up, down = g["items"][0], g["items"][-1]
        if up["delta"] >= 0.15:
            ins.append({"level": "good", "text": f"Biggest improvement: \u201c{up['name']}\u201d ({_f(up['base_avg'])} to {_f(up['latest_avg'])})."})
        if down["delta"] <= -0.15:
            ins.append({"level": "warn", "text": f"Biggest decline: \u201c{down['name']}\u201d ({_f(down['base_avg'])} to {_f(down['latest_avg'])}). Look into what changed."})
    a, b = base["scored"], latest["scored"]
    if max(a, b) > 2 * min(a, b):
        ins.append({"level": "info", "text": f"The two analyses differ a lot in size ({a:,} versus {b:,} rows). Make sure they cover comparable data before drawing conclusions."})
    out["insights"] = ins
    return out
