---
name: covered-call-evaluator
description: >-
  How to run NewsletterNuggets' covered-call evaluator
  (covered_call.evaluate_portfolio_covered_call) to find the best covered call
  to write against a share position. Use this whenever the user wants to
  screen, select, or size a covered call / short call against stock they own —
  including phrasings like "which call should I sell against my AAPL", "find a
  covered call", "what strike and expiration", "evaluate writing calls on this
  position", "optimal covered call", or anything involving assignment
  probability, premium yield, or the tax hit from being called away on
  specific lots. Reach for this skill even when the user names a ticker and a
  goal without saying "covered call" explicitly, as long as they hold shares
  and are thinking about selling upside calls.
---

# Covered-Call Evaluator

`covered_call.py` scans an option chain and returns the single most attractive
covered call to write against a share position, or reports that nothing clears
the risk / yield / tax constraints. It is a **strike-and-tenor selector**, not
a full execution planner — it picks one contract's strike and expiration and
tells you how many contracts your inventory could cover.

The whole point of the model is to answer one question honestly: **does writing
this call beat simply holding the shares, after accounting for capped upside,
assignment probability, and the tax you'd owe if called away?** Everything the
function computes ties back to that comparison.

## The one entry point

```python
from covered_call import evaluate_portfolio_covered_call

result = evaluate_portfolio_covered_call(
    ticker, S0, mu, q, sigma, hv, iv_rank,
    lots, tax_st, tax_lt, dte_list, get_option_chain,
    # optional tuning knobs below
)
```

Requires `scipy` (`pip install scipy`).

## Inputs you must supply

| Argument | Meaning |
|---|---|
| `ticker` | Symbol string; passed straight to `get_option_chain`. |
| `S0` | Current spot price of the underlying. |
| `mu` | Expected **real-world** annual drift of the underlying (a view, not the risk-free rate). Drives assignment probability. |
| `q` | Continuous dividend yield of the underlying (annualized). |
| `sigma` | Implied volatility used for the option (annualized, e.g. `0.35`). |
| `hv` | Historical/realized volatility, same units — used only for the IV/HV vol gate. |
| `iv_rank` | IV rank 0–100; used only for the vol gate. |
| `lots` | The tax lots you hold — see schema below. |
| `tax_st`, `tax_lt` | Short- and long-term capital-gains **rates** as decimals (e.g. `0.37`, `0.20`). |
| `dte_list` | Candidate expirations to scan, in **days** (e.g. `[21, 30, 45]`). |
| `get_option_chain` | A callback you provide — see below. |

### Lot schema (`lots`)

Each lot is a dict. Lots are how the model computes the **SpecID tax drag** of
being assigned — it sells your highest-basis shares first (HIFO) to minimize
realized gain.

```python
lots = [
    {"qty": 100, "basis": 82.50, "holding_days": 420},   # long-term lot
    {"qty": 100, "basis": 141.00, "holding_days": 45},   # short-term, high basis
]
```

- `qty` — shares in the lot.
- `basis` — cost basis per share.
- `holding_days` — days held; `>= 365` is taxed at `tax_lt`, otherwise `tax_st`.

Total shares across lots must be `>= 100` or the function returns "no
opportunity here". It reports `contracts_available = total_shares // 100`.

### The `get_option_chain` callback

You inject market data through this callback so the model stays free of any
particular data vendor. Signature and expected row shape:

```python
def get_option_chain(ticker: str, dte_days: int) -> list[dict]:
    # return the CALL options for this expiration
    return [
        {"strike": 115.0, "mid": 1.85, "bid": 1.80, "ask": 1.90, "open_interest": 640},
        ...
    ]
```

Each row needs `strike`, `mid`, `bid`, `ask`, `open_interest`. Rows missing a
two-sided quote, priced at zero, with a spread wider than 25% of mid, or with
open interest below 50 are filtered out as illiquid.

## Tuning knobs (all optional, with defaults)

| Argument | Default | Effect |
|---|---|---|
| `alpha_max` | `0.20` | Max acceptable assignment probability. Also sets the minimum strike (the `1 - alpha_max` price quantile). Lower it to stay further OTM. |
| `min_iv_rank` | `30.0` | IV-rank threshold for the vol gate. |
| `min_iv_hv_ratio` | `1.15` | IV/HV threshold for the vol gate. |
| `min_net_annualized_yield` | `0.04` | Reject anything whose annualized net yield over buy-and-hold is below this. |
| `r` | `0.0` | Discount rate that present-values the expiry-dated payoff and tax so they're comparable to premium received today. |
| `tax_penalty_weight` | `1.0` | How much of the assignment tax counts as real cost. `1.0` = conservative upper bound; `0.0` for a tax-advantaged account (IRA); in between to reflect that buy-and-hold only *defers* the same tax. |
| `require_both_vol_signals` | `True` | `True`: the vol gate needs **both** IV rank and IV/HV elevated (premium genuinely rich). `False`: either one passing is enough. |

## Reading the result

On success you get a dict like:

```python
{
    "status": "opportunity found",
    "ticker": "ABC",
    "strike": 115.0,
    "dte": 21,
    "mid_premium": 0.60,
    "p_assign": 0.048,                       # probability of being called away
    "expected_call_payoff_per_share": 0.198, # upside the short call gives up, per share
    "expected_tax_drag_per_contract": 0.0,   # expected (prob-weighted) assignment tax
    "annualized_net_yield": 0.070,           # yield ABOVE just holding the stock
    "score": 0.715,                          # yield / (p_assign + 0.05); the ranking metric
    "contracts_available": 2,                # total_shares // 100
}
```

- **`annualized_net_yield`** is the headline number: expected annualized return
  *in excess of* buy-and-hold. A covered call whose only value is the premium
  it collects while giving up equal expected upside scores ~0 — the model will
  not flatter it.
- **`p_assign`** is the real-world assignment probability, kept consistent with
  the strike floor and the payoff math.
- **`score`** ranks candidates by yield penalized for assignment risk; the
  returned opportunity is the highest-scoring one across all strikes and all
  `dte_list` tenors.

When nothing qualifies you get `{"status": "no opportunity here", "reason": ...}`
where `reason` names the gate that failed (vol filter, insufficient shares, or
no strike/tenor satisfying yield/risk/tax).

## Minimal worked example

```python
from covered_call import evaluate_portfolio_covered_call

lots = [
    {"qty": 100, "basis": 80.0,  "holding_days": 400},
    {"qty": 100, "basis": 120.0, "holding_days": 30},
]

def get_chain(ticker, dte):
    return [
        {"strike": 105, "mid": 2.0, "bid": 1.9,  "ask": 2.1,  "open_interest": 500},
        {"strike": 110, "mid": 1.2, "bid": 1.15, "ask": 1.25, "open_interest": 500},
        {"strike": 115, "mid": 0.6, "bid": 0.55, "ask": 0.65, "open_interest": 500},
    ]

result = evaluate_portfolio_covered_call(
    ticker="ABC", S0=100.0, mu=0.07, q=0.01, sigma=0.35, hv=0.28,
    iv_rank=45.0, lots=lots, tax_st=0.37, tax_lt=0.20,
    dte_list=[21, 45], get_option_chain=get_chain,
)
print(result["status"], result.get("strike"), result.get("annualized_net_yield"))
```

## Guidance when helping a user

- **Gather the market inputs first.** `S0`, `sigma`/`iv_rank`/`hv`, and the
  option chain are the pieces most users don't have at hand. If they can't
  supply a chain, help them wire `get_option_chain` to whatever data source
  they use rather than inventing quotes.
- **`mu` is a view, not a fact.** It's their expected drift for the stock;
  surface that it directly moves `p_assign` and the payoff, and pick a sober
  value (a modest equity-risk-premium-style number) unless they have a stronger
  opinion.
- **Match `tax_penalty_weight` to the account.** Use `0.0` inside an IRA/401k,
  `1.0` for the strict taxable-account upper bound, something in between if the
  user acknowledges they'd eventually realize the gain anyway.
- **A "no opportunity here" is a real answer.** The vol gate defaulting to
  strict (`require_both_vol_signals=True`) means the model declines to sell
  cheap premium. If the user wants to see marginal candidates, loosen the gate
  or lower `min_net_annualized_yield` deliberately — don't silently override it.
- **Sizing and rolling are out of scope.** The function picks one strike/tenor
  and reports `contracts_available`; it does not decide how many contracts to
  sell, ladder tenors, or plan rolls. Say so rather than implying it sized the
  position.
```

