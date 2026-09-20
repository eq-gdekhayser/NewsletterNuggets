"""Covered-call opportunity evaluation.

Given a share inventory (with tax lots), a volatility regime, and an option
chain, this module scans strike/tenor combinations and returns the most
attractive covered call to write against the position, or reports that no
combination clears the risk / yield / tax constraints.

Modeling notes
--------------
* Terminal share price S_T is modeled as lognormal under the *real-world*
  drift (mu - q), i.e. ln(S_T/S0) ~ N((mu - q - 0.5*sigma^2)*tau, sigma^2*tau).
  Assignment probability and the expected short-call payoff are derived from
  the same distribution so they stay internally consistent.
* The value added by writing a covered call versus simply holding the shares is
  exactly    premium - E[max(S_T - K, 0)] - tax_drag,
  because the equity legs (min(S_T, K) vs. S_T) differ only by -max(S_T-K, 0).
  We use the closed form for E[max(S_T - K, 0)] rather than approximating it.
"""

import math
from typing import List, Dict, Any, Callable

from scipy.stats import norm


def normal_cdf(x: float) -> float:
    return float(norm.cdf(x))


def inverse_normal_cdf(p: float) -> float:
    return float(norm.ppf(p))


def compute_specid_tax_drag(
    K: float,
    contract_shares: int,
    lots: List[Dict[str, Any]],
    rate_st: float,
    rate_lt: float,
) -> float:
    """Tax owed on assignment at strike K using HIFO (highest-basis-first) lot
    selection, which minimizes the realized gain per SpecID accounting.

    Loss lots (basis > K) contribute zero gain here; their loss-harvesting
    benefit is intentionally not credited, keeping the estimate conservative.
    """
    sorted_lots = sorted(lots, key=lambda x: x["basis"], reverse=True)
    remaining = contract_shares
    total_tax = 0.0

    for lot in sorted_lots:
        if remaining <= 0:
            break
        allocated = min(int(lot["qty"]), remaining)
        gain = max(0.0, K - lot["basis"])
        rate = rate_lt if lot["holding_days"] >= 365 else rate_st
        total_tax += allocated * gain * rate
        remaining -= allocated

    return total_tax


def evaluate_portfolio_covered_call(
    ticker: str,
    S0: float,
    mu: float,
    q: float,
    sigma: float,
    hv: float,
    iv_rank: float,
    lots: List[Dict[str, Any]],
    tax_st: float,
    tax_lt: float,
    dte_list: List[int],
    get_option_chain: Callable[[str, int], List[Dict[str, Any]]],
    alpha_max: float = 0.20,
    min_iv_rank: float = 30.0,
    min_iv_hv_ratio: float = 1.15,
    min_net_annualized_yield: float = 0.04,
    r: float = 0.0,
    tax_penalty_weight: float = 1.0,
    require_both_vol_signals: bool = False,
) -> Dict[str, Any]:
    """Scan an option chain for the best covered call to write.

    Parameters
    ----------
    r : float
        Continuously-compounded discount rate used to present-value the
        expiry-dated terms (short-call payoff and assignment tax) so they are
        comparable to the premium received today. Defaults to 0.0 (no
        discounting), which reproduces the un-discounted comparison.
    tax_penalty_weight : float
        Fraction of the assignment tax treated as a true economic drag,
        in [0, 1]. Buy-and-hold defers the *same* gains rather than avoiding
        them, so charging the full liability against a tax-free baseline
        overstates the cost. Use 1.0 for the conservative upper bound,
        0.0 for a tax-advantaged account, or an intermediate value to reflect
        expected deferral.
    require_both_vol_signals : bool
        If False (default), the vol gate passes when *either* IV rank or the
        IV/HV ratio is elevated. If True, it requires *both* — the stricter
        "premium is genuinely rich" interpretation.
    """
    if sigma <= 0:
        return {"status": "no opportunity here", "reason": "Non-positive volatility."}

    # --- Volatility filter gate -------------------------------------------
    iv_hv_ratio = sigma / hv if hv > 0 else 1.0
    rank_ok = iv_rank >= min_iv_rank
    ratio_ok = iv_hv_ratio >= min_iv_hv_ratio
    vol_ok = (rank_ok and ratio_ok) if require_both_vol_signals else (rank_ok or ratio_ok)
    if not vol_ok:
        return {
            "status": "no opportunity here",
            "reason": f"Vol filter failed: iv_rank={iv_rank}, iv/hv={iv_hv_ratio:.2f}",
        }

    total_shares = sum(lot["qty"] for lot in lots)
    contract_shares = 100
    if total_shares < contract_shares:
        return {
            "status": "no opportunity here",
            "reason": f"Insufficient share inventory ({total_shares} < {contract_shares})",
        }
    contracts_available = int(total_shares // contract_shares)

    # Loop-invariant: the z-score for the assignment-probability ceiling.
    z_target = inverse_normal_cdf(1.0 - alpha_max)

    best_score = -float("inf")
    best_opportunity = None

    for tau_days in dte_list:
        tau = tau_days / 365.25
        sqrt_tau = sigma * math.sqrt(tau)
        discount = math.exp(-r * tau)

        # Minimum strike whose assignment probability is <= alpha_max, i.e. the
        # (1 - alpha_max) quantile of the terminal price distribution.
        ln_K_S0_min = z_target * sqrt_tau + (mu - q - 0.5 * sigma ** 2) * tau
        K_min = S0 * math.exp(ln_K_S0_min)

        # E[S_T] under real-world drift, and its present value.
        expected_spot_drift = S0 * math.exp((mu - q) * tau)

        chain = get_option_chain(ticker, tau_days)
        if not chain:
            continue

        for opt in chain:
            K = opt.get("strike")
            prem = opt.get("mid")
            bid = opt.get("bid")
            ask = opt.get("ask")
            open_int = opt.get("open_interest", 0)

            if K is None or K < max(S0, K_min):
                continue
            if prem is None or prem <= 0:
                continue

            # Liquidity: require a two-sided market and a tight relative spread.
            if bid is None or ask is None or ask < bid:
                continue
            spread = ask - bid
            if spread / prem > 0.25 or open_int < 50:
                continue

            # Consistent assignment probability and short-call payoff from the
            # same lognormal:  d2 == -d2_rw,  p_assign == N(d2).
            d1 = (math.log(S0 / K) + (mu - q + 0.5 * sigma ** 2) * tau) / sqrt_tau
            d2 = d1 - sqrt_tau
            p_assign = normal_cdf(d2)
            if p_assign > alpha_max:
                continue

            # Expected value the short call gives up at expiry (undiscounted):
            #   E[max(S_T - K, 0)] = E[S_T]*N(d1) - K*N(d2)
            expected_call_payoff = expected_spot_drift * normal_cdf(d1) - K * p_assign

            # Assignment tax, present-valued and scaled by how much of it is a
            # genuine economic cost versus mere acceleration of deferred tax.
            assignment_tax = compute_specid_tax_drag(
                K, contract_shares, lots, tax_st, tax_lt
            )
            expected_tax_penalty = p_assign * assignment_tax
            tax_drag_per_share = (
                tax_penalty_weight * discount * expected_tax_penalty / contract_shares
            )

            # Value added vs. buy-and-hold, per share. The equity legs cancel
            # analytically, leaving: premium - PV(E[call payoff]) - PV(tax).
            excess_return_per_share = (
                prem - discount * expected_call_payoff - tax_drag_per_share
            )
            annualized_net_yield = (excess_return_per_share / S0) * (365.25 / tau_days)

            if annualized_net_yield >= min_net_annualized_yield:
                score = annualized_net_yield / (p_assign + 0.05)
                if score > best_score:
                    best_score = score
                    best_opportunity = {
                        "status": "opportunity found",
                        "ticker": ticker,
                        "strike": K,
                        "dte": tau_days,
                        "mid_premium": prem,
                        "p_assign": p_assign,
                        "expected_call_payoff_per_share": expected_call_payoff,
                        "expected_tax_drag_per_contract": expected_tax_penalty,
                        "annualized_net_yield": annualized_net_yield,
                        "score": score,
                        "contracts_available": contracts_available,
                    }

    if best_opportunity is None:
        return {
            "status": "no opportunity here",
            "reason": "No strike/tenor combination satisfied yield, risk, and tax constraints.",
        }

    return best_opportunity
