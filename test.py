import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def adaptive_weights(signals, returns, lookback=60):
    weights = {}

    # Calculate rolling ICs for each signal
    for signal_name, signal_values in signals.items():
        # Calculate rolling IC (correlation with next period returns)
        rolling_ic = [
            pearsonr(
                signal_values[i : i + lookback], returns[i + 1 : i + lookback + 1]
            )[0]
            for i in range(len(signal_values) - lookback)
        ]

        # Calculate rolling SNR
        rolling_snr = [
            abs(np.mean(signal_values[i : i + lookback]))
            / np.std(signal_values[i : i + lookback])
            for i in range(len(signal_values) - lookback)
        ]

        # Combined metric (IC-adjusted SNR)
        ic_snr_product = [abs(ic) * snr for ic, snr in zip(rolling_ic, rolling_snr)]

        weights[signal_name] = ic_snr_product[-1]  # Use most recent value

    # Normalize weights to sum to 1
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def detect_regime(market_data, lookback=30):
    volatility = np.std(market_data["returns"][-lookback:])
    avg_volume = np.mean(market_data["volume"][-lookback:])

    # Define regimes based on volatility and volume
    if volatility > volatility_threshold and avg_volume > volume_threshold:
        return "high_vol_high_vol"
    elif volatility > volatility_threshold:
        return "high_vol_low_vol"
    elif avg_volume > volume_threshold:
        return "low_vol_high_vol"
    else:
        return "low_vol_low_vol"


def construct_adaptive_signal(signals, market_data, signal_history):
    # Get base weights from IC and SNR
    base_weights = adaptive_weights(signals, market_data["returns"])

    # Detect current market regime
    regime = detect_regime(market_data)

    # Adjust weights based on historical performance in this regime
    regime_performance = calculate_regime_performance(signals, regime, signal_history)

    # Final weights combine base metrics with regime-specific performance
    final_weights = {}
    for signal_name in signals:
        final_weights[signal_name] = (
            base_weights[signal_name] * regime_performance[signal_name]
        )

    # Normalize weights
    weight_sum = sum(final_weights.values())
    final_weights = {k: v / weight_sum for k, v in final_weights.items()}

    # Construct composite signal
    composite_signal = sum(final_weights[s] * signals[s][-1] for s in signals)

    return composite_signal, final_weights


def decay_weighted_ic(signal, returns, half_life=10):
    correlations = []
    weights = []

    for i in range(len(signal) - 1):
        # Calculate correlation between signal and next period return
        period_ic = pearsonr(signal[i : i + 1], returns[i + 1 : i + 2])[0]
        decay_weight = np.exp(-np.log(2) * (len(signal) - i - 1) / half_life)

        correlations.append(period_ic)
        weights.append(decay_weight)

    # Normalize weights
    weights = np.array(weights) / sum(weights)

    # Weighted average IC
    weighted_ic = sum(w * ic for w, ic in zip(weights, correlations))
    return weighted_ic


def predictive_power_score(signal, future_returns):
    # Discretize signal into quantiles
    signal_quantiles = pd.qcut(signal, 5, labels=False)

    # Calculate average future return for each quantile
    quantile_returns = {}
    for q in range(5):
        mask = signal_quantiles == q
        quantile_returns[q] = np.mean(future_returns[mask])

    # Calculate monotonicity score (how well ordered the quantiles are)
    sorted_returns = sorted(quantile_returns.values())
    monotonicity = 1.0

    if sorted_returns == list(quantile_returns.values()):
        monotonicity = 1.0  # perfectly monotonic increasing
    elif sorted_returns == list(reversed(list(quantile_returns.values()))):
        monotonicity = -1.0  # perfectly monotonic decreasing
    else:
        # Calculate partial monotonicity
        increases = sum(
            1
            for i in range(len(quantile_returns) - 1)
            if quantile_returns[i + 1] > quantile_returns[i]
        )
        decreases = sum(
            1
            for i in range(len(quantile_returns) - 1)
            if quantile_returns[i + 1] < quantile_returns[i]
        )
        monotonicity = (increases - decreases) / (len(quantile_returns) - 1)

    return abs(monotonicity)  # Return absolute value for weighting


def optimal_signal(signals, returns_history, market_data):
    # Current signals dictionary {name: latest_value}
    current_signals = {name: values[-1] for name, values in signals.items()}

    # Calculate core metrics for each signal
    metrics = {}
    for name, historical_values in signals.items():
        metrics[name] = {
            "ic": decay_weighted_ic(historical_values, returns_history),
            "snr": abs(np.mean(historical_values)) / np.std(historical_values),
            "pps": predictive_power_score(historical_values, returns_history[1:]),
            "regime_score": regime_performance[name][current_regime],
        }

    # Combine metrics into a single weight
    weights = {}
    for name, metric_dict in metrics.items():
        # Formula: IC * SNR * (1 + PPS) * Regime_Score
        weights[name] = (
            abs(metric_dict["ic"])
            * metric_dict["snr"]
            * (1 + metric_dict["pps"])
            * metric_dict["regime_score"]
        )

    # Normalize weights
    total = sum(weights.values())
    normalized_weights = {k: v / total for k, v in weights.items()}

    # Generate final signal
    final_signal = sum(
        normalized_weights[name] * value for name, value in current_signals.items()
    )

    return final_signal, normalized_weights


def process_tick_data(tick_data, sampling_interval_ms=100):
    """
    Process raw tick data into regular intervals with derived signals.

    Args:
        tick_data: DataFrame with columns like timestamp, price, volume, bid, ask
        sampling_interval_ms: Interval for resampling in milliseconds
    """
    # Convert to microsecond precision timestamps if not already
    tick_data["timestamp"] = pd.to_datetime(tick_data["timestamp"], unit="ns")
    tick_data = tick_data.set_index("timestamp")

    # Resample to regular intervals
    sampled_data = (
        tick_data.resample(f"{sampling_interval_ms}ms")
        .agg(
            {
                "price": "last",
                "volume": "sum",
                "bid": "last",
                "ask": "last",
                "bid_size": "last",
                "ask_size": "last",
            }
        )
        .dropna()
    )

    # Add mid price
    sampled_data["mid"] = (sampled_data["bid"] + sampled_data["ask"]) / 2

    # Calculate micro-returns for next interval prediction
    sampled_data["next_return"] = sampled_data["mid"].pct_change().shift(-1)

    return sampled_data


def generate_hft_signals(sampled_tick_data, lookback_intervals=20):
    """Generate HFT-specific signals from sampled tick data."""

    signals = {}

    # Order book imbalance
    signals["order_imbalance"] = (
        sampled_tick_data["bid_size"] - sampled_tick_data["ask_size"]
    ) / (sampled_tick_data["bid_size"] + sampled_tick_data["ask_size"])

    # Micro-price (size-weighted mid)
    total_size = sampled_tick_data["bid_size"] + sampled_tick_data["ask_size"]
    signals["micro_price_diff"] = (
        (
            sampled_tick_data["bid"] * sampled_tick_data["ask_size"]
            + sampled_tick_data["ask"] * sampled_tick_data["bid_size"]
        )
        / total_size
    ) - sampled_tick_data["mid"]

    # Spread changes
    sampled_tick_data["spread"] = sampled_tick_data["ask"] - sampled_tick_data["bid"]
    signals["spread_z"] = (
        sampled_tick_data["spread"]
        - sampled_tick_data["spread"].rolling(lookback_intervals).mean()
    ) / sampled_tick_data["spread"].rolling(lookback_intervals).std()

    # Trade flow imbalance (using tick rule for trade direction)
    sampled_tick_data["tick_direction"] = np.sign(sampled_tick_data["price"].diff())
    signals["trade_flow"] = (
        sampled_tick_data["tick_direction"]
        * sampled_tick_data["volume"]
        / sampled_tick_data["volume"].rolling(lookback_intervals).mean()
    )

    # Tick-by-tick volatility
    price_changes = sampled_tick_data["mid"].diff()
    signals["micro_volatility"] = (
        price_changes.rolling(lookback_intervals).std() / sampled_tick_data["mid"]
    )

    # VPIN (Volume-synchronized Probability of Informed Trading)
    # Simplified version
    signals["vpin_proxy"] = (
        sampled_tick_data["volume"].rolling(lookback_intervals).std()
        / sampled_tick_data["volume"].rolling(lookback_intervals).mean()
    )

    return signals


def detect_microstructure_regime(tick_data, lookback=50):
    """Detect market microstructure regime from tick data."""

    # Calculate metrics
    spread = tick_data["ask"] - tick_data["bid"]
    spread_avg = spread.rolling(lookback).mean().iloc[-1]
    volatility = tick_data["mid"].pct_change().rolling(lookback).std().iloc[-1]
    volume = tick_data["volume"].rolling(lookback).mean().iloc[-1]

    # Classify regime
    if spread_avg > spread_threshold and volatility > vol_threshold:
        return "wide_spread_high_vol"
    elif spread_avg > spread_threshold:
        return "wide_spread_low_vol"
    elif volatility > vol_threshold:
        return "tight_spread_high_vol"
    else:
        return "tight_spread_low_vol"


def optimal_signal_hft(signals, tick_data, lookback=100):
    """
    Generate optimal combination of signals for HFT prediction.
    Optimized for ultra-short-term predictions from tick data.

    Args:
        signals: Dictionary of signal series
        tick_data: Processed tick data DataFrame
        lookback: Number of ticks to use for signal evaluation
    """
    # Get future returns (target for prediction)
    future_returns = tick_data["next_return"].fillna(0)

    # Calculate metrics for each signal
    signal_metrics = {}
    for name, values in signals.items():
        signal_series = values.fillna(0)  # Handle NaN values

        # Skip signals with insufficient data
        if len(signal_series) < lookback:
            continue

        # Use only the most recent lookback period
        recent_signal = signal_series[-lookback:]
        recent_returns = future_returns[-lookback - 1 : -1]  # Align with future returns

        # Skip if not enough return data
        if len(recent_returns) < lookback:
            continue

        # Calculate signal metrics
        try:
            # Calculate IC (predictive correlation)
            ic = np.corrcoef(recent_signal, recent_returns)[0, 1]
            if np.isnan(ic):
                ic = 0

            # Calculate SNR
            mean_signal = np.mean(recent_signal)
            std_signal = np.std(recent_signal)
            snr = abs(mean_signal) / std_signal if std_signal > 0 else 0

            # Calculate hit rate (directional accuracy)
            signal_direction = np.sign(recent_signal)
            return_direction = np.sign(recent_returns)
            hit_rate = np.mean(signal_direction == return_direction)

            # Calculate decay-weighted IC (more weight to recent observations)
            weights = np.exp(np.linspace(-3, 0, lookback))
            weights = weights / np.sum(weights)
            ic_weighted = np.sum(weights * signal_direction * return_direction)

            # Store metrics
            signal_metrics[name] = {
                "ic": ic,
                "ic_weighted": ic_weighted,
                "snr": snr,
                "hit_rate": hit_rate,
            }
        except:
            # Skip signals that cause errors
            continue

    # Current regime detection
    current_regime = detect_microstructure_regime(tick_data)

    # Adaptive weighting based on metrics and regime
    weights = {}
    for name, metrics in signal_metrics.items():
        # Baseline weight formula - adjust these coefficients based on your strategy
        weight = (
            0.3 * abs(metrics["ic"])
            + 0.3 * metrics["ic_weighted"]
            + 0.2 * metrics["snr"]
            + 0.2
            * (metrics["hit_rate"] - 0.5)
            * 2  # Scale hit rate from [0.5,1] to [0,1]
        )

        # Apply regime-specific adjustments
        if current_regime == "wide_spread_high_vol":
            # In high volatility, wide spread regimes, favor order book signals
            if "imbalance" in name or "spread" in name:
                weight *= 1.5
        elif current_regime == "tight_spread_low_vol":
            # In low volatility, tight spread regimes, favor micro-price signals
            if "micro" in name:
                weight *= 1.5

        weights[name] = max(weight, 0)  # Ensure non-negative weights

    # Normalize weights to sum to 1
    weight_sum = sum(weights.values())
    if weight_sum > 0:
        weights = {k: v / weight_sum for k, v in weights.items()}
    else:
        # Default to equal weighting if all weights are zero
        signals_count = len(signal_metrics)
        weights = {k: 1 / signals_count for k in signal_metrics.keys()}

    # Apply weights to current signal values
    current_signals = {
        name: values.iloc[-1] for name, values in signals.items() if name in weights
    }

    # Generate final combined signal
    combined_signal = sum(
        weights.get(name, 0) * value for name, value in current_signals.items()
    )

    return combined_signal, weights


def calibrate_signal_thresholds(historical_tick_data, lookback_periods=[50, 100, 200]):
    """Calibrate optimal signal thresholds using historical tick data."""
    results = {}

    for lookback in lookback_periods:
        # Process data
        sampled_data = process_tick_data(historical_tick_data)
        signals = generate_hft_signals(sampled_data, lookback)

        # Generate signals for each historical point
        signal_values = []
        actual_returns = []

        # Use rolling window
        for i in range(lookback, len(sampled_data) - 1):
            window_data = sampled_data.iloc[: i + 1]
            window_signals = {
                name: values.iloc[: i + 1] for name, values in signals.items()
            }

            combined_signal, _ = optimal_signal_hft(
                window_signals, window_data, lookback
            )
            next_return = sampled_data["next_return"].iloc[i + 1]

            signal_values.append(combined_signal)
            actual_returns.append(next_return)

        # Analyze signal effectiveness
        signal_array = np.array(signal_values)
        return_array = np.array(actual_returns)

        # Find optimal thresholds by maximizing Sharpe ratio
        best_sharpe = -np.inf
        best_threshold = 0

        for threshold in np.linspace(0.1, 2.0, 20):
            # Only take positions when signal exceeds threshold
            positions = np.zeros_like(signal_array)
            positions[signal_array > threshold] = 1
            positions[signal_array < -threshold] = -1

            strategy_returns = positions * return_array
            sharpe = (
                np.mean(strategy_returns) / np.std(strategy_returns)
                if np.std(strategy_returns) > 0
                else 0
            )

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_threshold = threshold

        results[lookback] = {
            "optimal_threshold": best_threshold,
            "sharpe_ratio": best_sharpe,
            "ic": np.corrcoef(signal_array, return_array)[0, 1],
        }

    return results


def calibrate_signal_thresholds(historical_tick_data, lookback_periods=[50, 100, 200]):
    """Calibrate optimal signal thresholds using historical tick data."""
    results = {}

    for lookback in lookback_periods:
        # Process data
        sampled_data = process_tick_data(historical_tick_data)
        signals = generate_hft_signals(sampled_data, lookback)

        # Generate signals for each historical point
        signal_values = []
        actual_returns = []

        # Use rolling window
        for i in range(lookback, len(sampled_data) - 1):
            window_data = sampled_data.iloc[: i + 1]
            window_signals = {
                name: values.iloc[: i + 1] for name, values in signals.items()
            }

            combined_signal, _ = optimal_signal_hft(
                window_signals, window_data, lookback
            )
            next_return = sampled_data["next_return"].iloc[i + 1]

            signal_values.append(combined_signal)
            actual_returns.append(next_return)

        # Analyze signal effectiveness
        signal_array = np.array(signal_values)
        return_array = np.array(actual_returns)

        # Find optimal thresholds by maximizing Sharpe ratio
        best_sharpe = -np.inf
        best_threshold = 0

        for threshold in np.linspace(0.1, 2.0, 20):
            # Only take positions when signal exceeds threshold
            positions = np.zeros_like(signal_array)
            positions[signal_array > threshold] = 1
            positions[signal_array < -threshold] = -1

            strategy_returns = positions * return_array
            sharpe = (
                np.mean(strategy_returns) / np.std(strategy_returns)
                if np.std(strategy_returns) > 0
                else 0
            )

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_threshold = threshold

        results[lookback] = {
            "optimal_threshold": best_threshold,
            "sharpe_ratio": best_sharpe,
            "ic": np.corrcoef(signal_array, return_array)[0, 1],
        }

    return results
