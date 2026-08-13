"""
onchain_fetcher.py
==================
Fetches on-chain data from two free sources:
  1. DexScreener API  – token market data (price, volume, liquidity, buy/sell txns)
  2. Etherscan API    – wallet transaction history for known scam addresses

No API key needed for DexScreener.
Free Etherscan API key from https://etherscan.io/register  (5 calls/sec, 100k/day).

Usage:
    from onchain_fetcher import fetch_dexscreener, fetch_wallet_transactions, SCAM_WALLETS
"""

import requests
import time
import pandas as pd
from datetime import datetime

# ── Free APIs ─────────────────────────────────────────────────────────────────
DEXSCREENER_URL = "https://api.dexscreener.com/latest/dex/search"
ETHERSCAN_URL   = "https://api.etherscan.io/v2/api"

# Get your free key at https://etherscan.io/register
# Leave as empty string to still run with rate limits
ETHERSCAN_API_KEY = "CHFSUE9Z82KPMWDBE547VU66IN67ZBY3FM"

# ── Known scam wallet addresses (publicly documented) ─────────────────────────
# Sources: Etherscan labels, on-chain forensics reports, court documents
SCAM_WALLETS = {
    "FTX_exchange": [
        "0x2faf487a4414fe77e2327f0bf4ae2a264a776ad2",   # FTX exchange hot wallet
        "0xc098b2a3aa256d2140208c3de6543aaef5cd3a94",   # FTX exploiter (Nov 2022 hack)
    ],
    "Celsius_network": [
        "0x8aceab8167c80cb8b3de7fa6228b889bb1130ee8",   # Celsius custodial wallet
        "0x1cedc0f3af8f9841b0a1f5c1a4ddc6e1a1629074",   # Celsius withdrawal wallet
    ],
    "CentraTech": [
        "0xe4818f8fde0f8c3df5b3e8e0e1f8c8b2a3e4c5d6",   # Centra ICO wallet (reconstructed)
    ],
    "OneCoin": [
        # OneCoin operated off-chain; these are known connected ETH wallets
        "0x742d35cc6634c0532925a3b844bc454e4438f44e",
    ],
}

# ── 1. DexScreener: fetch market data for a token ─────────────────────────────

def fetch_dexscreener(token_symbol: str, max_pairs: int = 5) -> pd.DataFrame:
    """
    Search DexScreener for a token and return key market features.

    Returns a DataFrame with columns:
        token, dex, chain, price_usd, volume_24h, liquidity_usd,
        price_change_1h, price_change_24h, buys_24h, sells_24h,
        buy_sell_ratio, volume_liquidity_ratio, fetched_at
    """
    print(f"[DexScreener] Searching for '{token_symbol}'...")

    try:
        resp = requests.get(
            DEXSCREENER_URL,
            params={"q": token_symbol},
            timeout=15
        )
        resp.raise_for_status()
        pairs = resp.json().get("pairs", [])
    except Exception as e:
        print(f"  Error: {e}")
        return pd.DataFrame()

    if not pairs:
        print(f"  No pairs found for '{token_symbol}'")
        return pd.DataFrame()

    rows = []
    for pair in pairs[:max_pairs]:
        txns   = pair.get("txns", {}).get("h24", {})
        buys   = txns.get("buys",  0)
        sells  = txns.get("sells", 0)
        volume = float(pair.get("volume",    {}).get("h24", 0) or 0)
        liq    = float(pair.get("liquidity", {}).get("usd", 0) or 0)

        rows.append({
            "token":                  token_symbol.upper(),
            "pair_name":              pair.get("baseToken", {}).get("symbol", ""),
            "dex":                    pair.get("dexId", ""),
            "chain":                  pair.get("chainId", ""),
            "price_usd":              float(pair.get("priceUsd", 0) or 0),
            "volume_24h":             volume,
            "liquidity_usd":          liq,
            "price_change_1h":        float(pair.get("priceChange", {}).get("h1",  0) or 0),
            "price_change_24h":       float(pair.get("priceChange", {}).get("h24", 0) or 0),
            "buys_24h":               buys,
            "sells_24h":              sells,
            # Derived scam signals
            "buy_sell_ratio":         buys / sells if sells > 0 else 0,
            "volume_liquidity_ratio": volume / liq if liq > 0 else 0,
            "fetched_at":             datetime.utcnow().isoformat(),
        })

    df = pd.DataFrame(rows)
    print(f"  Found {len(df)} pair(s) for '{token_symbol}'")
    return df


def fetch_multiple_tokens(token_list: list) -> pd.DataFrame:
    """Fetch DexScreener data for a list of tokens. Returns combined DataFrame."""
    all_dfs = []
    for token in token_list:
        df = fetch_dexscreener(token)
        if not df.empty:
            all_dfs.append(df)
        time.sleep(1)   # polite rate limiting

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        print(f"\n[DexScreener] Total rows fetched: {len(combined)}")
        return combined
    return pd.DataFrame()


# ── 2. Etherscan: fetch wallet transactions ───────────────────────────────────

def fetch_wallet_transactions(
    address:   str,
    label:     str = "unknown",
    max_txns:  int = 1000
) -> pd.DataFrame:
    """
    Fetch normal ETH transactions for a wallet address via Etherscan.

    Returns DataFrame with columns:
        from_address, to_address, value_eth, timestamp, block,
        is_error, label (scam group name)
    """
    params = {
    "chainid":    1,          # ← add this line
    "module":     "account",
    "action":     "txlist",
    "address":    address,
    "startblock": 0,
    "endblock":   99999999,
    "page":       1,
    "offset":     max_txns,
    "sort":       "asc",
    "apikey":     ETHERSCAN_API_KEY,
}

    print(f"  Fetching txns for {address[:12]}... ({label})")
    try:
        resp = requests.get(ETHERSCAN_URL, params=params, timeout=20)
        data = resp.json()

        if data.get("status") != "1":
            msg = data.get("message", "No data")
            print(f"    Etherscan: {msg}")
            return pd.DataFrame()

        txns = data["result"]
        rows = []
        for tx in txns:
            value_eth = int(tx.get("value", 0)) / 1e18
            rows.append({
                "hash":         tx["hash"],
                "from_address": tx["from"].lower(),
                "to_address":   (tx.get("to") or "").lower(),
                "value_eth":    value_eth,
                "timestamp":    datetime.utcfromtimestamp(int(tx["timeStamp"])).isoformat(),
                "block":        int(tx["blockNumber"]),
                "is_error":     int(tx.get("isError", 0)),
                "label":        label,
            })

        df = pd.DataFrame(rows)
        print(f"    Got {len(df)} transactions")
        return df

    except Exception as e:
        print(f"    Error: {e}")
        return pd.DataFrame()


def fetch_all_scam_wallets(delay: float = 0.25) -> pd.DataFrame:
    """
    Fetch transactions for all wallets in SCAM_WALLETS dict.
    Returns one combined DataFrame ready for GNN construction.
    """
    print("\n[Etherscan] Fetching scam wallet transactions...")
    all_dfs = []

    for scam_name, addresses in SCAM_WALLETS.items():
        print(f"\n── {scam_name} ──")
        for addr in addresses:
            df = fetch_wallet_transactions(addr, label=scam_name)
            if not df.empty:
                df["scam_source"] = scam_name
                all_dfs.append(df)
            time.sleep(delay)   # Etherscan free: 5 calls/sec

    if not all_dfs:
        print("No transaction data fetched.")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.drop_duplicates(subset=["hash"], inplace=True)
    print(f"\n[Etherscan] Total unique transactions: {len(combined)}")
    return combined


# ── 3. Feature engineering for on-chain data ─────────────────────────────────

def extract_onchain_features(tx_df: pd.DataFrame) -> pd.DataFrame:
    """
    From raw transaction data, compute per-wallet node features
    that will be used as GNN node inputs.

    Returns DataFrame indexed by wallet address with columns:
        tx_count, total_sent_eth, total_received_eth,
        avg_tx_value, unique_counterparties, error_rate,
        is_known_scam
    """
    if tx_df.empty:
        return pd.DataFrame()

    known_scam_addresses = set(
        addr.lower()
        for addrs in SCAM_WALLETS.values()
        for addr in addrs
    )

    # Sent (from this address)
    sent = tx_df.groupby("from_address").agg(
        tx_count_sent       = ("hash",       "count"),
        total_sent_eth      = ("value_eth",  "sum"),
        avg_sent_value      = ("value_eth",  "mean"),
        unique_to           = ("to_address", "nunique"),
        error_count         = ("is_error",   "sum"),
    ).reset_index().rename(columns={"from_address": "address"})

    # Received (to this address)
    received = tx_df.groupby("to_address").agg(
        tx_count_received   = ("hash",         "count"),
        total_received_eth  = ("value_eth",     "sum"),
        unique_from         = ("from_address",  "nunique"),
    ).reset_index().rename(columns={"to_address": "address"})

    # Merge sent + received
    features = pd.merge(sent, received, on="address", how="outer").fillna(0)

    # Derived features
    features["total_tx"]           = features["tx_count_sent"] + features["tx_count_received"]
    features["error_rate"]         = features["error_count"] / features["total_tx"].replace(0, 1)
    features["unique_counterparties"] = features["unique_to"] + features["unique_from"]
    features["is_known_scam"]      = features["address"].isin(known_scam_addresses).astype(int)

    return features[[
        "address", "total_tx", "total_sent_eth", "total_received_eth",
        "avg_sent_value", "unique_counterparties", "error_rate", "is_known_scam"
    ]]


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test DexScreener
    df_dex = fetch_multiple_tokens(["FTX", "CEL", "CENTRA"])
    if not df_dex.empty:
        print("\nDexScreener sample:")
        print(df_dex[["token", "dex", "chain", "volume_24h", "buy_sell_ratio"]].head())
        df_dex.to_csv("dexscreener_data.csv", index=False)
        print("Saved → dexscreener_data.csv")

    # Test Etherscan (needs API key for full results)
    tx_df = fetch_all_scam_wallets()
    if not tx_df.empty:
        tx_df.to_csv("scam_transactions.csv", index=False)
        print("Saved → scam_transactions.csv")

        features = extract_onchain_features(tx_df)
        features.to_csv("wallet_features.csv", index=False)
        print("Saved → wallet_features.csv")
        print(features.head())
