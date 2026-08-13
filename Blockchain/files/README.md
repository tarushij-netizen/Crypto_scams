# BCFL Prototype — Rotating Pseudonyms + Masked Updates on a Toy Blockchain

Pure-Python proof-of-concept for the design discussed: **identity-layer
privacy (stealth-address pseudonym rotation) + content-layer privacy
(pairwise-masked SecAgg-style updates) committed to a hash-chained
blockchain**, running a real (small) federated learning task.

## Files

| File | Role |
|---|---|
| `blockchain.py` | Hash-chained blocks, simple PoW, tamper detection (the chain-security layer) |
| `stealth.py` | DKSAP-style stealth addresses on secp256k1 — one-time address per round, self-recognizable via view key (the identity layer / point 1) |
| `secure_agg.py` | Pairwise ECDH-derived additive masking — Bonawitz-style, simplified (no dropout recovery) (the content layer) |
| `fl_task.py` | The actual FL task: linear regression over synthetic per-client data |
| `demo.py` | Wires it all together and prints concrete evidence for each claim |

## Run

```bash
pip install ecdsa numpy
python3 demo.py
```

## What it proves (each maps to a claim from our discussion)

1. **Chain integrity** — tampering with a past block's transaction data
   is detected immediately (`is_chain_valid()` fails with the exact
   block and reason).
2. **Identity unlinkability** — a client's address is different every
   round; consecutive addresses differ in ~128/256 bits (statistically
   indistinguishable from random), yet the client can still verify its
   own past addresses via its view key (needed for reputation
   tracking), while a *different* client's view key correctly fails to
   claim them.
3. **Content privacy + correct aggregation** — any single client's
   on-chain (masked) value is far (L2 distance) from its real
   gradient, but the **sum** of all masked values in a round exactly
   equals the sum of all real gradients (pairwise masks cancel) — this
   is the concrete demonstration of why hashing alone cannot give
   content privacy, but pairing it with additive masking can.
4. **The model still trains** — using only chain-committed pseudonymous,
   masked updates, the global model converges toward the true weights.

## Known simplifications (state these explicitly if you use this in your thesis)

- **No dropout recovery.** Real Bonawitz SecAgg secret-shares each
  client's self-mask seed so the server can reconstruct and subtract
  masks for clients who drop mid-round. This prototype omits that
  (assumes no dropout) to keep the core cancellation mechanism
  legible — flag this as future work, not a limitation of the
  underlying idea.
- **No self-mask.** Only pairwise masks are used; a self-mask term
  (Bonawitz's `PRG(b_i)`) was dropped since it isn't needed to
  demonstrate the cancellation property, and would just add
  dropout-recovery complexity without changing the privacy argument
  for this demo's threat model (a passive observer of the chain).
- **No reputation-binding zkSNARK yet.** This prototype shows rotation
  (point 1) and masking (content privacy) working together on-chain,
  but does not yet implement the DARTIC/Dimitriou-style ZK proof that
  binds rotating pseudonyms to a persistent reputation score (point
  4). That's the natural next module to add — `stealth.py`'s
  `owns_address` check is exactly the kind of statement that would get
  wrapped in a zero-knowledge proof so it convinces a smart contract
  without revealing which client it is.
- **PoW difficulty is artificially low** (3 leading hex zeros) purely
  so the demo runs fast — not a claim about mainnet-grade consensus.
