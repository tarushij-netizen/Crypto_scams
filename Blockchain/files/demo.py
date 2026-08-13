"""
demo.py
Full BCFL prototype tying together:
  - blockchain.py   (chain integrity layer)
  - stealth.py      (identity-layer privacy: rotating pseudonyms)
  - secure_agg.py   (content-layer privacy: masked updates)
  - fl_task.py       (the actual federated learning being secured)

Run: python3 demo.py

What this proves, concretely (not just claims about it):
  1. Identity unlinkability : each client's on-chain address is different
     every round and uncorrelated to any other round's address, yet the
     client itself can still recognize its own past addresses.
  2. Content privacy        : the value committed on-chain for any single
     client/round is far from that client's real gradient, but the SUM
     of all masked updates in a round exactly equals the sum of all real
     updates (masks cancel), so the FL aggregation stays correct.
  3. Chain integrity        : tampering with any past block's data is
     detected immediately by the validator.
  4. Model actually trains  : global model loss decreases round over
     round, using only chain-committed (pseudonymous, masked) updates.
"""

import numpy as np

from blockchain import Blockchain
from stealth import ClientIdentity
from secure_agg import AggKeypair, mask_update
from fl_task import DIM, TRUE_W, generate_client_data, local_gradient, mse

NUM_CLIENTS = 5
NUM_ROUNDS = 25
LR = 0.15


def main():
    print("=" * 70)
    print("BCFL PROTOTYPE — rotating pseudonyms + masked updates + blockchain")
    print("=" * 70)

    # ---- setup ----
    identities = [ClientIdentity(f"client_{i}") for i in range(NUM_CLIENTS)]
    agg_keys = [AggKeypair() for _ in range(NUM_CLIENTS)]
    client_data = [generate_client_data(seed=i) for i in range(NUM_CLIENTS)]

    chain = Blockchain()
    w = np.zeros(DIM)  # global model, starts at zero

    address_log = {i: [] for i in range(NUM_CLIENTS)}  # for the unlinkability check later
    plaintext_grad_log = []  # ground truth, for the privacy check later (demo-only visibility)

    print(f"\nTrue model weights (unknown to the aggregator): {np.round(TRUE_W, 3)}")
    print(f"Clients: {NUM_CLIENTS} | Rounds: {NUM_ROUNDS}\n")

    for rnd in range(1, NUM_ROUNDS + 1):
        tx_list = []
        real_grads = []
        masked_grads = []

        for i in range(NUM_CLIENTS):
            X, y = client_data[i]
            grad = local_gradient(X, y, w)
            real_grads.append(grad)

            masked = mask_update(grad, i, agg_keys, rnd)
            masked_grads.append(masked)

            address_hex, R = identities[i].derive_stealth_address()
            address_log[i].append(address_hex)

            tx_list.append(
                {
                    "address": address_hex,
                    "ephemeral_pubkey": R.to_string().hex(),
                    "masked_update": masked.tolist(),
                    "round": rnd,
                }
            )

        plaintext_grad_log.append(real_grads)

        block = chain.add_block(tx_list)

        # --- aggregator side: only ever sees masked_update values from the chain ---
        chain_masked_sum = sum(np.array(tx["masked_update"]) for tx in block.transactions)
        avg_grad = chain_masked_sum / NUM_CLIENTS
        w = w - LR * avg_grad

        loss = np.mean([mse(X, y, w) for X, y in client_data])
        print(f"Round {rnd:2d} | block #{block.index} mined (nonce={block.nonce}) "
              f"| avg-loss={loss:.5f}")

    # ================================================================
    print("\n" + "=" * 70)
    print("PROOF 1 — Chain integrity")
    print("=" * 70)
    valid, msg = chain.is_chain_valid()
    print(f"Before tampering: {msg}")

    # tamper with a past block
    tampered_block = chain.chain[3]
    original_tx = tampered_block.transactions[0]["masked_update"]
    tampered_block.transactions[0]["masked_update"] = [0.0] * DIM  # attacker edits history
    valid, msg = chain.is_chain_valid()
    print(f"After tampering block 3, client 0's update: {msg}")
    tampered_block.transactions[0]["masked_update"] = original_tx  # restore for rest of demo

    # ================================================================
    print("\n" + "=" * 70)
    print("PROOF 2 — Identity-layer unlinkability (rotating pseudonyms)")
    print("=" * 70)
    print("Client 0's on-chain address across all rounds:")
    for rnd, addr in enumerate(address_log[0], start=1):
        print(f"  round {rnd}: {addr}")

    # Hamming-distance-style check: consecutive addresses should look unrelated
    diffs = []
    for a, b in zip(address_log[0][:-1], address_log[0][1:]):
        ba, bb = bytes.fromhex(a), bytes.fromhex(b)
        hamming = sum(bin(x ^ y).count("1") for x, y in zip(ba, bb))
        diffs.append(hamming)
    print(f"\nBit-level Hamming distance between consecutive addresses: {diffs}")
    print(f"(32-byte hash = 256 bits; ~128 expected if addresses are unrelated/random -> "
          f"observed avg = {np.mean(diffs):.1f})")

    print("\nBut the client itself can still recognize its own past addresses")
    print("(self-linkability via the view key, needed for reputation tracking):")
    # re-derive R values is not stored above except inside tx_list; fetch from chain
    all_txs_client0 = []
    for b in chain.chain[1:]:
        for tx in b.transactions:
            if tx["address"] in address_log[0]:
                all_txs_client0.append(tx)
    # NOTE: matching by address here is just to fetch R for the demo check below;
    # an outside observer could NOT do this filtering, since nothing on-chain
    # marks these as belonging to client 0.
    from ecdsa import VerifyingKey, SECP256k1
    ok_count = 0
    for tx in all_txs_client0:
        R = VerifyingKey.from_string(bytes.fromhex(tx["ephemeral_pubkey"]), curve=SECP256k1)
        if identities[0].owns_address(tx["address"], R):
            ok_count += 1
    print(f"  {ok_count}/{len(all_txs_client0)} of client 0's own addresses verified as its own.")

    print("\nCross-check: client 1 tries to claim client 0's addresses (should fail):")
    fail_count = 0
    for tx in all_txs_client0:
        R = VerifyingKey.from_string(bytes.fromhex(tx["ephemeral_pubkey"]), curve=SECP256k1)
        if not identities[1].owns_address(tx["address"], R):
            fail_count += 1
    print(f"  {fail_count}/{len(all_txs_client0)} correctly rejected by client 1.")

    # ================================================================
    print("\n" + "=" * 70)
    print("PROOF 3 — Content-layer privacy + correct aggregation")
    print("=" * 70)
    r = 3  # inspect round 3
    real = plaintext_grad_log[r - 1][0]
    masked_tx = chain.chain[r].transactions[0]["masked_update"]
    masked = np.array(masked_tx)
    print(f"Round {r}, client 0:")
    print(f"  Real local gradient (NEVER put on-chain): {np.round(real, 4)}")
    print(f"  On-chain masked value (what everyone sees): {np.round(masked, 4)}")
    print(f"  L2 distance between them: {np.linalg.norm(real - masked):.4f}  "
          f"(large -> individual value is hidden)")

    real_sum = sum(plaintext_grad_log[r - 1])
    masked_sum = sum(np.array(tx["masked_update"]) for tx in chain.chain[r].transactions)
    print(f"\n  Sum of REAL gradients this round:   {np.round(real_sum, 6)}")
    print(f"  Sum of MASKED values this round:    {np.round(masked_sum, 6)}")
    print(f"  Difference: {np.linalg.norm(real_sum - masked_sum):.2e}  "
          f"(≈0 -> aggregation is exact despite masking)")

    print("\n" + "=" * 70)
    print("PROOF 4 — Model still trains correctly end-to-end")
    print("=" * 70)
    print(f"Learned weights : {np.round(w, 3)}")
    print(f"True weights    : {np.round(TRUE_W, 3)}")
    print(f"L2 error        : {np.linalg.norm(w - TRUE_W):.4f}")


if __name__ == "__main__":
    main()
