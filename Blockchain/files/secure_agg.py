"""
secure_agg.py
Content-layer privacy: pairwise additive masking in the style of
Bonawitz et al.'s Secure Aggregation. This is deliberately the
*simplified* core idea (pairwise cancellation), not the full protocol
(no dropout-recovery secret sharing) -- enough to prove the point that
was under discussion: "can hashing alone give content privacy?" No --
this module is the thing you actually need for that, and it shows the
masks cancel out on summation while any single client's masked value
reveals nothing about its real update.

Each client pair (i, j) derives a shared secret via ECDH on a separate
"aggregation keypair" (kept distinct from the stealth identity keys).
The shared secret seeds a PRG that expands into a mask vector for that
round. Client i (i<j) adds the pairwise mask, client j subtracts it --
since the shared secret is symmetric (ECDH), the two masks are
identical and cancel exactly when the server sums all masked updates.
"""

import hashlib
import numpy as np
from ecdsa import SigningKey, SECP256k1

CURVE = SECP256k1


class AggKeypair:
    """Separate keypair used only for pairwise mask agreement (not identity)."""

    def __init__(self):
        self.sk = SigningKey.generate(curve=CURVE)
        self.pk = self.sk.verifying_key


def _ecdh_shared_secret_int(my_sk: SigningKey, their_pk) -> int:
    point = their_pk.pubkey.point * my_sk.privkey.secret_multiplier
    return int.from_bytes(
        hashlib.sha256(point.x().to_bytes(32, "big")).digest(), "big"
    )


def _prg_vector(seed_int: int, length: int, round_num: int) -> np.ndarray:
    """Deterministically expand a seed + round number into a pseudorandom
    vector in [-1, 1] using SHA-256 in counter mode. Same seed + round
    always gives the same vector (needed so both parties in a pair
    derive an identical mask independently)."""
    seed_bytes = seed_int.to_bytes(32, "big", signed=False) if seed_int.bit_length() <= 256 \
        else hashlib.sha256(seed_int.to_bytes((seed_int.bit_length() + 7) // 8, "big")).digest()
    out = []
    counter = 0
    while len(out) < length:
        h = hashlib.sha256(seed_bytes + round_num.to_bytes(4, "big") + counter.to_bytes(4, "big")).digest()
        val = int.from_bytes(h[:8], "big") / (2 ** 64) * 2 - 1
        out.append(val)
        counter += 1
    return np.array(out[:length])


def mask_update(update_vec, my_pos, agg_keys, round_num):
    """
    update_vec: this client's real local gradient (np.ndarray)
    my_pos:     this client's index in the client list
    agg_keys:   list of AggKeypair, one per client (agg_keys[my_pos] is mine)
    round_num:  current FL round

    Returns the masked update to actually submit to the chain.
    """
    masked = update_vec.copy()
    my_sk = agg_keys[my_pos].sk
    for j, other in enumerate(agg_keys):
        if j == my_pos:
            continue
        shared = _ecdh_shared_secret_int(my_sk, other.pk)
        pairwise_mask = _prg_vector(shared, len(update_vec), round_num)
        if my_pos < j:
            masked = masked + pairwise_mask
        else:
            masked = masked - pairwise_mask
    return masked
