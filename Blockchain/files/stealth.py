"""
stealth.py
Identity-layer privacy: rotating pseudonyms via a real DKSAP-style
stealth-address construction on secp256k1 (the same curve Bitcoin/
Ethereum use). This is the "hashing-based pseudonym rotation" from
point 1 of the research discussion.

Construction (per the primer):
  Setup:   client holds (spend_sk, spend_pk=B) and (view_sk=a, view_pk=A),
           never published on-chain.
  Round t: sample ephemeral r, R = r*G
           s = H(r * A)              (sender side, using recipient's view pk)
           P_t = s*G + B             (one-time address for this round)
  Self-check: client recomputes s' = H(a * R) = H(r * A) (ECDH symmetry)
              and confirms P_t belongs to it.

Because P_t is derived from a fresh random r every round, an outside
observer sees only unrelated-looking addresses -- there is no public
computation that links P_1, P_2, P_3 to the same client.
"""

import hashlib
from ecdsa import SigningKey, SECP256k1

CURVE = SECP256k1
G = CURVE.generator
ORDER = CURVE.order


def _sha256_int(data: bytes) -> int:
    return int.from_bytes(hashlib.sha256(data).digest(), "big") % ORDER


def _point_to_hex_address(point) -> str:
    raw = point.x().to_bytes(32, "big") + point.y().to_bytes(32, "big")
    return hashlib.sha256(raw).hexdigest()


class ClientIdentity:
    """Long-term secret identity for one FL client. Never posted on-chain."""

    def __init__(self, client_id: str):
        self.client_id = client_id  # local bookkeeping only, for this demo's ground truth
        self.spend_sk = SigningKey.generate(curve=CURVE)
        self.view_sk = SigningKey.generate(curve=CURVE)
        self.spend_pk = self.spend_sk.verifying_key
        self.view_pk = self.view_sk.verifying_key

    def derive_stealth_address(self):
        """
        Generate this round's one-time pseudonymous address.
        Returns (address_hex, R) where R is the ephemeral pubkey to publish
        alongside the commit (needed by the client to later recognize the
        address; meaningless to anyone else).
        """
        r_sk = SigningKey.generate(curve=CURVE)
        R = r_sk.verifying_key

        shared_point = self.view_pk.pubkey.point * r_sk.privkey.secret_multiplier
        s = _sha256_int(shared_point.x().to_bytes(32, "big"))

        P_point = (s * G) + self.spend_pk.pubkey.point
        address_hex = _point_to_hex_address(P_point)
        return address_hex, R

    def owns_address(self, address_hex: str, R) -> bool:
        """Client-side check: is `address_hex` (with published ephemeral R) mine?"""
        shared_point = R.pubkey.point * self.view_sk.privkey.secret_multiplier
        s = _sha256_int(shared_point.x().to_bytes(32, "big"))
        P_point = (s * G) + self.spend_pk.pubkey.point
        return _point_to_hex_address(P_point) == address_hex
