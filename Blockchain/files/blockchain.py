"""
blockchain.py
Minimal but structurally real blockchain: hash-chained blocks, simple
proof-of-work, and tamper detection. This plays the role of the "chain
security" layer discussed in the primer (Section: hash chaining, Merkle
integrity, immutability).

Each block's transactions are the FL commits for one round: a list of
dicts {address, ephemeral_pubkey, masked_update, round}.
"""

import hashlib
import json
import time


class Block:
    def __init__(self, index, transactions, previous_hash, nonce=0, timestamp=None):
        self.index = index
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.transactions = transactions
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.hash = self.compute_hash()

    def compute_hash(self):
        block_string = json.dumps(
            {
                "index": self.index,
                "timestamp": self.timestamp,
                "transactions": self.transactions,
                "previous_hash": self.previous_hash,
                "nonce": self.nonce,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(block_string.encode()).hexdigest()


class Blockchain:
    difficulty = 3  # number of leading hex zeros required (kept low for demo speed)

    def __init__(self):
        self.chain = []
        self._create_genesis_block()

    def _create_genesis_block(self):
        genesis = Block(0, [], "0" * 64)
        self.chain.append(genesis)

    @property
    def last_block(self):
        return self.chain[-1]

    def _proof_of_work(self, block):
        block.nonce = 0
        computed_hash = block.compute_hash()
        target = "0" * self.difficulty
        while not computed_hash.startswith(target):
            block.nonce += 1
            computed_hash = block.compute_hash()
        return computed_hash

    def add_block(self, transactions):
        new_block = Block(len(self.chain), transactions, self.last_block.hash)
        new_block.hash = self._proof_of_work(new_block)
        self.chain.append(new_block)
        return new_block

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]
            if current.hash != current.compute_hash():
                return False, f"Block {i}: data tampered (stored hash != recomputed hash)"
            if current.previous_hash != previous.hash:
                return False, f"Block {i}: broken link to block {i - 1}"
            if not current.hash.startswith("0" * self.difficulty):
                return False, f"Block {i}: invalid proof-of-work"
        return True, "Chain valid — all blocks intact and correctly linked"

    def print_chain_summary(self):
        for b in self.chain:
            print(
                f"  Block {b.index} | nonce={b.nonce} | hash={b.hash[:16]}... | "
                f"prev={b.previous_hash[:16]}... | {len(b.transactions)} tx"
            )
