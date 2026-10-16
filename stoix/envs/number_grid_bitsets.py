"""Static-width uint32 spell sets, including maps with more than 32 spells.

Small scenarios keep a scalar; larger catalogues add a final word axis. All
runtime operations remain JAX; Python integers are used only to build tables.
"""
import jax.numpy as jnp


class PackedBits:
    def __init__(self, count):
        self.words = max(1, (count + 31) // 32)
        self.shape = () if self.words == 1 else (self.words,)
        self.bits = self.masks([1 << i for i in range(count)])

    def masks(self, values):
        if not self.shape:
            return jnp.array(values, jnp.uint32)
        scalar = isinstance(values, int)
        rows = [values] if scalar else values
        result = jnp.array([[(v >> (32 * w)) & 0xffffffff for w in range(self.words)] for v in rows], jnp.uint32)
        return result[0] if scalar else result.reshape(len(rows), self.words)

    def any(self, values):
        present = values != 0
        return jnp.any(present, axis=-1) if self.shape else present

    def contains(self, values, masks):
        return self.any(values & masks)

    def empty(self, *leading):
        return jnp.zeros(leading + self.shape, jnp.uint32)
