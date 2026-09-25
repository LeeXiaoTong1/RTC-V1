"""Fixed-budget balanced ordinary batches with replayable augmentation tickets."""
import hashlib
import random


class BalancedOrdinarySampler:
    def __init__(self, labels, batch_size, steps, seed):
        self.pools = [[i for i, y in enumerate(labels) if y == c] for c in (0, 1)]
        if not all(self.pools) or sum(map(len, self.pools)) != len(labels):
            raise ValueError('Ordinary sampling requires nonempty binary classes')
        if batch_size < 2 or batch_size % 2 or steps < 1:
            raise ValueError('Balanced batch must be even; steps must be positive')
        self.batch_size, self.steps, self.seed, self.epoch = batch_size, steps, seed, 1

    def __len__(self):
        return self.steps

    def set_epoch(self, epoch):
        if epoch < 1:
            raise ValueError('Epochs start at one')
        self.epoch = epoch

    def __iter__(self):
        half = self.batch_size // 2
        first = (self.epoch - 1) * self.steps * half
        orders = {}
        shuffle = random.Random(self.seed + self.epoch)
        for step in range(self.steps):
            tickets = []
            for label, pool in enumerate(self.pools):
                for slot in range(half):
                    visit = first + step * half + slot
                    cycle, position = divmod(visit, len(pool))
                    key = (label, cycle)
                    if key not in orders:
                        order = pool.copy()
                        seed = hashlib.sha256(f'{self.seed}:{label}:{cycle}'.encode()).digest()
                        random.Random(seed).shuffle(order)
                        orders = {k: v for k, v in orders.items() if k[0] != label}
                        orders[key] = order
                    # Unique draw, even when a small class repeats within a batch.
                    tickets.append((orders[key][position], 2 * visit + label))
            shuffle.shuffle(tickets)
            yield tickets
