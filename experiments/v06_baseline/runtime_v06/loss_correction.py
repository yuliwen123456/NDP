"""Compensate mixed loss reductions under Lightning automatic accumulation."""
SUM_PREFIXES = ('loss_mask', 'loss_dice', 'loss_box')

def group_factors(batch_index, num_batches, accumulation):
    start = (batch_index // accumulation) * accumulation
    actual = min(accumulation, num_batches - start)
    if actual < 1:
        raise ValueError('Invalid accumulation group')
    return actual, accumulation / actual

def correct_criterion(losses, group_size):
    return {key: value * group_size if any(
        key == prefix or key.startswith(prefix + '_') for prefix in SUM_PREFIXES
    ) else value for key, value in losses.items()}

def self_test():
    import torch
    # Compare gradients to physical-batch objectives, including partial groups.
    for batches in (1, 2, 3, 4, 5, 449):
        for start in range(0, batches, 4):
            r = min(4, batches - start)
            w = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
            aggregate = 0
            expected = 0
            for i in range(r):
                sums = (w * (i + 1)).square()
                mean = (w - (i + 2)).square()
                group, factor = group_factors(start + i, batches, 4)
                fixed = correct_criterion({'loss_mask': sums, 'loss_dice_11': sums,
                                           'loss_box': sums, 'loss_ce': mean,
                                           'loss_ood': mean}, group)
                aggregate = aggregate + sum(fixed.values()) * factor / 4
                expected = expected + 3 * sums + 2 * mean / r
            ga = torch.autograd.grad(aggregate, w, retain_graph=True)[0]
            ge = torch.autograd.grad(expected, w)[0]
            torch.testing.assert_close(aggregate, expected)
            torch.testing.assert_close(ga, ge)
    print('PASS: values and gradients; full/partial groups; auxiliary losses; 449-batch epoch')

if __name__ == '__main__':
    self_test()
