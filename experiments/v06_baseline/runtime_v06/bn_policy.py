"""Match BN running-stat retention per optimizer group, not physical-batch statistics."""
import math

def momentum_for_group(original, group):
    assert 0 < original < 1 and group >= 1
    return 1 - (1 - original) ** (1 / group)

def apply_group(rows, group):
    for name, module, original in rows:
        module.momentum = momentum_for_group(original, group)

def self_test():
    import torch
    for original in (0.02, 0.1):
        for group in (1, 2, 3, 4):
            current = momentum_for_group(original, group)
            assert math.isclose((1-current)**group, 1-original, abs_tol=1e-14)
            a = torch.nn.BatchNorm1d(3, momentum=original).double()
            b = torch.nn.BatchNorm1d(3, momentum=current).double()
            x = torch.arange(24, dtype=torch.float64).reshape(8,3) / 11
            y = x.clone().requires_grad_()
            z = x.clone().requires_grad_()
            ya,yb = a(y),b(z)
            torch.testing.assert_close(ya,yb,rtol=0,atol=0)
            (ya.square().sum()).backward(); (yb.square().sum()).backward()
            torch.testing.assert_close(y.grad,z.grad,rtol=0,atol=0)
            for _ in range(group-1):b(x)
            torch.testing.assert_close(a.running_mean,b.running_mean,rtol=1e-12,atol=1e-12)
            torch.testing.assert_close(a.running_var,b.running_var,rtol=1e-12,atol=1e-12)
    print('PASS: BN retention full/tail groups; unchanged train outputs/gradients; repeated-identical-batch running statistics')
    print('NOTE: This does not reproduce global-batch variance or BN train normalization.')

if __name__=='__main__': self_test()
