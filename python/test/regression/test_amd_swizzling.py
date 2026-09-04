import pytest
import torch

import tokenspeed_triton as triton
import tokenspeed_triton.language as tl
from tokenspeed_triton._internal_testing import is_compile_warmup, is_hip_cdna4


@triton.jit
def _one_warp_bf16_dot_transposed_rhs(lhs, rhs, out):
    m = tl.arange(0, 32)
    n = tl.arange(0, 32)
    k = tl.arange(0, 128)
    lhs = tl.load(lhs + m[:, None] * 128 + k[None, :])
    rhs = tl.load(rhs + n[:, None] * 128 + k[None, :])
    result = tl.dot(lhs, tl.trans(rhs))
    tl.store(out + m[:, None] * 32 + n[None, :], result)


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires an AMD CDNA4 GPU")
@pytest.mark.enable_warmup(min_capability=9)
def test_one_warp_bf16_dot_transposed_rhs(device):
    generator = torch.Generator(device=device).manual_seed(0)
    lhs = torch.randn((32, 128), generator=generator, device=device, dtype=torch.bfloat16)
    rhs = torch.randn((32, 128), generator=generator, device=device, dtype=torch.bfloat16)
    actual = torch.empty((32, 32), device=device, dtype=torch.float32)

    _one_warp_bf16_dot_transposed_rhs[(1, )](
        lhs,
        rhs,
        actual,
        num_warps=1,
    )
    if is_compile_warmup():
        return

    expected = lhs.float() @ rhs.float().T
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=0.0)
