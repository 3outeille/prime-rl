# Install Log — environment fixes

Changes needed to get the `uv` env working on this machine.

## Machine

- Ubuntu 20.04, **GLIBC 2.31** (`ldd --version`)

Several prebuilt wheels in the disagg/KV-transfer stack are built against newer
glibc and can't load here. vLLM/nixl import them lazily, so they only bite when
the relevant feature is actually used — but `deep-ep` crashed plain inference
because vLLM imports it just for being present (`has_deep_ep()` = `find_spec`).

## deep-ep / deep-gemm (GLIBC 2.32) — fixed in pyproject.toml

Symptom (at inference startup):

```
ImportError: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.32' not found
(required by .../deep_ep_cpp.cpython-312-x86_64-linux-gnu.so)
```

Fix: removed `deep-ep` and `deep-gemm` from the `disagg` extra in `pyproject.toml`
(see the comment there). `uv sync` then drops them from `uv.lock` and the venv.
They only matter for disaggregated serving / FP8 MoE kernels — not needed here.

> Note: this diverges from upstream, which gates these on `platform_machine` only
> (not glibc). The wheels are valid on x86_64 hosts with GLIBC >= 2.32. Don't
> upstream this removal; it's a local-machine constraint.

## mooncake-transfer-engine (GLIBC 2.34) — pinned, not yet a problem

`mooncake/engine.so` requires **GLIBC 2.34** and fails to import:

```
ImportError: ... version `GLIBC_2.34' not found (required by .../mooncake/engine.so)
```

It stays installed (pinned to `0.3.8` in `[tool.uv] override-dependencies`, and
pulled in transitively by nixl/vLLM). It's imported only during disaggregated KV
transfer, so normal inference/training is unaffected. If disagg is ever needed on
this box, mooncake will have to be built from source against GLIBC 2.31 or the
machine upgraded.

## Result

- `deep-ep`, `deep-gemm`: removed (pyproject + lock + venv)
- `mooncake-transfer-engine==0.3.8`: installed but unused at runtime
- `uv run inference @ configs/debug/infer.toml` works.
