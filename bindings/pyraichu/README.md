# pyraichu

Python binding for **RAICHU**, the native Rust engine for hybrid (PDMP)
simulation of complex systems.

```bash
pip install pyraichu
```

```python
import pyraichu
print(pyraichu.__version__)
```

Development install (from the repository root, inside a virtualenv):

```bash
maturin develop -m bindings/pyraichu/Cargo.toml
```

See the repository documentation (`docs/`) for the engine design and the
validation contract.
