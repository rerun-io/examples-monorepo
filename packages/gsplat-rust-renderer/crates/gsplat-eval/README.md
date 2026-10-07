# gsplat-eval

```bash
target/release/gsplat-eval dirs --render renders --gt ground-truth \
  --convention brush --lpips --out metrics.json
```

Pair directories by identical relative PNG paths. `--convention published` selects
the white-background checkpoint convention; `brush` is the default. See
[architecture](../../docs/architecture.md) for image boundaries and provenance.
