# Dataset

`customers.csv` is a generated synthetic dataset created by `scripts/generate_data.py`.
It represents fictional retail/subscription customers and includes a simulated `churned`
label. It does not contain real people or real company records.

To regenerate it:

```bash
python scripts/generate_data.py --rows 2500
```
