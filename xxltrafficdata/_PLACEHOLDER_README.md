# Data placeholders

The `pems03/ ... pems12/` directories and `preprocessed/` are intentionally empty
in this repository. Processed traffic flow data and adjacency files will be
released via a cloud-disk link (TODO: paste link here).

After downloading, place the contents so the layout matches:

```
xxltrafficdata/
├── pems03/<year>.npz           ← raw / yearly flow tensors
├── pems03/<year>_adj.npz       ← yearly adjacency
├── pems04/...
└── preprocessed/               ← intermediate EAC-format outputs
```

The notebooks (`pemsXX_yearly_nodes.ipynb`, `pemsXX_build_eac_data.ipynb`)
regenerate these files end-to-end from raw PEMS dumps.
