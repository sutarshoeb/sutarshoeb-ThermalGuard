# Final Assets

This folder contains the stronger imported export from the merged Colab/download package.

It is organized so the project has one local source of truth for:

- the final dataset archive used in the later training stage
- the notebook associated with the stronger training environment
- the final benchmark-ready result folders
- the model configuration files that belong with that export

These folders are now populated inside `final_assets/colab_export/`:

- `dataset/`
- `models/`
- `notebook/`
- `results/`

The Flask app is configured to read its comparison weights from this area.
