# Checkpoints

Model checkpoints are not committed to Git because they are large binary training artifacts.

For v17 inference, place the trained checkpoint here:

```text
checkpoints/satellite_constrained_v17_server/best.pth
```

Recommended distribution options:

- GitHub Releases for a single downloadable checkpoint.
- Git LFS if you intentionally want weights tracked by Git.
- External storage such as Google Drive, OneDrive, or Hugging Face Hub.

The repository code expects the checkpoint path configured in `app.py` or passed explicitly at runtime.
