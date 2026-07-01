# DDColor Requirement

This project requires the official DDColor code and the `ddcolor_paper_tiny` pretrained weights.

They are not bundled in this repository. Download them separately, then either place the official DDColor repository here or point `config.yaml` to an external location.

Expected files:

```text
PATH/TO/DDColor/
  ddcolor/
  weights_hf/
    ddcolor_paper_tiny/
      pytorch_model.bin
```

Configure:

```yaml
ddcolor:
  code_path: "PATH/TO/DDColor"
  weights_path: "PATH/TO/DDColor/weights_hf/ddcolor_paper_tiny/pytorch_model.bin"
```

DDColor is used only as a frozen local model. It is not trained or fine-tuned by this project.
