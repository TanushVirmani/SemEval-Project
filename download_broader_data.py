"""Download CodeMirage for the academic/demo track (not SemEval compliant)."""

import json
import model_utils
from huggingface_hub import HfApi, hf_hub_download

if __name__ == "__main__":
    repo = "HanxiGuo/CodeMirage"
    revision = HfApi().dataset_info(repo).sha
    root = model_utils.BASE_DIR / "data/codemirage"
    root.mkdir(parents=True, exist_ok=True)
    for split in ["train", "test"]:
        print("Downloading", split, flush=True)
        path = hf_hub_download(
            repo, filename=f"{split}.csv", repo_type="dataset", revision=revision
        )
        (root / f"{split}.path").write_text(path)
        print(path, flush=True)
    (root / "provenance.json").write_text(
        json.dumps(
            {
                "dataset": repo,
                "revision": revision,
                "license": "cc-by-nc-nd-4.0",
                "source": "https://huggingface.co/datasets/HanxiGuo/CodeMirage",
                "competition_compliant": False,
            },
            indent=2,
        )
    )
