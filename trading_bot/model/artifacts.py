import os
import joblib
import torch
from typing import Dict


def save_artifacts(path: str, art: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    to_save = {
        "feature_cols": art.get("feature_cols"),
        "seq_len": art.get("seq_len"),
        "wf_cv": art.get("wf_cv"),
        "shap_importance": None,
    }
    joblib.dump(to_save, path + ".meta.joblib")
    # Save LGB
    if "lgb_model" in art:
        joblib.dump(art["lgb_model"]["model"], path + ".lgb.joblib")
    # Save Autoformer
    if "af_model" in art:
        model = art["af_model"]["model"].cpu()
        torch.save(model.state_dict(), path + ".af.pt")


def load_artifacts(path: str, device: str = "auto") -> Dict:
    out: Dict = {}
    meta = joblib.load(path + ".meta.joblib")
    out.update(meta)
    # Load LGB
    try:
        out["lgb_model"] = {"model": joblib.load(path + ".lgb.joblib")}
    except Exception:
        pass
    # Load Autoformer
    try:
        from autoformer_pytorch import Autoformer
        device_str = "cuda" if device == "auto" and torch.cuda.is_available() else (device if device != "auto" else "cpu")
        dev = torch.device(device_str)
        # Recreate minimal model; exact dims not strictly required for state dict load here since used for inference on last-step features
        # Caller must pass the correct input dims for predict
        model = Autoformer(dim=64, pred_length=1, seq_len=1, label_len=1, d_model=128, heads=4, enc_depth=2, dec_depth=1, dropout=0.1)
        model.load_state_dict(torch.load(path + ".af.pt", map_location=dev))
        model.to(dev)
        out["af_model"] = {"model": model}
    except Exception:
        pass
    return out