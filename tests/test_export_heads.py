import torch

from src.export.heads import ExportWrapper, search_q_weight
from src.models.encoder import EncoderModel
from src.train.config import RampConfig, load_config


def test_export_wrapper_head_selection():
    torch.manual_seed(0)
    model = EncoderModel().eval()
    x = torch.rand(4, 8, 8, 111)
    p, v, vq, mlh, mlh_q = model(x, return_all=True)

    out = ExportWrapper(model, "search")(x)
    torch.testing.assert_close(out[1], torch.softmax(vq, dim=-1))
    torch.testing.assert_close(out[3], mlh_q)

    out = ExportWrapper(model, "outcome")(x)
    torch.testing.assert_close(out[0], p)
    torch.testing.assert_close(out[1], torch.softmax(v, dim=-1))
    torch.testing.assert_close(out[2], mlh)
    torch.testing.assert_close(out[3], mlh)

    # Value-only variants keep the outcome-trained moves-left head.
    out = ExportWrapper(model, "q_value")(x)
    torch.testing.assert_close(out[1], torch.softmax(vq, dim=-1))
    torch.testing.assert_close(out[3], mlh)

    out = ExportWrapper(model, "blend")(x)
    torch.testing.assert_close(out[1], 0.5 * (torch.softmax(v, dim=-1) + torch.softmax(vq, dim=-1)))
    torch.testing.assert_close(out[3], mlh)

    out = ExportWrapper(model, "blend", q_weight=0.25)(x)
    torch.testing.assert_close(
        out[1], 0.75 * torch.softmax(v, dim=-1) + 0.25 * torch.softmax(vq, dim=-1)
    )


def test_search_q_weight_follows_the_ramp():
    ramp = {"start": 0.0, "end": 1.0, "start_step": 1000, "end_step": 3000}
    config = {"search_heads": "blend", "search_q_weight": 0.5, "search_q_ramp": ramp}
    assert search_q_weight({"config": config, "step": 500}) == 0.0
    assert search_q_weight({"config": config, "step": 2500}) == 0.75
    assert search_q_weight({"config": config, "step": 9000}) == 1.0
    # Without a ramp: the constant weight (0.5 when the checkpoint has none).
    config["search_q_ramp"] = {}
    assert search_q_weight({"config": config, "step": 2500}) == 0.5
    assert search_q_weight({"config": {"search_heads": "blend"}}) == 0.5


def test_ramp_config():
    ramp = RampConfig(start=8.0, end=0.0, start_step=100, end_step=300)
    assert ramp.on and ramp.at(0) == 8.0 and ramp.at(200) == 4.0 and ramp.at(10_000) == 0.0
    assert not RampConfig().on


def test_ramps_load_from_yaml(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        "search_heads: blend\n"
        "search_q_ramp: {start: 0.0, end: 1.0, start_step: 10, end_step: 20}\n"
        "selfplay:\n  random_opening_plies: 8\n"
        "  random_opening_ramp: {start: 8, end: 0, start_step: 10, end_step: 20}\n"
    )
    cfg = load_config(path)
    assert cfg.search_q_ramp.at(15) == 0.5
    assert cfg.selfplay.random_opening_ramp.at(15) == 4.0
    assert cfg.to_dict()["search_q_ramp"]["end_step"] == 20
