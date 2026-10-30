import torch

from src.export.heads import ExportWrapper
from src.models.small.small_model import SmallModel


def test_export_wrapper_head_selection():
    torch.manual_seed(0)
    model = SmallModel().eval()
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
