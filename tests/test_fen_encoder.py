from src.models.small.fen_encoder import fen_to_tensor

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_startpos_shape():
    t = fen_to_tensor(START_FEN)
    assert t.shape == (8, 8, 19)


def test_white_to_move_plane():
    t = fen_to_tensor(START_FEN)
    # plane 12 is the side-to-move plane; should be all 1.0 for white-to-move
    assert (t[:, :, 12] == 1.0).all()
