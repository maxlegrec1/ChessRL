from src.rating import ladder

RATINGS = ladder.load_anchor_ratings()


def fake_play(true_rating: float, games: int = 48, log: list | None = None):
    """Deterministic match: the model wins round(p * games) games against each anchor."""

    def play(name: str) -> list[float]:
        if log is not None:
            log.append(name)
        wins = round(ladder.expected(true_rating, RATINGS[name]) * games)
        return [1.0] * wins + [0.0] * (games - wins)

    return play


def rate(true_rating: float, estimate: float, max_extra: int, log: list | None = None) -> float:
    picked = ladder.pick_anchors(estimate, RATINGS, k=2)
    scores = ladder.adaptive_anchor_scores(
        fake_play(true_rating, log=log), picked, RATINGS, max_extra=max_extra
    )
    results = [(RATINGS[n], x) for n, s in scores.items() for x in s]
    return ladder.fit_rating(results, prior=estimate)[0]


def test_original_ladder_trails_on_a_sweep():
    # The sims-400 run's real rating at step 10k: previous estimate 616, anchors greedy
    # and sf_s0_r25, no losses -> the two virtual draws at 616 decide the fit (1,082).
    results = (
        [(RATINGS["greedy"], 1.0)] * 43
        + [(RATINGS["greedy"], 0.5)] * 5
        + [(RATINGS["sf_s0_r25"], 1.0)] * 36
        + [(RATINGS["sf_s0_r25"], 0.5)] * 12
    )
    assert abs(ladder.fit_rating(results, prior=616)[0] - 1082) < 5


def test_steep_climb_steps_up_to_matched_anchors():
    # Previous estimate ~620, the network is really ~1,490 (what matched anchors measured).
    fixed, original = rate(1490, 620, max_extra=3), rate(1490, 620, max_extra=0)
    assert abs(fixed - 1490) < 120
    assert abs(fixed - 1490) < abs(original - 1490)


def test_no_extra_games_when_the_estimate_is_right():
    log: list = []
    rate(1890, 1890, max_extra=3, log=log)
    assert len(log) == 2


def test_steps_down_when_swept_by_the_weaker_anchor():
    # A collapse: the estimate says 1,900, the network now plays like ~1,100.
    assert abs(rate(1100, 1900, max_extra=3) - 1100) < 150


def test_next_anchor_skips_played_and_stops_at_the_ends():
    names = sorted(RATINGS, key=RATINGS.get)
    assert ladder.next_anchor(names[-1], RATINGS, up=True, exclude=set()) is None
    assert ladder.next_anchor(names[0], RATINGS, up=False, exclude=set()) is None
    assert ladder.next_anchor(names[0], RATINGS, up=True, exclude={names[1]}) == names[2]
