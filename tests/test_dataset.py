"""Tests for DreamCatcher filename parsing and the grouped split.

The parsing case here is not hypothetical: 3,000 real files named
``04_new-...`` fell through the old pattern and were silently lumped into one
anonymous group that ended up as 55% of the validation set.
"""

from __future__ import annotations

from src.train.dataset import CLASS_BRUXISM, grouped_split, parse_session


class TestParseSession:
    def test_plain_name(self):
        assert parse_session("04-1-7-user_12.5_22.5.wav") == ("04-1-7", "04")

    def test_tagged_name_keeps_tag_in_session(self):
        """04_new-1-1 is a different recording from 04-1-1."""
        session, _ = parse_session("04_new-1-1-user_1034.88_1044.88.wav")
        assert session == "04_new-1-1"
        assert session != parse_session("04-1-1-user_1.wav")[0]

    def test_tagged_name_maps_to_the_same_subject(self):
        """Otherwise subject 04 could sit on both sides of a subject split."""
        assert parse_session("04_new-2-3-user_1.wav")[1] == "04"

    def test_unrecognized_name_is_its_own_group(self):
        assert parse_session("something_else.wav") == ("", "")


class TestGroupedSplit:
    """``grouped_split`` returns ``(train_groups, val_groups)`` as name sets."""

    def _data(self):
        groups = ["a"] * 10 + ["b"] * 10 + ["c"] * 10 + ["d"] * 10
        labels = ([CLASS_BRUXISM] + ["other"] * 9) * 4
        return groups, labels

    def test_no_group_lands_on_both_sides(self):
        groups, labels = self._data()
        train, val = grouped_split(groups, labels, val_fraction=0.25)
        assert not train & val

    def test_every_group_is_assigned(self):
        groups, labels = self._data()
        train, val = grouped_split(groups, labels, val_fraction=0.25)
        assert train | val == set(groups)

    def test_both_sides_get_some_of_the_rare_class(self):
        groups, labels = self._data()
        train, val = grouped_split(groups, labels, val_fraction=0.25)
        assert train and val
