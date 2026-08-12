"""MOT17 parsing, verification and result writing.

Built against synthetic sequences rather than the real 5.6 GB download, so
these run in milliseconds and pass on a machine that has never seen the
dataset. What they check is the handling that makes a benchmark number
match somebody else's: the ignore-class filter, the 1-indexed frame
convention, and the output format TrackEval expects.
"""
from __future__ import annotations

import pytest

from humandetect import mot17


def write_sequence(
    root,
    name="MOT17-02-FRCNN",
    *,
    split="train",
    length=3,
    gt_lines=None,
    width=1920,
    height=1080,
):
    """Create a minimal but structurally valid sequence folder."""
    seq = root / split / name
    (seq / "img1").mkdir(parents=True)
    (seq / "det").mkdir(parents=True)

    for frame in range(1, length + 1):
        (seq / "img1" / f"{frame:06d}.jpg").write_bytes(b"not-a-real-jpeg")
    (seq / "det" / "det.txt").write_text("1,-1,10,10,20,40,0.9,-1,-1,-1\n")

    (seq / "seqinfo.ini").write_text(
        "[Sequence]\n"
        f"name={name}\n"
        "imDir=img1\n"
        "frameRate=30\n"
        f"seqLength={length}\n"
        f"imWidth={width}\n"
        f"imHeight={height}\n"
        "imExt=.jpg\n"
    )

    if gt_lines is not None:
        (seq / "gt").mkdir(parents=True)
        (seq / "gt" / "gt.txt").write_text("\n".join(gt_lines) + "\n")
    return seq


# --------------------------------------------------------------------------
# Name resolution
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "given,expected",
    [
        ("02", "MOT17-02-FRCNN"),
        ("MOT17-02", "MOT17-02-FRCNN"),
        ("MOT17-02-FRCNN", "MOT17-02-FRCNN"),
        ("MOT17-02-DPM", "MOT17-02-DPM"),
        ("MOT17-02-SDP", "MOT17-02-SDP"),
    ],
)
def test_resolve_accepts_every_spelling(tmp_path, given, expected):
    assert mot17.resolve(given, root=tmp_path).name == expected


def test_train_sequences_are_the_seven_scenes(tmp_path):
    names = [p.name for p in mot17.train_sequences(root=tmp_path)]
    assert len(names) == 7
    assert names == [f"MOT17-{s}-FRCNN" for s in mot17.TRAIN_SCENES]


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------
def test_only_class_one_with_conf_one_is_evaluated(tmp_path):
    """The ignore-class filter: get this wrong and false positives inflate."""
    seq = write_sequence(tmp_path, gt_lines=[
        "1,1,10,10,20,40,1,1,1.0",    # pedestrian, counted
        "1,2,50,10,20,40,0,1,1.0",    # conf flag 0, ignored
        "1,3,90,10,20,40,1,7,1.0",    # static person, ignored
        "1,4,130,10,20,40,1,12,1.0",  # reflection, ignored
        "2,1,12,10,20,40,1,1,0.4",    # same identity, next frame
    ])
    boxes = mot17.read_gt(seq)
    assert len(boxes) == 5
    evaluated = [b for b in boxes if b.is_evaluated]
    assert len(evaluated) == 2
    assert {b.track_id for b in evaluated} == {1}


def test_gt_columns_are_read_in_order(tmp_path):
    seq = write_sequence(tmp_path, gt_lines=["7,3,912,484,97,109,1,1,0.86"])
    box = mot17.read_gt(seq)[0]
    assert (box.frame, box.track_id) == (7, 3)
    assert (box.left, box.top, box.width, box.height) == (912, 484, 97, 109)
    assert (box.conf, box.class_id, box.visibility) == (1, 1, 0.86)


def test_short_gt_row_is_rejected(tmp_path):
    seq = write_sequence(tmp_path, gt_lines=["1,1,10,10,20,40"])
    with pytest.raises(mot17.MOT17Error, match="expected 9 columns"):
        mot17.read_gt(seq)


def test_missing_gt_is_named_clearly(tmp_path):
    seq = write_sequence(tmp_path)  # no gt written
    with pytest.raises(mot17.MOT17Error, match="gt.txt missing"):
        mot17.read_gt(seq)


# --------------------------------------------------------------------------
# Sequence verification
# --------------------------------------------------------------------------
def test_valid_sequence_reports_its_counts(tmp_path):
    seq = write_sequence(tmp_path, gt_lines=[
        "1,1,10,10,20,40,1,1,1.0",
        "1,2,50,10,20,40,1,3,1.0",   # a car
        "2,1,12,10,20,40,1,1,1.0",
    ])
    stats = mot17.verify_sequence(seq, expect_gt=True)
    assert stats.length == 3
    assert stats.resolution == "1920x1080"
    assert (stats.gt_rows, stats.gt_evaluated, stats.gt_ids) == (3, 2, 1)
    assert stats.class_counts[1] == 2
    assert stats.class_counts[3] == 1


def test_frame_count_must_match_seqinfo(tmp_path):
    seq = write_sequence(tmp_path, length=3)
    (seq / "img1" / "000003.jpg").unlink()
    with pytest.raises(mot17.MOT17Error, match="frames on disk"):
        mot17.verify_sequence(seq, expect_gt=False)


def test_gt_frame_beyond_sequence_is_rejected(tmp_path):
    seq = write_sequence(tmp_path, length=2,
                         gt_lines=["5,1,10,10,20,40,1,1,1.0"])
    with pytest.raises(mot17.MOT17Error, match="outside 1..2"):
        mot17.verify_sequence(seq, expect_gt=True)


def test_repeated_id_within_a_frame_is_rejected(tmp_path):
    """Would make the identity assignment behind IDF1 ambiguous."""
    seq = write_sequence(tmp_path, gt_lines=[
        "1,1,10,10,20,40,1,1,1.0",
        "1,1,50,10,20,40,1,1,1.0",
    ])
    with pytest.raises(mot17.MOT17Error, match="twice in frame"):
        mot17.verify_sequence(seq, expect_gt=True)


def test_unknown_class_is_rejected(tmp_path):
    seq = write_sequence(tmp_path, gt_lines=["1,1,10,10,20,40,1,99,1.0"])
    with pytest.raises(mot17.MOT17Error, match="unknown gt class"):
        mot17.verify_sequence(seq, expect_gt=True)


def test_test_split_must_not_have_ground_truth(tmp_path):
    seq = write_sequence(tmp_path, name="MOT17-01-FRCNN", split="test",
                         gt_lines=["1,1,10,10,20,40,1,1,1.0"])
    with pytest.raises(mot17.MOT17Error, match="unexpectedly has gt"):
        mot17.verify_sequence(seq, expect_gt=False)


# --------------------------------------------------------------------------
# Result writing
# --------------------------------------------------------------------------
def test_results_are_written_in_motchallenge_format(tmp_path):
    out = tmp_path / "MOT17-02-FRCNN.txt"
    written = mot17.write_results(
        [(2, 7, (10, 20, 30, 40), 0.9123), (1, 7, (11, 21, 30, 40), 0.5)],
        out, seq_length=10,
    )
    assert written == 2
    lines = out.read_text().splitlines()
    # Sorted by frame, and the trailing -1s are the unused 3D fields.
    assert lines[0] == "1,7,11,21,30,40,0.5000,-1,-1,-1"
    assert lines[1] == "2,7,10,20,30,40,0.9123,-1,-1,-1"


def test_writing_a_frame_beyond_the_sequence_is_rejected(tmp_path):
    with pytest.raises(mot17.MOT17Error, match="outside 1..5"):
        mot17.write_results(
            [(6, 1, (10, 20, 30, 40), 0.9)], tmp_path / "out.txt", seq_length=5
        )


def test_writing_a_repeated_id_in_one_frame_is_rejected(tmp_path):
    with pytest.raises(mot17.MOT17Error, match="repeated within a frame"):
        mot17.write_results(
            [(1, 3, (10, 20, 30, 40), 0.9), (1, 3, (50, 20, 30, 40), 0.8)],
            tmp_path / "out.txt", seq_length=5,
        )


def test_empty_results_write_an_empty_file(tmp_path):
    """A tracker that finds nothing is scoreable, not an error."""
    out = tmp_path / "out.txt"
    assert mot17.write_results([], out, seq_length=5) == 0
    assert out.read_text() == ""
