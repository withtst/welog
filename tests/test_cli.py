import sys
from types import SimpleNamespace

from welog.cli import main, make_parser


def test_gui_command_dispatches_to_gui_instead_of_silently_returning(monkeypatch, tmp_path):
    calls = []

    def launch(output, **kwargs):
        calls.append((output, kwargs))
        return 17

    monkeypatch.setitem(sys.modules, "welog.gui", SimpleNamespace(launch_gui=launch))
    assert main(["gui", "--no-auto-pick", "--max-frames", "12", "--output-root", str(tmp_path)]) == 17
    assert calls == [(tmp_path, {"auto_pick": False, "dwell": 1.5, "countdown": 3, "max_frames": 12})]


def test_capture_cli_accepts_explicit_historical_direction():
    args = make_parser().parse_args(["capture", "--direction", "up", "--output", "unused"])
    assert args.direction == "up"
    assert args.backend=='background'


def test_background_activate_conflict_is_rejected_before_window_access(monkeypatch):
    import pytest
    from welog import cli,windows
    def forbidden():raise AssertionError('Invalid options must not query desktop windows')
    monkeypatch.setattr(windows,'WindowsDesktop',forbidden)
    args=make_parser().parse_args(['capture','--roi','0,0,320,240','--output','unused','--activate'])
    with pytest.raises(ValueError,match='incompatible'):cli.capture(args)


def test_debug_retention_is_explicit_and_quota_is_configurable():
    parser=make_parser()
    assert not parser.parse_args(['capture','--output','unused']).debug_save_images
    args=parser.parse_args(['capture','--output','unused','--debug-save-images','--debug-max-mb','64'])
    assert args.debug_save_images and args.debug_max_mb==64


def test_replay_cli_defaults_to_recorded_capture_direction():
    args = make_parser().parse_args(["replay", "archive", "--output", "output"])
    assert args.direction == "auto"


def test_cli_defaults_to_full_capture_with_optional_frame_limit():
    parser=make_parser()
    assert parser.parse_args(['gui']).max_frames==0
    assert parser.parse_args(['capture','--output','unused']).max_frames==0
    assert parser.parse_args(['capture','--output','unused','--max-frames','50']).max_frames==50
