import logging

from welog.diagnostics import configure_logging


def test_log_records_stop_reason_and_error_traceback_without_console(tmp_path):
    path = configure_logging(tmp_path, install_hooks=False)
    logger = logging.getLogger("welog.capture")
    try:
        logger.info("capture_end reason=no_scroll_motion cumulative_px=0")
        try:
            raise RuntimeError("fixture failure")
        except RuntimeError:
            logger.exception("capture_error")
        text = path.read_text(encoding="utf-8")
        assert "no_scroll_motion" in text
        assert "RuntimeError: fixture failure" in text
        assert "Traceback" in text
    finally:
        parent = logging.getLogger("welog")
        for handler in list(parent.handlers):
            if getattr(handler, "_welog_handler", False):
                parent.removeHandler(handler)
                handler.close()
