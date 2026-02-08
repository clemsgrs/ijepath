import logging

from ijepath.utils.log_utils import TqdmLoggingHandler, setup_logging


def test_setup_logging_is_tqdm_safe_and_idempotent():
    logger_name = "ijepath.test.log_utils"
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()

    setup_logging(name=logger_name, level=logging.INFO)
    first_handlers = list(logger.handlers)
    assert any(isinstance(h, TqdmLoggingHandler) for h in first_handlers)

    setup_logging(name=logger_name, level=logging.INFO)
    second_handlers = list(logger.handlers)

    assert len(second_handlers) == len(first_handlers)
    assert sum(isinstance(h, TqdmLoggingHandler) for h in second_handlers) == 1

