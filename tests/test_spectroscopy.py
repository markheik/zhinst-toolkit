from unittest.mock import patch

import pytest


@pytest.fixture()
def spectroscopy(shfqa):
    yield shfqa.qachannels[0].spectroscopy


@pytest.fixture()
def m_deviceutils():
    with patch(
        "zhinst.toolkit.driver.nodes.spectroscopy.deviceutils", autospec=True
    ) as deviceutils:
        yield deviceutils


def test_configure_result_logger(mock_connection, spectroscopy, m_deviceutils):
    spectroscopy.configure_result_logger(result_length=10)
    m_deviceutils.configure_result_logger_for_spectroscopy.assert_called_with(
        mock_connection.return_value,
        "DEV1234",
        0,
        result_length=10,
        num_averages=1,
        averaging_mode=0,
    )
    spectroscopy.configure_result_logger(
        result_length=0, num_averages=2, averaging_mode=1
    )
    m_deviceutils.configure_result_logger_for_spectroscopy.assert_called_with(
        mock_connection.return_value,
        "DEV1234",
        0,
        result_length=0,
        num_averages=2,
        averaging_mode=1,
    )


def test_run(mock_connection, spectroscopy, m_deviceutils):
    spectroscopy.run()
    m_deviceutils.enable_result_logger.assert_called_with(
        mock_connection.return_value,
        "DEV1234",
        0,
        mode="spectroscopy",
    )


def test_stop(mock_connection, spectroscopy):
    # already disabled
    mock_connection.return_value.getInt.return_value = 0
    spectroscopy.stop()
    mock_connection.return_value.set.assert_called_with(
        "/dev1234/qachannels/0/spectroscopy/result/enable", False
    )
    # never disabled
    mock_connection.return_value.getInt.return_value = 1
    with pytest.raises(TimeoutError) as e_info:
        spectroscopy.stop(timeout=0.5)


def test_wait_done(mock_connection, spectroscopy):
    # already disabled
    mock_connection.return_value.getInt.return_value = 0
    spectroscopy.wait_done()
    # never disabled
    mock_connection.return_value.getInt.return_value = 1
    with pytest.raises(TimeoutError) as e_info:
        spectroscopy.wait_done(timeout=0.5)


def test_read(mock_connection, spectroscopy, m_deviceutils):
    spectroscopy.read()
    m_deviceutils.get_result_logger_data.assert_called_with(
        mock_connection.return_value, "DEV1234", 0, mode="spectroscopy", timeout=10
    )
    spectroscopy.read(timeout=1)
    m_deviceutils.get_result_logger_data.assert_called_with(
        mock_connection.return_value, "DEV1234", 0, mode="spectroscopy", timeout=1
    )
