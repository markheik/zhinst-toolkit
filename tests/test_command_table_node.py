import json
from array import array
from collections import OrderedDict
from unittest import mock
from unittest.mock import patch

import pytest

from zhinst.toolkit.driver.nodes.command_table_node import CommandTableNode


@pytest.fixture()
def command_table_node(shfsg):
    yield CommandTableNode(
        shfsg.root,
        ("sgchannels", "0", "awg", "commandtable"),
        device_type="shfsg",
    )


def test_attributes_init_node(command_table_node):
    assert command_table_node._device_type == "shfsg"
    assert command_table_node.raw_tree == ("sgchannels", "0", "awg", "commandtable")


def test_correct_ct_node_schema_loaded(shfsg):
    mock_json = {"test ": 123}
    with patch(
        "builtins.open", mock.mock_open(read_data=json.dumps(mock_json))
    ) as mock_open:
        ct_node = CommandTableNode(
            shfsg.root,
            ("sgchannels", "0", "awg", "commandtable"),
            device_type="shfsg",
        )
        assert ct_node.load_validation_schema() == mock_json


@pytest.mark.parametrize(
    "payload, validate",
    [
        (
            {
                "$schema": "https://json-schema.org/draft-04/schema#",
                "header": {"version": "1.1", "userString": "Test string"},
                "table": [],
            },
            True,
        ),
        (
            {
                "$schema": "https://json-schema.org/draft-04/schema#",
                "header": {"version": "1.1", "userString": "Test string"},
                "table": [],
            },
            False,
        ),
        (
            json.dumps(
                {
                    "$schema": "https://json-schema.org/draft-04/schema#",
                    "header": {"version": "1.1", "userString": "Test string"},
                    "table": [],
                }
            ),
            False,
        ),
    ],
)
def test_ct_node_upload_to_device(
    payload, validate, mock_connection, command_table_node
):
    mock_connection.return_value.set = mock.Mock(side_effect=RuntimeError)
    command_table_node.check_status = mock.Mock(return_value="")
    command_table_node.upload_to_device(payload, validate=validate)
    mock_connection.return_value.setVector.assert_called_with(
        "/dev1234/sgchannels/0/awg/commandtable/data",
        (
            '{"$schema": "https://json-schema.org/draft-04/schema#", '
            '"header": {"version": "1.1", "userString": "Test string"}, "table": []}'
        ),
    )


@pytest.fixture
def shfsg_ct_node(mock_connection, shfsg):
    schema = {
        "$schema": "https://json-schema.org/draft-04/schema#",
        "header": {"version": "1.1", "userString": "Test string"},
        "table": [],
    }
    mock_connection.return_value.get.return_value = OrderedDict(
        [
            (
                "/dev1234/sgchannels/0/awg/commandtable/data",
                {
                    "timestamp": array("q", [3880906635]),
                    "value": [schema],
                },
            )
        ]
    )
    return CommandTableNode(
        shfsg.root,
        ("sgchannels", "0", "awg", "commandtable"),
        device_type="shfsg",
    )


def test_ct_node_load_from_device_correct_data(shfsg_ct_node, mock_connection):
    ct = shfsg_ct_node.load_from_device()
    assert ct.as_dict() == {
        "$schema": "https://json-schema.org/draft-04/schema#",
        "header": {"version": "1.1", "userString": "Test string"},
        "table": [],
    }


def test_ct_node_load_from_device_correct_node(shfsg_ct_node, mock_connection):
    shfsg_ct_node.load_from_device()
    mock_connection.return_value.get.assert_called_with(
        "/dev1234/sgchannels/0/awg/commandtable/data", settingsonly=False, flat=True
    )


def test_ct_node_ct_not_supported(command_table_node):
    command_table_node._device_type = "shfasdddsssss132"
    with pytest.raises(KeyError) as e_info:
        command_table_node.load_validation_schema()


def test_ct_node_check_status_3bit_up(command_table_node, mock_connection):
    mock_connection.return_value.getInt.return_value = 8
    with pytest.raises(RuntimeError):
        command_table_node.check_status()


def test_ct_node_check_status_1bit_up(command_table_node, mock_connection):
    mock_connection.return_value.getInt.return_value = 1
    assert command_table_node.check_status() is True

    mock_connection.return_value.getInt.return_value = 2
    assert command_table_node.check_status() is False


def test_ct_node_status_called(command_table_node, mock_connection):
    mock_connection.return_value.getInt.return_value = 1
    command_table_node.check_status()
    mock_connection.return_value.getInt.assert_called_with(
        "/dev1234/sgchannels/0/awg/commandtable/status"
    )
