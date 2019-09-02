from ctapipe_stage1 import Stage1Process
from ctapipe.core import ToolConfigurationError
from pytest import raises
from traitlets.config import Config


def test_no_output_options_enabled():

    config = Config(
        {"Stage1Process": {"write_images": False, "write_parameters": False}}
    )

    tool = Stage1Process(config=config)
    tool.run()

