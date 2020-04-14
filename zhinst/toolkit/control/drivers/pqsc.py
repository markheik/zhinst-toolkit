# Copyright (C) 2020 Zurich Instruments
#
# This software may be modified and distributed under the terms
# of the MIT license. See the LICENSE file for details.

import numpy as np

from zhinst.toolkit.control.drivers.base import BaseInstrument
from zhinst.toolkit.interface import DeviceTypes


class PQSC(BaseInstrument):
    """High-level driver for the Zurich Instruments PQSC.
    
    Typical Usage:
        >>>import zhinst.toolkit as tk
        >>>pqsc = tk.PQSC("pqsc", "dev1234")
        >>>pqsc.setup()
        >>>pqsc.connect_device()
        >>>pqsc.nodetree
        >>>...
    
    Arguments:
        name (str): Identifier for the PQSC.
        serial (str): Serial number of the device, e.g. 'dev1234'. The serial 
            number can be found on the back panel of the instrument.

    """

    def __init__(self, name, serial, **kwargs):
        super().__init__(name, DeviceTypes.PQSC, serial, **kwargs)
