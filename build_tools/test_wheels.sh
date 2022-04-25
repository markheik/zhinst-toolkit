#!/bin/bash
wheelPath=$(find dist -type f -name \*.whl)
pip install --upgrade --force-reinstall $wheelPath
python3 -c "import zhinst.toolkit; from zhinst.toolkit import __version__; print(__version__); assert __version__ == $1"
