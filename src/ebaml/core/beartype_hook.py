"""Install the beartype import hook for the ``ebaml`` package.

``ebaml/__init__.py`` imports this module before any other submodule. The import hook then checks the type hints of
every module that loads after it.
"""

from beartype import BeartypeConf
from beartype.claw import beartype_package

# The numeric tower lets an int satisfy a float hint, as PEP 484 says.
# Checking annotated variables is off, because structlog's lazy logger proxy does not match its own logger type.
beartype_package("ebaml", conf=BeartypeConf(is_pep484_tower=True, claw_is_pep526=False))
