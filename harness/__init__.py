"""Test-time harness utilities for AgentOdyssey agents.

The harness deliberately keeps game facts outside model parameters.  It can be
used with the observation/action protocol used by :mod:`eval` without knowing
the concrete generated game's world schema.
"""

from .actions import ActionValidityMask, ActionValidation
from .events import EventLog, InteractionEvent
from .ledger import Fact, StateLedger
from .core import Harness

__all__ = [
    "ActionValidityMask",
    "ActionValidation",
    "EventLog",
    "InteractionEvent",
    "Fact",
    "StateLedger",
    "Harness",
]
