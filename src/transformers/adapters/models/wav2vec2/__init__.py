from typing import TYPE_CHECKING

from ....utils import _LazyModule


_import_structure = {
    "adapter_model": [
        "Wav2Vec2AdapterModel",
    ]
}


if TYPE_CHECKING:
    from .adapter_model import Wav2Vec2AdapterModel

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
    )