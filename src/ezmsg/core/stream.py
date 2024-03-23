from typing import Type, Optional

from .netprotocol import DEFAULT_SHM_SIZE
from .addressable import Addressable


class Stream(Addressable):
    """

    """

    msg_type: Type

    def __init__(self, msg_type: Type):
        super().__init__()
        self.msg_type = msg_type

    def __repr__(self) -> str:
        _addr = self.address if self._location is not None else "unlocated"
        return f"Stream:{_addr}[{self.msg_type}]"


class InputStream(Stream):
    """
    Can be added to any ``Component`` as a member variable. Methods may subscribe to it.
    """
    def __repr__(self) -> str:
        return f"Input{super().__repr__()}()"


class OutputStream(Stream):
    """
    Can be added to any ``Component`` as a member variable. Methods may publish to it.
    """
    host: Optional[str]
    port: Optional[int]
    num_buffers: int
    buf_size: int
    force_tcp: bool

    def __init__(
        self,
        msg_type: Type,
        host: Optional[str] = None,
        port: Optional[int] = None,
        num_buffers: int = 32,
        buf_size: int = DEFAULT_SHM_SIZE,
        force_tcp: bool = False,
    ) -> None:
        super().__init__(msg_type)
        self.host = host
        self.port = port
        self.num_buffers = num_buffers
        self.buf_size = buf_size
        self.force_tcp = force_tcp

    def __repr__(self) -> str:
        preamble = f"Output{super().__repr__()}"
        return f"{preamble}({self.num_buffers=}, {self.force_tcp=})"
