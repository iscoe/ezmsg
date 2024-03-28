import ezmsg.core as ez

from typing import AsyncGenerator, Optional, Any


class DebugLogSettings(ez.Settings):
    """
    ``Settings`` class associated with :obj:`DebugLog`

    Args:
        name: Useful name for the logger. The name is included in the logstring so that if multiple DebugLogs
            are used in one pipeline, their messages can be differentiated.
        max_length: Sets a maximum number of chars which will be printed from the message.
            If the message is longer, the log message will be truncated.
    """
    name: str = "DEBUG"
    max_length: Optional[int] = 400


class DebugLog(ez.Unit):
    """
    Logs messages that pass through.
    """

    SETTINGS: DebugLogSettings

    INPUT = ez.InputStream(Any)
    """Send messages to log here."""

    OUTPUT = ez.OutputStream(Any)
    """Send messages back out to continue through the graph."""

    @ez.subscriber(INPUT, zero_copy=True)
    @ez.publisher(OUTPUT)
    async def log(self, msg: Any) -> AsyncGenerator:
        logstr = f"{self.SETTINGS.name} - {msg=}"
        if self.SETTINGS.max_length is not None:
            if len(logstr) > self.SETTINGS.max_length:
                logstr = logstr[: self.SETTINGS.max_length]
                logstr = logstr + " ... [truncated]"
        ez.logger.info(logstr)
        yield self.OUTPUT, msg
