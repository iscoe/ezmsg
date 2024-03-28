import asyncio
import json
import typing
import time

from dataclasses import field, dataclass, replace
from pathlib import Path

import ezmsg.core as ez

from .messagecodec import MessageDecoder, LogStart


@dataclass
class ReplayStatusMessage:
    """
    Message which gives the status of a file replay.

    Args:
        filename: The name of the file currently being replayed.
        idx: The line number of the message that was just published.
        total: Number of messages in the file.
        done: Whether the file has finished replaying.
    """
    filename: Path
    idx: int
    total: int
    done: bool = False


@dataclass
class FileReplayMessage:
    """
    Add a file to the queue.

    Args:
        filename: The path of the file to replay.
        rate: in Hertz at which the messages will be published.
            0 = realtime (if timestamps in file)
            If not specified, messages will publish as fast as possible.
    """
    filename: typing.Optional[Path] = None
    rate: typing.Optional[float] = None  # Hz


class MessageReplaySettings(ez.Settings, FileReplayMessage):
    """
    Settings for :obj:`MesssageReplay` Unit.

    Args:
        progress: will use tqdm to indicate progress through the file. tqdm must be installed.
    """
    progress: bool = False


class MessageReplayState(ez.State):
    replay_files: "asyncio.Queue[FileReplayMessage]"
    running: asyncio.Event
    stop: asyncio.Event


class MessageReplay(ez.Unit):
    """
    Stream messages from files created by :obj:`MessageLogger`.
    Stores a queue of files to stream and streams from them in order.
    """

    SETTINGS: MessageReplaySettings
    STATE: MessageReplayState

    INPUT_FILE = ez.InputStream(FileReplayMessage)
    """Add a new file to the queue."""

    INPUT_PAUSED = ez.InputStream(bool)
    """Send ``True`` to pause the stream, ``False`` to restart the stream."""

    INPUT_STOP = ez.InputStream(bool)
    """
    Stop the stream. Send ``True`` to also clear the queue.
    Send ``False`` to reset to the beginning of the current file.
    """

    OUTPUT_MESSAGE = ez.OutputStream(typing.Any)
    """The output on which the messages from the files will be streamed."""

    OUTPUT_TOTAL = ez.OutputStream(int)
    """
    Publishes an integer total of messages which have been published on OUTPUT_MESSAGE from a single file.
    Resets when a file completes.
    """

    OUTPUT_REPLAY_STATUS = ez.OutputStream(ReplayStatusMessage)
    """Publishes status messages."""

    async def initialize(self) -> None:
        self.STATE.replay_files = asyncio.Queue()
        if self.SETTINGS.filename is not None:
            self.STATE.replay_files.put_nowait(self.SETTINGS)
        self.STATE.running = asyncio.Event()
        self.STATE.running.set()
        self.STATE.stop = asyncio.Event()

    @ez.subscriber(INPUT_FILE)
    async def queue_file(self, msg: FileReplayMessage) -> None:
        if msg.filename is not None:
            self.STATE.replay_files.put_nowait(msg)

    @ez.subscriber(INPUT_PAUSED)
    async def set_paused(self, paused: bool) -> None:
        if paused:
            self.STATE.running.clear()
        else:
            self.STATE.running.set()

    @ez.subscriber(INPUT_STOP)
    async def stop(self, clear_queue: bool) -> None:
        if clear_queue:  # If we stop with "true"; clear the queue
            while not self.STATE.replay_files.empty():
                self.STATE.replay_files.get_nowait()
        self.STATE.stop.set()

    @ez.publisher(OUTPUT_MESSAGE)
    @ez.publisher(OUTPUT_TOTAL)
    @ez.publisher(OUTPUT_REPLAY_STATUS)
    async def replay(self) -> typing.AsyncGenerator:
        while True:
            replay_file = await self.STATE.replay_files.get()
            if replay_file.filename is None:
                continue

            last_msg_t: typing.Optional[float] = None
            num_msgs = sum(1 for _ in open(replay_file.filename, "r"))
            replay_msg = ReplayStatusMessage(replay_file.filename, 0, num_msgs)
            yield self.OUTPUT_REPLAY_STATUS, replay_msg

            if self.STATE.stop.is_set():
                self.STATE.stop.clear()

            pub_msgs = 0
            playback_t0 = float()
            replay_t0 = float()
            with open(replay_file.filename, "r") as f:
                if self.SETTINGS.progress:
                    try:
                        import tqdm

                        f = tqdm.tqdm(f, total=num_msgs)
                    except ImportError:
                        ez.logger.info("progress requires tqdm installed")

                for line_idx, line in enumerate(f):
                    processing_start_time = time.time()

                    if not self.STATE.running.is_set():
                        await self.STATE.running.wait()

                    if self.STATE.stop.is_set():
                        self.STATE.stop.clear()
                        break

                    replay_msg = replace(replay_msg, idx=line_idx + 1)
                    yield self.OUTPUT_REPLAY_STATUS, replay_msg

                    try:
                        msg = json.loads(line, cls=MessageDecoder)
                    except json.JSONDecodeError:
                        ez.logger.warning(
                            f"Could not load line {line_idx} from {self.SETTINGS.filename}"
                        )
                    else:
                        ts = msg["ts"]
                        obj = msg["obj"]
                        if last_msg_t is None and ts is not None:
                            last_msg_t = ts
                            playback_t0 = ts
                            replay_t0 = time.time()

                        if replay_file.rate is not None:
                            if replay_file.rate > 0:
                                await asyncio.sleep(1.0 / replay_file.rate)
                            elif (
                                replay_file.rate == 0
                                and last_msg_t is not None
                                and ts is not None
                            ):
                                lag = (time.time() - replay_t0) - (ts - playback_t0)
                                if lag < 0.0:
                                    processing_time = (
                                        time.time() - processing_start_time
                                    )
                                    sleep_time = max(
                                        ts - (last_msg_t + processing_time), 0.0
                                    )
                                    await asyncio.sleep(sleep_time)
                                last_msg_t = ts

                        if isinstance(obj, LogStart):
                            continue

                        yield self.OUTPUT_MESSAGE, obj
                        pub_msgs += 1

            yield self.OUTPUT_REPLAY_STATUS, replace(replay_msg, done=True)
            yield self.OUTPUT_TOTAL, pub_msgs


class MessageCollectorState(ez.State):
    messages: typing.List[typing.Any] = field(default_factory=list)


class MessageCollector(ez.Unit):
    """
    Collects ``Messages`` into a local list.
    """

    STATE: MessageCollectorState

    INPUT_MESSAGE = ez.InputStream(typing.Any)
    """Send messages here to be collected."""

    OUTPUT_MESSAGE = ez.OutputStream(typing.Any)
    """Messages will pass straight through after being recorded and be published here."""

    @ez.subscriber(INPUT_MESSAGE)
    @ez.publisher(OUTPUT_MESSAGE)
    async def on_message(self, msg: typing.Any) -> typing.AsyncGenerator:
        self.STATE.messages.append(msg)
        yield self.OUTPUT_MESSAGE, msg

    @property
    def messages(self) -> typing.List[typing.Any]:
        """
        Access the list of messages.

        :return: A list of messages which have been collected.
        """
        return self.STATE.messages
