"""A deterministic TCP Reno congestion control and retransmission kernel.

The module models the sending half of a TCP Reno connection without touching a
real socket.  Time is supplied by an injectable Clock and every packet travels
through an EventQueue, so a whole transfer can be replayed step by step and
inspected from a test.

On top of that transport plumbing the sender keeps the classic state machine:
slow start, congestion avoidance, fast retransmit with fast recovery, a
retransmission timer with exponential back off, an RTT estimator after
Jacobson and Karels, and flow control driven by the window advertised by the
peer.
"""

from __future__ import annotations

import heapq
import itertools
import math
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# --------------------------------------------------------------------------
# Protocol constants
# --------------------------------------------------------------------------

MSS = 1000
INITIAL_CWND = 4 * MSS
INITIAL_SSTHRESH = 64 * MSS
INITIAL_RTO = 3.0
MIN_RTO = 0.2
MAX_RTO = 60.0
MAX_WINDOW = 1 << 30
CLOCK_GRANULARITY = 0.05
ALPHA = 0.125
BETA = 0.25
DUP_ACK_THRESHOLD = 3
PERSIST_INTERVAL = 1.0
DEFAULT_RECEIVE_CAPACITY = 16 * MSS
MAX_SEND_BUFFER = 1 << 26

# Event kinds used on the simulation queue.
EVENT_SEND = "send"
EVENT_RECEIVE = "receive"
EVENT_ACK = "ack"
EVENT_RTO = "rto"
EVENT_PERSIST = "persist"


class TcpError(Exception):
    """Base class for the errors raised by this module."""


class InvalidSegmentError(TcpError, ValueError):
    """Raised when a segment or an advertised window is out of range."""


class InvalidAckError(TcpError, ValueError):
    """Raised when an acknowledgement number cannot belong to the connection."""


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_int(
    value: Any,
    name: str,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
    error: type = ValueError,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise error("%s must be an integer" % name)
    if minimum is not None and value < minimum:
        raise error("%s must be at least %d" % (name, minimum))
    if maximum is not None and value > maximum:
        raise error("%s must be at most %d" % (name, maximum))
    return value


def _check_time(value: Any, name: str, allow_zero: bool = True) -> float:
    if not _is_finite_number(value):
        raise ValueError("%s must be a finite number" % name)
    value = float(value)
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError("%s must be positive" % name)
    return value


# --------------------------------------------------------------------------
# Virtual time and the event queue
# --------------------------------------------------------------------------


class Clock:
    """A virtual clock that only moves when the simulation advances it."""

    def __init__(self, start: float = 0.0) -> None:
        if not _is_finite_number(start) or float(start) < 0:
            raise ValueError("clock start must be a finite non-negative number")
        self._now = float(start)

    @property
    def now(self) -> float:
        """The current virtual time."""
        return self._now

    def advance(self, delta: float) -> float:
        """Move the clock forward and return the new time."""
        if not _is_finite_number(delta) or float(delta) < 0:
            raise ValueError("the clock can only move forward by a finite amount")
        self._now += float(delta)
        return self._now

    def __repr__(self) -> str:
        return "Clock(now=%r)" % (self._now,)


class Event:
    """A single entry of the queue: a kind of work to run at a given time."""

    __slots__ = ("time", "kind", "payload")

    def __init__(self, time: float, kind: str, payload: Any = None) -> None:
        self.time = float(time)
        self.kind = kind
        self.payload = payload

    def __repr__(self) -> str:
        return "Event(time=%r, kind=%r, payload=%r)" % (self.time, self.kind, self.payload)


class EventQueue:
    """A time ordered queue of pending events, first in first out on ties."""

    def __init__(self) -> None:
        self._heap: List[Tuple[float, int, Event]] = []
        self._order = itertools.count()

    def push(self, time: float, kind: str, payload: Any = None) -> Event:
        """Schedule an event and return it."""
        if not _is_finite_number(time) or float(time) < 0:
            raise ValueError("event time must be a finite non-negative number")
        event = Event(float(time), kind, payload)
        heapq.heappush(self._heap, (event.time, next(self._order), event))
        return event

    def peek(self) -> Event:
        """Return the next event without removing it."""
        if not self._heap:
            raise IndexError("the event queue is empty")
        return self._heap[0][2]

    def pop(self) -> Event:
        """Remove and return the next event."""
        if not self._heap:
            raise IndexError("the event queue is empty")
        return heapq.heappop(self._heap)[2]

    def clear(self) -> None:
        self._heap.clear()

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)

    def __repr__(self) -> str:
        return "EventQueue(pending=%d)" % (len(self._heap),)


# --------------------------------------------------------------------------
# RTT estimation
# --------------------------------------------------------------------------


class RttEstimator:
    """Round trip time estimator following Jacobson and Karels."""

    def __init__(self, initial_rto: float = INITIAL_RTO, granularity: float = CLOCK_GRANULARITY) -> None:
        if not _is_finite_number(initial_rto) or not 0 < float(initial_rto) <= MAX_RTO:
            raise ValueError("initial_rto must be a positive number of at most %d" % MAX_RTO)
        if not _is_finite_number(granularity) or float(granularity) <= 0:
            raise ValueError("granularity must be a positive number")
        self.initial_rto = float(initial_rto)
        self.granularity = float(granularity)
        self.srtt: Optional[float] = None
        self.rttvar: Optional[float] = None
        self.samples = 0

    def update(self, sample: float) -> float:
        """Fold one round trip measurement into the estimate."""
        if not _is_finite_number(sample) or float(sample) <= 0:
            raise ValueError("an rtt sample must be a positive number")
        sample = float(sample)
        if self.srtt is None:
            self.srtt = sample
            self.rttvar = sample / 2.0
        else:
            self.rttvar = (1.0 - BETA) * self.rttvar + BETA * abs(self.srtt - sample)
            self.srtt = (1.0 - ALPHA) * self.srtt + ALPHA * sample
        self.samples += 1
        return sample

    def rto(self) -> float:
        """The retransmission timeout implied by the current estimate."""
        if self.srtt is None:
            return min(max(self.initial_rto, MIN_RTO), MAX_RTO)
        value = self.srtt + max(self.granularity, 4.0 * self.rttvar)
        return min(max(value, MIN_RTO), MAX_RTO)

    def __repr__(self) -> str:
        return "RttEstimator(srtt=%r, rttvar=%r, samples=%d)" % (self.srtt, self.rttvar, self.samples)


# --------------------------------------------------------------------------
# Receiving side
# --------------------------------------------------------------------------


class Receiver:
    """Reassembles in order data and reports the window it can still accept."""

    def __init__(self, isn: int = 0, capacity: int = DEFAULT_RECEIVE_CAPACITY) -> None:
        self.isn = _check_int(isn, "isn", minimum=0)
        self.capacity = _check_int(capacity, "capacity", minimum=0)
        self.next_expected = self.isn
        self.unread = 0
        self.ooo: Dict[int, int] = {}
        self.received_total = 0
        self.duplicate_segments = 0

    def window(self) -> int:
        """The number of bytes this receiver is still able to buffer."""
        buffered = self.unread + sum(self.ooo.values())
        return max(0, self.capacity - buffered)

    def set_capacity(self, capacity: int) -> int:
        """Model the application changing how much room it leaves to the peer."""
        self.capacity = _check_int(capacity, "capacity", minimum=0)
        return self.capacity

    def read(self, nbytes: int) -> int:
        """Model the application consuming buffered data and return how much it took."""
        nbytes = _check_int(nbytes, "nbytes", minimum=0)
        taken = min(nbytes, self.unread)
        self.unread -= taken
        return taken

    def ack_number(self) -> int:
        """The acknowledgement number this receiver would put on a segment now."""
        return self.next_expected

    def deliver(self, seq: int, length: int) -> int:
        """Take one arriving segment and return the acknowledgement number to send back."""
        seq = _check_int(seq, "sequence number", minimum=0, error=InvalidSegmentError)
        length = _check_int(length, "segment length", minimum=0, error=InvalidSegmentError)
        if length == 0:
            return self.next_expected
        end = seq + length
        if end <= self.next_expected:
            self.duplicate_segments += 1
            return self.next_expected
        if seq > self.next_expected:
            if seq - self.next_expected < self.window():
                self.ooo[seq] = length
            self.next_expected = end
            return self.next_expected
        self.next_expected = end
        self.unread += length
        self.received_total += length
        self._drain_out_of_order()
        return self.next_expected

    def _drain_out_of_order(self) -> None:
        while self.next_expected in self.ooo:
            length = self.ooo.pop(self.next_expected)
            self.next_expected += length
            self.unread += length
            self.received_total += length

    def __repr__(self) -> str:
        return "Receiver(next_expected=%d, window=%d, received=%d)" % (
            self.next_expected,
            self.window(),
            self.received_total,
        )


# --------------------------------------------------------------------------
# Sending side
# --------------------------------------------------------------------------


class OutstandingSegment:
    """A segment that has been handed to the network and not yet acknowledged."""

    __slots__ = ("seq", "length", "tx_time", "transmissions", "retransmitted")

    def __init__(self, seq: int, length: int, tx_time: float) -> None:
        self.seq = seq
        self.length = length
        self.tx_time = tx_time
        self.transmissions = 0
        self.retransmitted = False

    @property
    def end(self) -> int:
        """The first sequence number past this segment."""
        return self.seq + self.length

    def __repr__(self) -> str:
        return "OutstandingSegment(seq=%d, length=%d, retransmitted=%r)" % (
            self.seq,
            self.length,
            self.retransmitted,
        )


class RenoConnection:
    """The sending half of a Reno connection: window, timers and statistics."""

    def __init__(
        self,
        clock: Clock,
        queue: EventQueue,
        mss: int = MSS,
        initial_cwnd: int = INITIAL_CWND,
        initial_ssthresh: int = INITIAL_SSTHRESH,
        initial_rto: float = INITIAL_RTO,
        isn: int = 0,
        rwnd: int = DEFAULT_RECEIVE_CAPACITY,
    ) -> None:
        self.mss = _check_int(mss, "mss", minimum=1)
        self.isn = _check_int(isn, "isn", minimum=0)
        initial_cwnd = _check_int(initial_cwnd, "initial_cwnd", minimum=self.mss, maximum=MAX_WINDOW)
        self.ssthresh = _check_int(initial_ssthresh, "initial_ssthresh", minimum=1, maximum=MAX_WINDOW)
        self.rwnd = _check_int(rwnd, "rwnd", minimum=0, maximum=MAX_WINDOW)
        self.clock = clock
        self.queue = queue

        self.cwnd = initial_cwnd
        self.rtt = RttEstimator(initial_rto=initial_rto)
        self.rto = self.rtt.rto()

        self.cum_ack = self.isn
        self._next_seq = self.isn
        self._unsent = 0
        self.outstanding: List[OutstandingSegment] = []

        self.dup_acks = 0
        self.fast_recovery = False

        self.timer_deadline: Optional[float] = None
        self.timer_generation = 0
        self.persist_deadline: Optional[float] = None
        self.persist_generation = 0

        self.stats: Dict[str, int] = {
            "segments_sent": 0,
            "retransmissions": 0,
            "fast_retransmits": 0,
            "timeouts": 0,
            "persist_probes": 0,
            "duplicate_acks": 0,
            "new_acks": 0,
            "stale_acks": 0,
        }

    # -- state exposed to callers -------------------------------------------

    @property
    def unsent_bytes(self) -> int:
        """Bytes the application has queued and the sender has not handed over yet."""
        return self._unsent

    @property
    def next_seq(self) -> int:
        """The sequence number the next new byte will carry."""
        return self._next_seq

    def flight_size(self) -> int:
        """Bytes that are in flight, that is sent but not yet acknowledged."""
        return sum(segment.length for segment in self.outstanding)

    def usable_window(self) -> int:
        """How many more bytes the sender may put in flight right now."""
        return min(self.cwnd, self.rwnd) - self.flight_size()

    # -- application interface ---------------------------------------------

    def send(self, nbytes: int) -> int:
        """Queue bytes from the application and hand out what the window allows."""
        nbytes = _check_int(nbytes, "nbytes", minimum=1)
        if self._unsent + nbytes > MAX_SEND_BUFFER:
            raise ValueError("the send buffer cannot hold %d more bytes" % nbytes)
        self._unsent += nbytes
        self._transmit()
        return nbytes

    def set_peer_window(self, rwnd: int) -> int:
        """Record the window advertised by the peer."""
        self.rwnd = _check_int(rwnd, "rwnd", minimum=0, maximum=MAX_WINDOW)
        return self.rwnd

    # -- incoming acknowledgements -----------------------------------------

    def on_ack(self, ack: int, rwnd: Optional[int] = None) -> None:
        """Process one acknowledgement coming back from the peer."""
        if not isinstance(ack, int) or isinstance(ack, bool):
            raise InvalidAckError("the acknowledgement number must be an integer")
        if ack < self.isn or ack > self._next_seq:
            raise InvalidAckError("the acknowledgement number %d is outside the sent range" % ack)
        if rwnd is not None:
            self.rwnd = _check_int(rwnd, "rwnd", minimum=0, maximum=MAX_WINDOW, error=InvalidSegmentError)
        if ack < self.cum_ack:
            self.stats["stale_acks"] += 1
            return
        if ack == self.cum_ack:
            if self.outstanding:
                self._on_duplicate_ack()
        else:
            self._on_new_ack(ack)
        self._transmit()

    def _on_duplicate_ack(self) -> None:
        """React to an acknowledgement that repeats the last one."""
        self.stats["duplicate_acks"] += 1
        self.dup_acks += 1
        if self.dup_acks > DUP_ACK_THRESHOLD:
            if self.fast_recovery:
                self.cwnd += self.mss
            else:
                self._enter_fast_recovery()

    def _on_new_ack(self, ack: int) -> None:
        """React to an acknowledgement that covers data which was not acknowledged before."""
        self.stats["new_acks"] += 1
        self.cum_ack = ack
        acknowledged = [segment for segment in self.outstanding if segment.end <= ack]
        self.outstanding = [segment for segment in self.outstanding if segment.end > ack]
        for segment in acknowledged:
            if segment.end == ack:
                self._update_rtt(segment)
        self.dup_acks = 0
        if self.fast_recovery:
            self.fast_recovery = False
            self.cwnd = self.ssthresh
        else:
            self._grow_window()
        self._restart_timer()

    def _grow_window(self) -> None:
        """Open the congestion window after an acknowledgement of new data."""
        if self.cwnd < self.ssthresh:
            self.cwnd += self.mss
        else:
            self.cwnd += max(1, (self.mss * self.mss) // self.cwnd)
        self.cwnd = min(self.cwnd, MAX_WINDOW)

    def _update_rtt(self, segment: OutstandingSegment) -> None:
        """Take a round trip measurement from an acknowledged segment."""
        sample = self.clock.now - segment.tx_time
        if sample <= 0:
            return
        self.rtt.update(sample)
        self.rto = self.rtt.rto()

    # -- loss recovery ------------------------------------------------------

    def _enter_fast_recovery(self) -> None:
        """React to enough duplicate acknowledgements with a retransmission."""
        self.ssthresh = max(self.flight_size() // 2, 2 * self.mss)
        self.cwnd = self.ssthresh
        self.fast_recovery = True
        self.stats["fast_retransmits"] += 1
        self._retransmit_oldest()

    def on_retransmit_timeout(self, generation: Optional[int] = None) -> bool:
        """Handle the retransmission timer firing, ignoring stale timer events."""
        if generation is not None and generation != self.timer_generation:
            return False
        self.timer_deadline = None
        if not self.outstanding:
            return False
        self.stats["timeouts"] += 1
        self.ssthresh = self.cwnd // 2
        self.cwnd = self.mss
        self.fast_recovery = False
        self.dup_acks = 0
        self._retransmit_oldest()
        self._back_off_rto()
        self._arm_timer()
        return True

    def _retransmit_oldest(self) -> Optional[OutstandingSegment]:
        """Send the oldest segment in flight again."""
        if not self.outstanding:
            return None
        segment = self.outstanding[0]
        segment.retransmitted = True
        self._transmit_segment(segment)
        self.stats["retransmissions"] += 1
        return segment

    def _back_off_rto(self) -> float:
        """Widen the retransmission timeout after it has expired."""
        self.rto = self.rto * 2
        return self.rto

    # -- timers -------------------------------------------------------------

    def _arm_timer(self) -> float:
        """Start the retransmission timer so that it expires a timeout from now."""
        self.timer_generation += 1
        self.timer_deadline = self.clock.now + self.rto
        self.queue.push(self.timer_deadline, EVENT_RTO, self.timer_generation)
        return self.timer_deadline

    def _disarm_timer(self) -> None:
        """Stop the retransmission timer."""
        self.timer_deadline = None
        self.timer_generation += 1

    def _restart_timer(self) -> None:
        """Follow the acknowledgement with the retransmission timer."""
        if not self.outstanding:
            self._disarm_timer()
        elif self.timer_deadline is None:
            self._arm_timer()

    def _persist_blocked(self) -> bool:
        """Whether a closed peer window is what keeps the send buffer from moving."""
        if self.outstanding or self._unsent <= 0:
            return False
        return self.usable_window() < 0

    def _arm_persist(self) -> Optional[float]:
        """Start the persist timer if it is not already running."""
        if self.persist_deadline is None:
            self.persist_generation += 1
            self.persist_deadline = self.clock.now + PERSIST_INTERVAL
            self.queue.push(self.persist_deadline, EVENT_PERSIST, self.persist_generation)
        return self.persist_deadline

    def _disarm_persist(self) -> None:
        """Stop the persist timer."""
        self.persist_deadline = None
        self.persist_generation += 1

    def _sync_persist_timer(self) -> None:
        """Keep the persist timer in step with the state of the peer window."""
        if self._persist_blocked():
            self._arm_persist()
        else:
            self._disarm_persist()

    def on_persist_timeout(self, generation: Optional[int] = None) -> bool:
        """Handle the persist timer firing, ignoring stale timer events."""
        if generation is not None and generation != self.persist_generation:
            return False
        self.persist_deadline = None
        if self.rwnd <= 0 and self._unsent > 0:
            self._send_probe()
        self._transmit()
        return True

    def _send_probe(self) -> Optional[OutstandingSegment]:
        """Push a single byte past a closed peer window to invite a window update."""
        length = 1
        segment = OutstandingSegment(seq=self._next_seq, length=length, tx_time=self.clock.now)
        self._next_seq += length
        self._unsent -= length
        self.outstanding.append(segment)
        self._transmit_segment(segment)
        self.stats["persist_probes"] += 1
        return segment

    # -- outgoing data ------------------------------------------------------

    def _transmit(self) -> None:
        """Hand buffered bytes to the network while the window allows it."""
        while self._unsent > 0:
            if self.rwnd <= 0:
                break
            length = min(self.mss, self._unsent)
            if self.usable_window() < length:
                break
            segment = OutstandingSegment(seq=self._next_seq, length=length, tx_time=self.clock.now)
            self._next_seq += length
            self._unsent -= length
            self.outstanding.append(segment)
            self._transmit_segment(segment)
        if self.outstanding and self.timer_deadline is None:
            self._arm_timer()
        self._sync_persist_timer()

    def _transmit_segment(self, segment: OutstandingSegment) -> None:
        """Put one segment on the wire."""
        segment.tx_time = self.clock.now
        segment.transmissions += 1
        self.stats["segments_sent"] += 1
        self.queue.push(self.clock.now, EVENT_SEND, (segment.seq, segment.length))

    def __repr__(self) -> str:
        return "RenoConnection(cwnd=%d, ssthresh=%d, inflight=%d, rto=%.3f)" % (
            self.cwnd,
            self.ssthresh,
            self.flight_size(),
            self.rto,
        )


# --------------------------------------------------------------------------
# Simulation plumbing
# --------------------------------------------------------------------------


class Link:
    """A deterministic one way path: fixed delay plus optional one shot losses."""

    def __init__(
        self,
        one_way_delay: float = 0.05,
        drop_seqs: Iterable[int] = (),
        drop_all_data: bool = False,
        drop_acks: bool = False,
    ) -> None:
        self.one_way_delay = _check_time(one_way_delay, "one_way_delay")
        self.drop_seqs: Set[int] = {_check_int(seq, "drop seq", minimum=0) for seq in drop_seqs}
        self.drop_all_data = bool(drop_all_data)
        self.drop_acks = bool(drop_acks)
        self._dropped_once: Set[int] = set()
        self.transmitted = 0
        self.dropped = 0
        self.delivered = 0

    def offers(self, seq: int) -> bool:
        """Return True when this transmission of the segment is allowed through."""
        self.transmitted += 1
        drop = self.drop_all_data or (seq in self.drop_seqs and seq not in self._dropped_once)
        if drop:
            self._dropped_once.add(seq)
            self.dropped += 1
            return False
        self.delivered += 1
        return True

    def __repr__(self) -> str:
        return "Link(delay=%.3f, dropped=%d)" % (self.one_way_delay, self.dropped)


class Simulator:
    """Runs a sender and a receiver against a link until the queue drains."""

    def __init__(
        self,
        mss: int = MSS,
        one_way_delay: float = 0.05,
        initial_cwnd: int = INITIAL_CWND,
        initial_ssthresh: int = INITIAL_SSTHRESH,
        initial_rto: float = INITIAL_RTO,
        receive_capacity: int = DEFAULT_RECEIVE_CAPACITY,
        auto_read: bool = False,
        isn: int = 0,
        link: Optional[Link] = None,
    ) -> None:
        self.clock = Clock()
        self.queue = EventQueue()
        self.link = link if link is not None else Link(one_way_delay=one_way_delay)
        self.receiver = Receiver(isn=isn, capacity=receive_capacity)
        self.sender = RenoConnection(
            clock=self.clock,
            queue=self.queue,
            mss=mss,
            initial_cwnd=initial_cwnd,
            initial_ssthresh=initial_ssthresh,
            initial_rto=initial_rto,
            isn=isn,
            rwnd=self.receiver.window(),
        )
        self.auto_read = bool(auto_read)
        self.events_processed = 0
        self.max_rto = float(initial_rto)

    def send(self, nbytes: int) -> int:
        """Queue application bytes on the sender."""
        return self.sender.send(nbytes)

    def run(self, until: Optional[float] = None, max_events: int = 200000) -> int:
        """Process events until the queue drains or the time limit is reached."""
        processed = 0
        while len(self.queue):
            event = self.queue.peek()
            if until is not None and event.time > until:
                break
            self.queue.pop()
            self.clock.advance(event.time - self.clock.now)
            self._dispatch(event)
            processed += 1
            self.events_processed += 1
            if self.sender.rto > self.max_rto:
                self.max_rto = self.sender.rto
            if processed >= max_events:
                break
        return processed

    def _dispatch(self, event: Event) -> None:
        if event.kind == EVENT_SEND:
            self._on_send(*event.payload)
        elif event.kind == EVENT_RECEIVE:
            self._on_receive(*event.payload)
        elif event.kind == EVENT_ACK:
            self._on_ack(*event.payload)
        elif event.kind == EVENT_RTO:
            self.sender.on_retransmit_timeout(event.payload)
        elif event.kind == EVENT_PERSIST:
            self.sender.on_persist_timeout(event.payload)
        else:
            raise ValueError("unknown event kind %r" % (event.kind,))

    def _on_send(self, seq: int, length: int) -> None:
        if not self.link.offers(seq):
            return
        self.queue.push(self.clock.now + self.link.one_way_delay, EVENT_RECEIVE, (seq, length))

    def _on_receive(self, seq: int, length: int) -> None:
        ack = self.receiver.deliver(seq, length)
        if self.auto_read:
            self.receiver.read(self.receiver.unread)
        if self.link.drop_acks:
            self.link.dropped += 1
            return
        self.queue.push(self.clock.now + self.link.one_way_delay, EVENT_ACK, (ack, self.receiver.window()))

    def _on_ack(self, ack: int, rwnd: int) -> None:
        self.sender.on_ack(ack, rwnd)

    def __repr__(self) -> str:
        return "Simulator(now=%.3f, pending=%d, received=%d)" % (
            self.clock.now,
            len(self.queue),
            self.receiver.received_total,
        )
