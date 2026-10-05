# tcpre

A dependency free TCP Reno congestion control and retransmission kernel.

The package models the sending half of a Reno connection without touching a
real socket. Time comes from an injectable virtual clock and every packet
travels through an event queue, so a whole transfer is deterministic and can
be replayed step by step from a test. It covers slow start, congestion
avoidance, fast retransmit with fast recovery, a retransmission timer with
exponential back off, RTT estimation after Jacobson and Karels, and flow
control against the window advertised by the peer.

## Layout

    tcpre/__init__.py     public names re-exported by the package
    tcpre/core.py         congestion control and retransmission kernel
    tests/__init__.py     test package marker
    tests/test_core.py    behavioural test suite

## Running the tests

From the project root:

    python3 -m unittest discover -s tests -v

Only the Python standard library is required; there is nothing to install.
