"""Behavioural tests for the tcpre congestion control and retransmission kernel."""

import unittest

from tcpre.core import (
    Clock,
    EventQueue,
    InvalidAckError,
    InvalidSegmentError,
    Link,
    MAX_RTO,
    MSS,
    Receiver,
    RenoConnection,
    RttEstimator,
    Simulator,
)


class CongestionWindowTests(unittest.TestCase):
    def setUp(self):
        self.mss = MSS

    def test_congestion_window_growth(self):
        mss = self.mss
        sim = Simulator(
            mss=mss,
            initial_cwnd=4 * mss,
            initial_ssthresh=64 * mss,
            receive_capacity=64 * mss,
        )
        conn = sim.sender
        conn.send(4 * mss)
        self.assertEqual(conn.cwnd, 4 * mss)
        self.assertEqual(conn.flight_size(), 4 * mss)
        self.assertLessEqual(conn.flight_size(), min(conn.cwnd, conn.rwnd))

        seen = [conn.cwnd]
        for step in range(1, 5):
            sim.clock.advance(0.05)
            conn.on_ack(step * mss)
            seen.append(conn.cwnd)
            self.assertLessEqual(conn.flight_size(), min(conn.cwnd, conn.rwnd))
            self.assertLessEqual(conn.cwnd, 64 * mss)
        self.assertEqual(conn.cwnd, 8 * mss)
        self.assertEqual(conn.flight_size(), 0)
        for previous, current in zip(seen, seen[1:]):
            self.assertLess(previous, current)

        sim2 = Simulator(
            mss=mss,
            initial_cwnd=32 * mss,
            initial_ssthresh=32 * mss,
            receive_capacity=64 * mss,
        )
        conn2 = sim2.sender
        conn2.send(32 * mss)
        self.assertEqual(conn2.flight_size(), 32 * mss)
        previous = conn2.cwnd
        for step in range(1, 5):
            sim2.clock.advance(0.05)
            conn2.on_ack(step * mss)
            self.assertGreater(conn2.cwnd, previous)
            self.assertLess(conn2.cwnd - previous, mss)
            previous = conn2.cwnd
        self.assertEqual(conn2.cwnd - 32 * mss, 4 * ((mss * mss) // (32 * mss)))

    def test_fast_retransmit_after_three_duplicate_acknowledgements(self):
        mss = self.mss
        sim = Simulator(
            mss=mss,
            initial_cwnd=5 * mss,
            initial_ssthresh=64 * mss,
            receive_capacity=64 * mss,
        )
        conn = sim.sender
        conn.send(5 * mss)
        self.assertEqual(conn.flight_size(), 5 * mss)

        sim.clock.advance(0.05)
        conn.on_ack(mss)
        self.assertEqual(conn.cum_ack, mss)
        self.assertFalse(conn.fast_recovery)

        for _ in range(3):
            sim.clock.advance(0.01)
            conn.on_ack(mss)

        self.assertTrue(conn.fast_recovery)
        self.assertEqual(conn.stats["fast_retransmits"], 1)
        self.assertEqual(conn.stats["retransmissions"], 1)
        self.assertEqual(conn.outstanding[0].seq, mss)
        self.assertTrue(conn.outstanding[0].retransmitted)
        self.assertEqual(conn.ssthresh, max(conn.flight_size() // 2, 2 * mss))
        self.assertEqual(conn.cwnd, conn.ssthresh + 3 * mss)

        sim.clock.advance(0.01)
        conn.on_ack(mss)
        self.assertEqual(conn.cwnd, conn.ssthresh + 4 * mss)

        link = Link(one_way_delay=0.05, drop_seqs=[mss])
        sim2 = Simulator(
            mss=mss,
            initial_cwnd=5 * mss,
            initial_ssthresh=64 * mss,
            receive_capacity=64 * mss,
            link=link,
        )
        sim2.send(5 * mss)
        sim2.run(until=50.0)
        self.assertEqual(sim2.receiver.received_total, 5 * mss)
        self.assertEqual(sim2.receiver.ack_number(), 5 * mss)
        self.assertEqual(sim2.sender.stats["fast_retransmits"], 1)
        self.assertEqual(sim2.sender.stats["timeouts"], 0)
        self.assertEqual(sim2.sender.stats["retransmissions"], 1)

    def test_retransmission_timeout_collapses_the_window(self):
        mss = self.mss
        link = Link(one_way_delay=0.05, drop_all_data=True)
        sim = Simulator(mss=mss, initial_cwnd=2 * mss, initial_rto=1.0, link=link)
        conn = sim.sender
        conn.send(2 * mss)
        self.assertEqual(conn.flight_size(), 2 * mss)

        sim.run(until=2.0)
        self.assertEqual(conn.stats["timeouts"], 1)
        self.assertEqual(conn.cwnd, mss)
        self.assertGreaterEqual(conn.ssthresh, 2 * mss)
        self.assertEqual(conn.ssthresh, 2 * mss)
        self.assertIsNotNone(conn.timer_deadline)
        self.assertEqual(conn.stats["retransmissions"], 1)
        self.assertTrue(conn.outstanding[0].retransmitted)

        sim.run(until=4.0)
        self.assertEqual(conn.stats["timeouts"], 2)
        self.assertEqual(conn.cwnd, mss)
        self.assertGreaterEqual(conn.ssthresh, 2 * mss)


class RetransmissionTests(unittest.TestCase):
    def setUp(self):
        self.mss = MSS

    def test_retransmission_timeout_backoff_stays_bounded(self):
        mss = self.mss
        link = Link(one_way_delay=0.05, drop_all_data=True)
        sim = Simulator(mss=mss, initial_cwnd=mss, initial_rto=1.0, link=link)
        conn = sim.sender
        conn.send(5 * mss)

        sim.run(until=300.0)
        self.assertGreaterEqual(conn.stats["timeouts"], 8)
        self.assertLessEqual(sim.max_rto, MAX_RTO)
        self.assertLessEqual(conn.rto, MAX_RTO)
        self.assertEqual(conn.rto, MAX_RTO)
        self.assertEqual(conn.timer_deadline, sim.clock.now + conn.rto)

    def test_retransmitted_segments_stay_out_of_the_rtt_estimate(self):
        mss = self.mss
        sim = Simulator(mss=mss, initial_cwnd=2 * mss, initial_rto=1.0)
        conn = sim.sender
        conn.send(2 * mss)

        sim.clock.advance(2.0)
        self.assertTrue(conn.on_retransmit_timeout(conn.timer_generation))
        self.assertTrue(conn.outstanding[0].retransmitted)
        self.assertEqual(conn.rtt.samples, 0)

        sim.clock.advance(1.0)
        conn.on_ack(mss)
        self.assertEqual(conn.rtt.samples, 0)
        self.assertIsNone(conn.rtt.srtt)
        self.assertIsNone(conn.rtt.rttvar)

        conn.send(mss)
        sim.clock.advance(1.0)
        conn.on_ack(2 * mss)
        self.assertEqual(conn.rtt.samples, 1)
        self.assertIsNotNone(conn.rtt.srtt)
        self.assertIsNotNone(conn.rtt.rttvar)

    def test_retransmission_timer_follows_acknowledgements(self):
        mss = self.mss
        sim = Simulator(
            mss=mss,
            initial_cwnd=4 * mss,
            initial_rto=1.0,
            receive_capacity=64 * mss,
        )
        conn = sim.sender
        conn.send(4 * mss)
        self.assertIsNotNone(conn.timer_deadline)
        self.assertAlmostEqual(conn.timer_deadline, sim.clock.now + conn.rto, places=9)

        sim.clock.advance(0.9)
        conn.on_ack(mss)
        self.assertEqual(conn.flight_size(), 3 * mss)
        self.assertAlmostEqual(conn.timer_deadline, sim.clock.now + conn.rto, places=9)
        self.assertAlmostEqual(conn.timer_deadline, 0.9 + conn.rto, places=9)

        settled = conn.cwnd
        self.assertFalse(conn.on_retransmit_timeout(conn.timer_generation - 1))
        self.assertEqual(conn.cwnd, settled)
        self.assertEqual(conn.stats["timeouts"], 0)

        sim2 = Simulator(
            mss=mss,
            one_way_delay=0.5,
            initial_cwnd=4 * mss,
            initial_rto=3.0,
            receive_capacity=8 * mss,
            auto_read=True,
        )
        sim2.send(64 * mss)
        sim2.run(until=200.0)
        self.assertEqual(sim2.receiver.received_total, 64 * mss)
        self.assertEqual(sim2.sender.stats["timeouts"], 0)
        self.assertEqual(sim2.sender.stats["retransmissions"], 0)


class FlowControlTests(unittest.TestCase):
    def setUp(self):
        self.mss = MSS

    def test_receiver_reports_the_contiguous_prefix(self):
        mss = self.mss
        receiver = Receiver(isn=0, capacity=16 * mss)

        self.assertEqual(receiver.deliver(0, mss), mss)
        self.assertEqual(receiver.ack_number(), mss)

        self.assertEqual(receiver.deliver(2 * mss, mss), mss)
        self.assertEqual(receiver.ack_number(), mss)
        self.assertEqual(receiver.received_total, mss)

        self.assertEqual(receiver.deliver(3 * mss, mss), mss)
        self.assertEqual(receiver.ack_number(), mss)
        self.assertEqual(receiver.received_total, mss)

        self.assertEqual(receiver.deliver(mss, mss), 4 * mss)
        self.assertEqual(receiver.ack_number(), 4 * mss)
        self.assertEqual(receiver.received_total, 4 * mss)
        self.assertEqual(receiver.window(), 12 * mss)

        self.assertEqual(receiver.deliver(mss, mss), 4 * mss)
        self.assertEqual(receiver.duplicate_segments, 1)

    def test_closed_peer_window_stops_and_resumes_the_transfer(self):
        mss = self.mss
        sim = Simulator(
            mss=mss,
            one_way_delay=0.05,
            initial_cwnd=4 * mss,
            receive_capacity=4 * mss,
        )
        conn = sim.sender
        sim.send(12 * mss)

        sim.run(until=0.9)
        self.assertEqual(sim.receiver.received_total, 4 * mss)
        self.assertEqual(sim.receiver.window(), 0)
        self.assertEqual(conn.unsent_bytes, 8 * mss)
        self.assertEqual(conn.flight_size(), 0)
        self.assertEqual(conn.stats["timeouts"], 0)
        self.assertIsNotNone(conn.persist_deadline)

        sim.receiver.set_capacity(12 * mss)
        sim.run(until=60.0)
        self.assertEqual(sim.receiver.received_total, 12 * mss)
        self.assertEqual(conn.unsent_bytes, 0)
        self.assertGreaterEqual(conn.stats["persist_probes"], 1)


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.mss = MSS

    def test_boundaries_and_invalid_input(self):
        mss = self.mss
        clock = Clock()
        with self.assertRaises(ValueError):
            clock.advance(-0.5)
        with self.assertRaises(ValueError):
            clock.advance(float("nan"))
        self.assertEqual(clock.now, 0.0)
        with self.assertRaises(ValueError):
            Clock(start=-1.0)

        with self.assertRaises(ValueError):
            RenoConnection(Clock(), EventQueue(), mss=0)
        with self.assertRaises(ValueError):
            RenoConnection(Clock(), EventQueue(), mss=mss, initial_cwnd=mss - 1)
        with self.assertRaises(ValueError):
            RenoConnection(Clock(), EventQueue(), mss=mss, rwnd=-1)
        with self.assertRaises(ValueError):
            Receiver(capacity=-1)

        estimator = RttEstimator()
        with self.assertRaises(ValueError):
            estimator.update(0.0)
        with self.assertRaises(ValueError):
            estimator.update(-2.0)
        self.assertEqual(estimator.samples, 0)
        estimator.update(1.0)
        self.assertAlmostEqual(estimator.srtt, 1.0)
        self.assertAlmostEqual(estimator.rttvar, 0.5)
        estimator.update(2.0)
        self.assertAlmostEqual(estimator.srtt, 1.125)
        self.assertAlmostEqual(estimator.rttvar, 0.625)
        self.assertAlmostEqual(estimator.rto(), 3.625)

        receiver = Receiver(isn=0, capacity=8 * mss)
        with self.assertRaises(InvalidSegmentError):
            receiver.deliver(-1, mss)
        with self.assertRaises(InvalidSegmentError):
            receiver.deliver(0, -1)
        self.assertEqual(receiver.deliver(0, 0), 0)
        self.assertEqual(receiver.received_total, 0)
        self.assertEqual(receiver.deliver(0, mss), mss)
        self.assertEqual(receiver.deliver(0, mss), mss)
        self.assertEqual(receiver.duplicate_segments, 1)
        self.assertEqual(receiver.received_total, mss)

        conn = RenoConnection(Clock(), EventQueue(), mss=mss, initial_cwnd=2 * mss)
        conn.send(2 * mss)
        with self.assertRaises(InvalidAckError):
            conn.on_ack(3 * mss)
        with self.assertRaises(InvalidSegmentError):
            conn.on_ack(mss, -1)
        self.assertEqual(conn.cum_ack, 0)
        conn.on_ack(mss)
        settled = conn.cwnd
        conn.on_ack(0)
        self.assertEqual(conn.stats["stale_acks"], 1)
        self.assertEqual(conn.cum_ack, mss)
        self.assertEqual(conn.cwnd, settled)
        self.assertEqual(conn.dup_acks, 0)
        self.assertFalse(conn.on_retransmit_timeout(conn.timer_generation + 5))
        self.assertEqual(conn.cwnd, settled)
        self.assertEqual(conn.stats["timeouts"], 0)


if __name__ == "__main__":
    unittest.main()
