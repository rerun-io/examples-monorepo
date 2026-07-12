"""Behavior checks for MOV video-track preparation."""

import unittest
from pathlib import Path

import av
import numpy as np

from arkitscenes_download.ingest.mov import iter_video_samples, prepare_video_track


class IngestMovTest(unittest.TestCase):
    """Check AV1 preparation and packet iteration."""

    def test_prepared_track_becomes_ordered_raw_frame_samples(self) -> None:
        """VideoStream samples are raw AV1 packets with ordered PTS."""
        mov_path: Path = Path("data/raw/Training/47332195/47332195.mov")
        if not mov_path.exists():
            self.skipTest("ARKitScenes sample MOV is unavailable")

        video = prepare_video_track(mov_path, 2, 0)
        batches = list(iter_video_samples(video))
        timestamps = np.concatenate([batch.timestamps for batch in batches])
        payloads = [payload for batch in batches for payload in batch.payloads]
        is_keyframes = np.concatenate([batch.is_keyframes for batch in batches])

        self.assertEqual(len(timestamps), 1080)
        self.assertTrue((timestamps[1:] > timestamps[:-1]).all())
        self.assertEqual(len(timestamps), len(payloads))
        self.assertEqual(len(timestamps), len(is_keyframes))
        self.assertTrue(is_keyframes[0])
        self.assertFalse(payloads[0].startswith((b"\x00\x00\x01", b"\x00\x00\x00\x01")))

    def test_packet_iterator_is_bounded_and_av1_packets_are_not_reordered(self) -> None:
        """Prepared AV1 packets have PTS equal to DTS and batches stay bounded."""
        mov_path: Path = Path("data/raw/Training/47332195/47332195.mov")
        if not mov_path.exists():
            self.skipTest("ARKitScenes sample MOV is unavailable")

        video = prepare_video_track(mov_path, 2, 0)
        batches = list(iter_video_samples(video, batch_size=127))

        self.assertEqual(sum(len(batch.timestamps) for batch in batches), 1080)
        self.assertTrue(all(len(batch.timestamps) <= 127 for batch in batches))
        with av.open(str(video.path), options={"advanced_editlist": "0"}) as container:
            stream: av.video.stream.VideoStream = container.streams.video[0]
            self.assertEqual(stream.codec_context.codec.id, av.Codec("av1", "r").id)
            self.assertTrue(all(packet.pts == packet.dts for packet in container.demux(stream) if packet.size != 0))


if __name__ == "__main__":
    unittest.main()
