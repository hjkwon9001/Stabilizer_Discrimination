import io
import os
import re
import unittest
from unittest.mock import patch

from stabdisc.progress import ProgressUpdate, TerminalProgress


class Terminal(io.StringIO):
    def isatty(self):
        return True


class ProgressTests(unittest.TestCase):
    def test_redirected_progress_is_plain_and_disabled_is_silent(self):
        stream = io.StringIO()
        with TerminalProgress(stream=stream) as progress:
            progress(ProgressUpdate("searching", backend="python", solved_projectors=128))
        output = stream.getvalue()
        self.assertIn("128 states solved", output)
        self.assertIn("elapsed", output)
        self.assertNotIn("\r", output)
        self.assertNotIn("\x1b", output)
        self.assertNotIn("%", output)
        self.assertNotIn("ETA", output)
        self.assertIn("Complete", output)
        stream = io.StringIO()
        with TerminalProgress(enabled=False, stream=stream) as progress:
            progress(ProgressUpdate("searching"))
        self.assertEqual(stream.getvalue(), "")

    def test_known_root_fraction_and_partial_bound(self):
        stream = Terminal()
        progress = TerminalProgress(stream=stream)
        progress(ProgressUpdate("solving", completed=5, total=15, best_probability=0.75))
        progress.close()
        output = stream.getvalue()
        self.assertIn("5/15 roots (33.3%)", output)
        self.assertIn("best so far (lower bound) 0.75", output)
        self.assertNotIn("success probability", output)
        self.assertNotIn("100.0%", output)
        self.assertIn("\r", output)
        self.assertTrue(output.endswith("\n"))

    def test_interruption_never_fakes_completion(self):
        stream = io.StringIO()
        with self.assertRaises(KeyboardInterrupt):
            with TerminalProgress(stream=stream) as progress:
                progress(ProgressUpdate("solving", completed=1, total=15))
                raise KeyboardInterrupt()
        self.assertIn("Interrupted", stream.getvalue())
        self.assertNotIn("Complete", stream.getvalue())
        self.assertNotIn("100.0%", stream.getvalue())

    def test_failed_event_preserved_on_normal_close(self):
        stream = io.StringIO()
        with TerminalProgress(stream=stream) as progress:
            progress(ProgressUpdate("failed", message="solver failed"))
        self.assertNotIn("Complete", stream.getvalue())

    def test_plain_logs_rate_limited_and_final_deduplicated(self):
        stream = io.StringIO()
        with TerminalProgress(stream=stream) as progress:
            for count in range(500):
                progress(ProgressUpdate("solving", completed=count, total=1000))
            progress(ProgressUpdate("complete", completed=1000, total=1000, best_probability=0.75))
        lines = stream.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("success probability 0.75", lines[-1])

    def test_control_characters_are_not_forwarded(self):
        stream = io.StringIO()
        progress = TerminalProgress(stream=stream)
        progress(ProgressUpdate("solving", message="hello\x1b[31m\nworld"))
        progress.close()
        self.assertNotIn("\x1b", stream.getvalue())

    def test_tty_wraps_and_clears_every_previous_row(self):
        stream = Terminal()
        progress = TerminalProgress(stream=stream)
        with patch("stabdisc.progress.shutil.get_terminal_size", return_value=os.terminal_size((60, 24))):
            progress(ProgressUpdate("solving", completed=0, total=4095, best_probability=0.5,
                                    message="last reported branch: worker 0, 3,500,000 states solved"))
            first = stream.getvalue()
            self.assertTrue(all(len(line) <= 59 for line in first.strip("\r").splitlines()))
            self.assertIn("best so far (lower bound)", first)
            progress.close()
        self.assertEqual(stream.getvalue().count("\x1b[2K"), len(first.strip("\r").splitlines()))
        self.assertIn("3,500,000 states solved", re.sub(r"\s+", " ", first))


if __name__ == "__main__":
    unittest.main()
