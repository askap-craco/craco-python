import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

# Mock missing HPC/astronomy packages if not installed locally
for mod_name in [
    "astropy", "astropy.io", "astropy.io.fits", "astropy.time",
    "craft", "craft.uvfits", "craft.craco",
    "aces", "aces.askapdata", "aces.askapdata.schedblock",
    "casacore", "casacore.tables",
    "craco.fixuvfits",
    "slack_sdk",
    "psycopg2", "psycopg2.extras",
    "sqlalchemy",
    "clink", "clink.api"
]:
    if mod_name not in sys.modules:
        try:
            __import__(mod_name)
        except ImportError:
            sys.modules[mod_name] = MagicMock()

import stat

# Ensure src/ is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from craco.casda_archiver import (
    ArchiveStatus,
    parse_sbid,
    parse_metadata_xml,
    get_calibration_files,
    build_ready_for_copy_payload,
    setup_clink_environment,
    make_user_writable
)

class TestCasdaArchiverClink(unittest.TestCase):
    def test_parse_sbid(self):
        self.assertEqual(parse_sbid(82418), 82418)
        self.assertEqual(parse_sbid("82418"), 82418)
        self.assertEqual(parse_sbid("SB82418"), 82418)
        self.assertEqual(parse_sbid("SB082418"), 82418)
        self.assertEqual(parse_sbid("sb82418"), 82418)

    def test_archive_status_enum(self):
        self.assertEqual(ArchiveStatus.DEFAULT, 0)
        self.assertEqual(ArchiveStatus.READY_FOR_COPY_SENT, 10)
        self.assertEqual(ArchiveStatus.COPY_QUEUED, 11)
        self.assertEqual(ArchiveStatus.COPY_EXECUTING, 12)
        self.assertEqual(ArchiveStatus.COPY_FINISHED, 13)
        self.assertEqual(ArchiveStatus.READY_FOR_PURGE, 20)
        self.assertEqual(ArchiveStatus.PURGED, 30)

    def test_parse_metadata_xml(self):
        xml_content = """<metadata>
  <filename>cracoData.LTR_1812-2849.SB82418.beam12.20260220224148.uvfits</filename>
  <project>AS116</project>
  <sbid>82418</sbid>
  <beam>12</beam>
  <scanid>20260220224148</scanid>
  <scanstart>2026-02-20T22:43:06</scanstart>
  <scanend>2026-02-20T22:56:55</scanend>
  <ra>4.741468714800596</ra>
  <dec>-0.5228195996979885</dec>
  <coordsystem>J2000</coordsystem>
  <fieldname>LTR_1812-2849</fieldname>
  <polarisations>XX</polarisations>
  <numchan>288</numchan>
  <centrefreq>887490740.7407407</centrefreq>
  <chanwidth>1000000.0</chanwidth>
  <timeSteps>7495</timeSteps>
  <inttime>0.11059200018644333</inttime>
</metadata>"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(xml_content)
            tmp_path = f.name

        try:
            meta = parse_metadata_xml(tmp_path)
            self.assertEqual(meta["project"], "AS116")
            self.assertEqual(meta["sbid"], 82418)
            self.assertEqual(meta["beam"], 12)
            self.assertEqual(meta["fieldname"], "LTR_1812-2849")
            self.assertEqual(meta["numchan"], 288)
            self.assertAlmostEqual(meta["ra"], 4.741468714800596, places=5)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_build_ready_for_copy_payload(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            sbid = 82418
            archive_dir = os.path.join(tmp_dir, f"SB{sbid}")
            scan_dir = os.path.join(archive_dir, "20260220224148")
            cal_dir = os.path.join(archive_dir, "cal")
            os.makedirs(scan_dir, exist_ok=True)
            os.makedirs(cal_dir, exist_ok=True)

            xml_path = os.path.join(scan_dir, f"cracoData.TestField.SB{sbid}.beam00.20260220224148.craco_metadata.xml")
            xml_content = f"""<metadata>
  <filename>cracoData.TestField.SB{sbid}.beam00.20260220224148.uvfits</filename>
  <project>AS116</project>
  <sbid>{sbid}</sbid>
  <beam>0</beam>
  <fieldname>TestField</fieldname>
</metadata>"""
            with open(xml_path, "w") as f:
                f.write(xml_content)

            cal_table = os.path.join(cal_dir, f"cracoCal.TestField.SB{sbid}.beam00.B0")
            with open(cal_table, "w") as f:
                f.write("dummy cal content")

            payload = build_ready_for_copy_payload(sbid, archive_folder=archive_dir)
            
            self.assertEqual(payload["schedulingBlock"]["id"], str(sbid))
            self.assertEqual(payload["schedulingBlock"]["owner"], "AS116")
            self.assertEqual(payload["schedulingBlock"]["alias"], "TestField")
            self.assertEqual(payload["schedulingBlock"]["state"], "OBSERVED")
            self.assertEqual(payload["craco"]["archive_folder"], archive_dir)
            self.assertEqual(len(payload["craco"]["scans"]), 1)
            self.assertEqual(payload["craco"]["scans"][0]["scanid"], "20260220224148")
            self.assertEqual(len(payload["craco"]["calibration"]["files"]), 1)
            self.assertEqual(payload["craco"]["calibration"]["files"][0], f"cracoCal.TestField.SB{sbid}.beam00.B0")

    def test_setup_clink_environment(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"CLINK_BACKEND": "clink.backends.dummy", "TEST_KEY": "TEST_VAL"}')
            tmp_path = f.name

        try:
            setup_clink_environment(tmp_path)
            self.assertEqual(os.environ.get("CLINK_BACKEND"), "clink.backends.dummy")
            self.assertEqual(os.environ.get("TEST_KEY"), "TEST_VAL")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_emit_ready_for_copy_test_mode(self):
        from craco.casda_archiver import ClinkPublisher
        import io
        from contextlib import redirect_stdout
        
        pub = ClinkPublisher()
        f = io.StringIO()
        with redirect_stdout(f):
            result = pub.emit_ready_for_copy(sbid=82418, test=True)
            
        self.assertTrue(result)
        out = f.getvalue()
        self.assertIn("--- CLINK EVENT PAYLOAD ---", out)
        self.assertIn("Subject URN: urn:askap:craco:::archive-folder//data/craco/craco/archive/SB82418", out)
        self.assertIn('"id": "82418"', out)

    def test_make_user_writable_file(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test data")
            tmp_file = f.name

        try:
            # Set to readonly
            os.chmod(tmp_file, 0o444)
            current_mode = os.stat(tmp_file).st_mode
            self.assertFalse(bool(current_mode & stat.S_IWUSR))

            # Make user writable
            res = make_user_writable(tmp_file)
            self.assertTrue(res)
            updated_mode = os.stat(tmp_file).st_mode
            self.assertTrue(bool(updated_mode & stat.S_IWUSR))
        finally:
            if os.path.exists(tmp_file):
                os.chmod(tmp_file, 0o644)
                os.remove(tmp_file)

    def test_make_user_writable_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            sub_dir = os.path.join(tmp_dir, "scans", "00", "20260220224148")
            os.makedirs(sub_dir, exist_ok=True)
            test_file = os.path.join(sub_dir, "b00.uvfits")
            with open(test_file, "w") as f:
                f.write("dummy uvfits")

            try:
                # Set directory and file to readonly
                os.chmod(test_file, 0o444)
                os.chmod(sub_dir, 0o555)

                self.assertFalse(bool(os.stat(sub_dir).st_mode & stat.S_IWUSR))
                self.assertFalse(bool(os.stat(test_file).st_mode & stat.S_IWUSR))

                # Make parent dir and file writable
                self.assertTrue(make_user_writable(sub_dir))
                self.assertTrue(make_user_writable(test_file))

                self.assertTrue(bool(os.stat(sub_dir).st_mode & stat.S_IWUSR))
                self.assertTrue(bool(os.stat(test_file).st_mode & stat.S_IWUSR))
            finally:
                os.chmod(sub_dir, 0o755)
                os.chmod(test_file, 0o644)

    def test_make_user_writable_nonexistent_and_none(self):
        self.assertFalse(make_user_writable(None))
        self.assertFalse(make_user_writable("/nonexistent/dummy/path/file.uvfits"))

    def test_ensure_writable_only_targets_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            scan_dir = os.path.join(tmp_dir, "scans", "00", "20260220224148")
            os.makedirs(scan_dir, exist_ok=True)
            test_file = os.path.join(scan_dir, "b00.uvfits")
            with open(test_file, "w") as f:
                f.write("dummy")

            try:
                os.chmod(test_file, 0o444)
                os.chmod(scan_dir, 0o555)

                from craco.casda_archiver import UvfitsCasdaMetadata
                mock_ucm = MagicMock(spec=UvfitsCasdaMetadata)
                mock_ucm.uvfitspath = test_file
                UvfitsCasdaMetadata.ensure_writable(mock_ucm)

                # File should be writable, parent dir should remain unchanged (readonly)
                self.assertTrue(bool(os.stat(test_file).st_mode & stat.S_IWUSR))
                self.assertFalse(bool(os.stat(scan_dir).st_mode & stat.S_IWUSR))
            finally:
                os.chmod(scan_dir, 0o755)
                os.chmod(test_file, 0o644)

    def test_cli_chmod_flag_defaults(self):
        from unittest.mock import patch
        from craco.casda_archiver import main

        with patch("sys.argv", ["casda_archiver.py", "--sbid", "82418", "--scan", "00", "--tstart", "20260220224148"]), \
             patch("craco.casda_archiver.ScanCasdaMetadata") as mock_scm_cls:
            mock_scm = MagicMock()
            mock_scm_cls.return_value = mock_scm
            main()
            mock_scm.make_scan_writable.assert_called_once()

        with patch("sys.argv", ["casda_archiver.py", "--sbid", "82418", "--scan", "00", "--tstart", "20260220224148", "--no-chmod"]), \
             patch("craco.casda_archiver.ScanCasdaMetadata") as mock_scm_cls:
            mock_scm = MagicMock()
            mock_scm_cls.return_value = mock_scm
            main()
            mock_scm.make_scan_writable.assert_not_called()

    def test_is_craco_event(self):
        from craco.casda_archiver import ClinkListener
        from unittest.mock import patch

        with patch("craco.casda_archiver.setup_clink_environment"), \
             patch("craco.casda_archiver.ArchiveManager"):
            listener = MagicMock(spec=ClinkListener)
            listener._is_craco_event = ClinkListener._is_craco_event.__get__(listener, ClinkListener)

            # 1. queue contains craco (case-insensitive)
            ev1 = MagicMock()
            ev1.data = {"item": {"queue": "craco_processing_queue"}}
            self.assertTrue(listener._is_craco_event(ev1))

            # 2. item path contains craco
            ev2 = MagicMock()
            ev2.data = {"item": {"path": "/data/craco/craco/archive/SB82418"}}
            self.assertTrue(listener._is_craco_event(ev2))

            # 3. subject_urn contains craco
            ev3 = MagicMock()
            ev3.data = {}
            ev3.subject_urn = "urn:askap:datamanager:::purge-item//data/craco/craco/archive/SB82418"
            self.assertTrue(listener._is_craco_event(ev3))

            # 4. Non-CRACO event
            ev4 = MagicMock()
            ev4.data = {"item": {"queue": "OTHER", "path": "/askapingest/ruby/askap-scheduling-blocks/88017"}}
            ev4.subject_urn = "urn:askap:datamanager:::purge-item//askapingest/ruby/askap-scheduling-blocks/88017"
            ev4.subject = None
            self.assertFalse(listener._is_craco_event(ev4))

            # 5. DB fallback check is disabled for safety (subsystems share SBIDs)
            # mock_am = MagicMock()
            # mock_am.get_records_by_query.return_value = [{"sbid": 82418}]
            # ev5 = MagicMock()
            # ev5.data = {}
            # ev5.subject_urn = None
            # ev5.subject = None
            # self.assertTrue(listener._is_craco_event(ev5, sbid=82418, am=mock_am))

    def test_extract_sbid_queue_events(self):
        from craco.casda_archiver import ClinkListener
        from unittest.mock import patch

        with patch("craco.casda_archiver.setup_clink_environment"), \
             patch("craco.casda_archiver.ArchiveManager"):
            listener = MagicMock(spec=ClinkListener)
            listener._extract_sbid = ClinkListener._extract_sbid.__get__(listener, ClinkListener)

            # Queue-level URN with no digits should return None without error
            ev_queue = MagicMock()
            ev_queue.data = {}
            ev_queue.subject_urn = MagicMock()
            ev_queue.subject_urn.resource.id = "POST_OBSERVATION_HIGH"
            ev_queue.subject = None
            self.assertIsNone(listener._extract_sbid(ev_queue))

            # Queue-level URN for CRACO with item path containing SBID
            ev_craco_queue = MagicMock()
            ev_craco_queue.data = {"item": {"path": "/data/craco/craco/archive/SB82418", "queue": "CRACO"}}
            ev_craco_queue.subject_urn = MagicMock()
            ev_craco_queue.subject_urn.resource.id = "CRACO"
            ev_craco_queue.subject = None
            self.assertEqual(listener._extract_sbid(ev_craco_queue), 82418)


if __name__ == "__main__":
    unittest.main()


