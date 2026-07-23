import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

# Mock missing HPC/astronomy packages if not installed locally
for mod_name in [
    "astropy", "astropy.io", "astropy.io.fits", "astropy.time",
    "craft", "craft.uvfits",
    "aces", "aces.askapdata", "aces.askapdata.schedblock",
    "casacore", "casacore.tables",
    "craco.fixuvfits"
]:
    if mod_name not in sys.modules:
        try:
            __import__(mod_name)
        except ImportError:
            sys.modules[mod_name] = MagicMock()

# Ensure src/ is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from craco.casda_archiver import (
    ArchiveStatus,
    parse_metadata_xml,
    get_calibration_files,
    build_ready_for_copy_payload,
    setup_clink_environment
)

class TestCasdaArchiverClink(unittest.TestCase):
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
            
            self.assertEqual(payload["schedulingBlock"]["id"], sbid)
            self.assertEqual(payload["schedulingBlock"]["owner"], "AS116")
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

if __name__ == "__main__":
    unittest.main()
