"""Regression tests for non-destructive LLM decision-CSV extension."""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_analysis_llm as analysis


class LLMDecisionCSVTests(unittest.TestCase):
    def _write_row(self, path: Path, header: list[str]) -> None:
        row = {field: "" for field in header}
        row.update(
            scenario="S1",
            episode="0",
            step="0",
            agent_id="1",
            move="Stay",
            reproduce="False",
            did_reproduce="False",
            did_move="False",
        )
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            writer.writeheader()
            writer.writerow(row)

    def test_missing_optional_columns_are_added_without_losing_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decisions.csv"
            self._write_row(path, analysis.DECISIONS_HEADER[:-2])

            with patch.object(analysis, "DECISIONS_CSV", path):
                analysis._ensure_decisions_header_for_append()
                analysis._ensure_decisions_header_for_append()

            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(list(rows[0]), analysis.DECISIONS_HEADER)
            self.assertEqual(rows[0]["scenario"], "S1")
            self.assertEqual(rows[0]["cached"], "")
            self.assertEqual(rows[0]["error"], "")

    def test_new_rows_use_the_same_combined_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decisions.csv"
            self._write_row(path, analysis.DECISIONS_HEADER)

            with patch.object(analysis, "DECISIONS_CSV", path):
                analysis._ensure_decisions_header_for_append()
                analysis._append_decisions(
                    [
                        analysis.DecisionRecord(
                            scenario="S1", episode=1, step=0, agent_id=2,
                            food_N=0.0, food_S=0.0, food_E=0.0, food_W=0.0,
                            occ_N=0, occ_S=0, occ_E=0, occ_W=0,
                            food_here=0.0, energy=1.0, free_cells=4,
                            move="Stay", reproduce=False,
                            did_reproduce=False, did_move=False,
                            cached=False, error=None,
                        )
                    ]
                )

            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
            self.assertEqual(reader.fieldnames, analysis.DECISIONS_HEADER)
            self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
