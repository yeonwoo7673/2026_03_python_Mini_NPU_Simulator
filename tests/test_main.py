"""Mini NPU Simulator 단위 테스트 (표준 라이브러리 unittest만 사용).

실행: python -m unittest discover -s tests -v
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import main as npu  # noqa: E402

DATA_PATH = os.path.join(ROOT, "data.json")
ERROR_DATA_PATH = os.path.join(ROOT, "data_error_cases.json")

CROSS_3 = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
X_3 = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]


def quiet(func, *args, **kwargs):
    """표준 출력을 숨기고 함수를 실행한다."""
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def line_reader(lines):
    """input() 대신 쓸 가짜 입력기. 다 쓰면 EOFError."""
    iterator = iter(lines)

    def reader():
        try:
            return next(iterator)
        except StopIteration:
            raise EOFError from None

    return reader


class MatrixTest(unittest.TestCase):
    def test_get_set(self):
        m = npu.Matrix(3)
        m.set(1, 2, 7)
        self.assertEqual(m.get(1, 2), 7.0)
        self.assertEqual(m.get(0, 0), 0.0)

    def test_out_of_range(self):
        with self.assertRaises(IndexError):
            npu.Matrix(3).get(3, 0)

    def test_from_rows_rejects_bad_shapes_and_values(self):
        bad_inputs = [
            [],                          # 빈 배열
            [[1, 2], [3]],               # 행/열 불일치
            [[1, 2, 3], [4, 5, 6]],      # 비정사각 (2행 3열)
            [[1, "a"], [1, 2]],          # 문자열
            [[True, 0], [0, 0]],         # bool
            [[float("nan"), 0], [0, 0]], # NaN
            "not a list",
        ]
        for rows in bad_inputs:
            with self.subTest(rows=rows):
                with self.assertRaises(npu.SchemaError):
                    npu.Matrix.from_rows(rows)

    def test_supports_required_sizes(self):
        for n in (3, 5, 13, 25):
            m = npu.Matrix.from_rows([[0.0] * n for _ in range(n)])
            self.assertEqual(m.size, n)
            self.assertEqual(len(m.to_flat()), n * n)

    def test_to_flat_is_row_major(self):
        m = npu.Matrix.from_rows([[1, 2], [3, 4]])
        self.assertEqual(m.to_flat(), [1.0, 2.0, 3.0, 4.0])


class MacTest(unittest.TestCase):
    def test_mission_examples(self):
        cross = npu.Matrix.from_rows(CROSS_3)
        x = npu.Matrix.from_rows(X_3)
        self.assertEqual(npu.mac(cross, cross), 5.0)   # Case 1
        self.assertEqual(npu.mac(cross, x), 1.0)       # Case 2

    def test_size_mismatch_raises(self):
        with self.assertRaises(npu.SchemaError):
            npu.mac(npu.Matrix(3), npu.Matrix(5))

    def test_flat_equals_2d_on_data_json(self):
        data, _ = npu.load_json(DATA_PATH)
        filters, _ = npu.load_filters(data["filters"])
        for case_id, entry in data["patterns"].items():
            n = int(case_id.split("_")[1])
            pattern = npu.Matrix.from_rows(entry["input"])
            for filt in filters[n].values():
                with self.subTest(case=case_id):
                    self.assertEqual(npu.mac(pattern, filt),
                                     npu.mac_flat(pattern.to_flat(), filt.to_flat()))

    def test_measure_returns_positive_ms(self):
        m = npu.generate_cross(5)
        self.assertGreater(npu.measure_avg_ms(npu.mac, m, m), 0.0)


class DecideTest(unittest.TestCase):
    def test_normal(self):
        self.assertEqual(npu.decide(5.0, 1.0, "A", "B"), "A")
        self.assertEqual(npu.decide(1.0, 5.0, "A", "B"), "B")

    def test_float_error_is_tie(self):
        # 0.1 을 9번 더하면 0.8999999999999999 — 수학적으로는 0.9와 같다.
        total = 0.0
        for _ in range(9):
            total += 0.1
        self.assertNotEqual(total, 0.9)
        self.assertEqual(npu.decide(0.9, total, "A", "B"), npu.UNDECIDED)

    def test_epsilon_boundary(self):
        self.assertEqual(npu.decide(1.0, 1.0 + 5e-10, "A", "B"), npu.UNDECIDED)
        self.assertEqual(npu.decide(1.0, 1.0 + 2e-9, "A", "B"), "B")


class LabelTest(unittest.TestCase):
    def test_normalize(self):
        cases = {"+": "Cross", "cross": "Cross", "Cross": "Cross", " CROSS ": "Cross",
                 "x": "X", "X": "X"}
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(npu.normalize_label(raw), expected)

    def test_unknown(self):
        for raw in ("o", "", None, 1, "plus"):
            with self.subTest(raw=raw):
                self.assertIsNone(npu.normalize_label(raw))


class DataJsonTest(unittest.TestCase):
    """제공된 data.json: 3 PASS, 3 FAIL(동점)."""

    @classmethod
    def setUpClass(cls):
        cls.outcome = quiet(npu.run_json_mode, DATA_PATH)
        cls.by_id = {r.case_id: r for r in cls.outcome["results"]}

    def test_filters_loaded_with_standard_labels(self):
        self.assertEqual(sorted(self.outcome["filters"]), [5, 13, 25])
        for pair in self.outcome["filters"].values():
            self.assertEqual(sorted(pair), ["Cross", "X"])
        self.assertEqual(self.outcome["filter_errors"], {})

    def test_summary(self):
        s = self.outcome["summary"]
        self.assertEqual((s["total"], s["passed"], s["failed"]), (6, 3, 3))

    def test_pass_cases(self):
        for case_id in ("size_5_2", "size_13_1", "size_25_2"):
            with self.subTest(case=case_id):
                self.assertTrue(self.by_id[case_id].passed)

    def test_tie_cases_fail_as_undecided(self):
        for case_id in ("size_5_1", "size_13_2", "size_25_1"):
            with self.subTest(case=case_id):
                r = self.by_id[case_id]
                self.assertFalse(r.passed)
                self.assertEqual(r.decision, npu.UNDECIDED)
                self.assertEqual(r.cause, npu.CAUSE_NUMERIC)

    def test_naive_comparison_would_pick_wrong_label(self):
        # epsilon 없이 > 로만 비교하면 동점 3건이 전부 expected 의 반대로 판정된다.
        for case_id in ("size_5_1", "size_13_2", "size_25_1"):
            r = self.by_id[case_id]
            naive = "Cross" if r.cross_score > r.x_score else "X"
            with self.subTest(case=case_id):
                self.assertNotEqual(naive, r.expected)

    def test_perf_covers_required_sizes(self):
        self.assertEqual([row["n"] for row in self.outcome["perf"]], [3, 5, 13, 25])
        self.assertTrue(all(row["same"] for row in self.outcome["perf"]))


class ErrorCasesTest(unittest.TestCase):
    """의도적 오류 데이터: 프로그램이 멈추지 않고 케이스 단위로 FAIL 처리하는지."""

    EXPECTED = {
        "size_5_1": (True, None),
        "size_5_2": (True, None),
        "size_5_3": (False, npu.CAUSE_SCHEMA),    # 라벨 'o'
        "size_5_4": (False, npu.CAUSE_SCHEMA),    # 키 5 vs input 3×3
        "size_5_5": (False, npu.CAUSE_SCHEMA),    # 행/열 불일치
        "size_5_6": (False, npu.CAUSE_SCHEMA),    # 문자열 값
        "size_5_7": (False, npu.CAUSE_SCHEMA),    # expected 누락
        "size_5_8": (False, npu.CAUSE_LOGIC),     # 판정 Cross vs expected X
        "size_5_9": (False, npu.CAUSE_NUMERIC),   # 0 == 0 동점
        "size_5_10": (False, npu.CAUSE_SCHEMA),   # input 누락
        "size_3_1": (False, npu.CAUSE_SCHEMA),    # 필터 크기 불일치
        "size_7_1": (False, npu.CAUSE_SCHEMA),    # X 필터 누락
        "size_9_1": (False, npu.CAUSE_SCHEMA),    # size_9 필터 없음
        "pattern_1": (False, npu.CAUSE_SCHEMA),   # 키 형식 오류
    }

    @classmethod
    def setUpClass(cls):
        cls.outcome = quiet(npu.run_json_mode, ERROR_DATA_PATH)
        cls.by_id = {r.case_id: r for r in cls.outcome["results"]}

    def test_every_case_classified(self):
        self.assertEqual(set(self.by_id), set(self.EXPECTED))
        for case_id, (passed, cause) in self.EXPECTED.items():
            with self.subTest(case=case_id):
                self.assertEqual(self.by_id[case_id].passed, passed)
                self.assertEqual(self.by_id[case_id].cause, cause)
                if not passed:
                    self.assertTrue(self.by_id[case_id].reason)

    def test_summary(self):
        s = self.outcome["summary"]
        self.assertEqual((s["total"], s["passed"], s["failed"]), (14, 2, 12))

    def test_filter_errors_recorded(self):
        errors = self.outcome["filter_errors"]
        self.assertIn(3, errors)
        self.assertIn(7, errors)
        self.assertIn("five", errors)


class LoadJsonTest(unittest.TestCase):
    def test_missing_file(self):
        data, error = npu.load_json(os.path.join(ROOT, "no_such_file.json"))
        self.assertIsNone(data)
        self.assertIn("찾을 수 없습니다", error)

    def test_broken_json_does_not_crash(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                         encoding="utf-8") as f:
            f.write('{"filters": {')
            path = f.name
        try:
            self.assertIsNone(quiet(npu.run_json_mode, path))
        finally:
            os.remove(path)

    def test_missing_sections(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                         encoding="utf-8") as f:
            json.dump({"meta": {}}, f)
            path = f.name
        try:
            outcome = quiet(npu.run_json_mode, path)
            self.assertEqual(outcome["summary"]["total"], 0)
            self.assertIn("filters", outcome["filter_errors"])
        finally:
            os.remove(path)


class GeneratorTest(unittest.TestCase):
    def test_cross_and_x_3(self):
        self.assertEqual(npu.generate_cross(3).data,
                         [[float(v) for v in row] for row in CROSS_3])
        self.assertEqual(npu.generate_x(3).data,
                         [[float(v) for v in row] for row in X_3])

    def test_scores_for_any_n(self):
        for n in (3, 5, 13, 25, 51):
            cross, x = npu.generate_cross(n), npu.generate_x(n)
            with self.subTest(n=n):
                self.assertEqual(npu.mac(cross, cross), 2 * n - 1)
                self.assertEqual(npu.mac(x, x), 2 * n - 1)
                self.assertEqual(npu.mac(cross, x), 1.0)   # 중심 1칸만 겹침

    def test_invalid_sizes(self):
        for n in (1, 2, 4, 0, -3, 1001, "5", True):
            with self.subTest(n=n):
                with self.assertRaises(npu.SchemaError):
                    npu.generate_cross(n)

    def test_mode3(self):
        outcome = quiet(npu.run_generator_mode, line_reader(["4", "3 5 13"]))
        self.assertEqual(outcome["sizes"], [3, 5, 13])
        self.assertEqual(outcome["summary"]["failed"], 0)


class UserModeTest(unittest.TestCase):
    def test_example_input(self):
        lines = ["0 1 0", "1 1 1", "0 1 0",
                 "1 0 1", "0 1 0", "1 0 1",
                 "1 0 1", "0 1 0", "1 0 1"]
        outcome = quiet(npu.run_user_mode, line_reader(lines))
        self.assertEqual((outcome["score_a"], outcome["score_b"]), (1.0, 5.0))
        self.assertEqual(outcome["decision"], "B")
        self.assertEqual(outcome["perf"][0]["ops"], 9)

    def test_bad_lines_are_reprompted(self):
        lines = ["0 1", "0 1 0", "1 1 1 1", "1 1 1", "", "0 a 0", "0 1 0",
                 "1 0 1", "0 1 0", "1 0 1",
                 "0 1 0", "1 1 1", "0 1 0"]
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            outcome = npu.run_user_mode(line_reader(lines))
        self.assertEqual(outcome["decision"], "A")
        self.assertEqual(buffer.getvalue().count("입력 형식 오류"), 4)

    def test_tie_is_undecided(self):
        lines = ["0.1 0.1 0.1"] * 3 + ["0 0 0", "0 0.9 0", "0 0 0"] + ["1 1 1"] * 3
        outcome = quiet(npu.run_user_mode, line_reader(lines))
        self.assertEqual(outcome["decision"], npu.UNDECIDED)

    def test_generator_shortcut(self):
        outcome = quiet(npu.run_user_mode, line_reader(["cross", "x", "+"]))
        self.assertEqual(outcome["decision"], "A")

    def test_eof_exits_cleanly(self):
        # 입력 도중 EOF(Ctrl+D / 파이프 종료)가 와도 traceback 없이 종료 코드 1
        with mock.patch("builtins.input", side_effect=["0 1 0", EOFError]):
            self.assertEqual(quiet(npu.main, ["--mode", "1"]), 1)

    def test_menu_rejects_invalid_choice(self):
        with mock.patch("builtins.input", side_effect=["9", "abc", "2"]):
            self.assertEqual(quiet(npu.choose_mode), "2")


class MainTest(unittest.TestCase):
    def test_main_mode2_returns_zero(self):
        self.assertEqual(quiet(npu.main, ["--mode", "2", "--data", DATA_PATH]), 0)


if __name__ == "__main__":
    unittest.main()
