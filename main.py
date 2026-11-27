#!/usr/bin/env python3
"""
Mini NPU Simulator
==================
MAC(Multiply-Accumulate) 연산으로 십자가(Cross) / X 패턴을 판별하는 콘솔 시뮬레이터.

- 외부 라이브러리 없이 표준 라이브러리만 사용 (Python 3.8+)
- 모드 1: 3×3 필터 2개 + 패턴 직접 입력
- 모드 2: data.json 일괄 분석 (라벨 정규화 + 스키마 검증 + PASS/FAIL)
- 모드 3: 패턴 생성기 (보너스)
"""

import math
import gc
import time
import argparse
import os
import sys
import json
import re

# ---------------------------------------------------------------------------
# 상수 / 공통 예외
# ---------------------------------------------------------------------------
LABEL_CROSS = "Cross"
LABEL_X = "X"
UNDECIDED = "UNDECIDED"


class SchemaError(ValueError):
    """입력 데이터의 구조/형식이 기대와 다를 때 사용하는 예외."""


def to_number(value, where=""):
    """JSON/입력 값을 float으로 바꾼다. bool, 문자열, NaN/Infinity는 거부한다."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"숫자가 아닌 값 {value!r} {where}".strip())
    number = float(value)
    if not math.isfinite(number):
        raise SchemaError(f"유한하지 않은 값 {value!r} {where}".strip())
    return number


def format_number(value):
    """행렬 출력용 짧은 숫자 표기 (1.0 -> 1, 0.3 -> 0.3)."""
    return f"{value:g}"


# ---------------------------------------------------------------------------
# 데이터 구조: N×N 정사각 행렬
# ---------------------------------------------------------------------------
class Matrix:
    """N×N 패턴/필터 저장소. get/set 으로 특정 위치 값을 읽고 쓴다."""

    def __init__(self, size, fill=0.0):
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise SchemaError(f"행렬 크기는 1 이상의 정수여야 합니다: {size!r}")
        self.size = size
        self.data = [[float(fill) for _ in range(size)] for _ in range(size)]

    @classmethod
    def from_rows(cls, rows):
        """2차원 리스트를 검증(정사각형, 숫자)한 뒤 Matrix로 바꾼다."""
        if not isinstance(rows, list) or not rows:
            raise SchemaError("비어 있지 않은 2차원 배열이 아닙니다")
        n = len(rows)
        matrix = cls(n)
        for r, row in enumerate(rows):
            if not isinstance(row, list):
                raise SchemaError(f"{r + 1}행이 배열이 아닙니다: {row!r}")
            if len(row) != n:
                raise SchemaError(
                    f"행/열 개수 불일치: {n}행인데 {r + 1}행의 열 개수가 {len(row)}개"
                )
            for c, value in enumerate(row):
                matrix.set(r, c, value)
        return matrix

    def _check_index(self, r, c):
        if not (0 <= r < self.size and 0 <= c < self.size):
            raise IndexError(
                f"범위를 벗어난 위치 ({r}, {c}) — 크기 {self.size}×{self.size}"
            )

    def get(self, r, c):
        """(r, c) 위치의 값을 읽는다."""
        self._check_index(r, c)
        return self.data[r][c]

    def set(self, r, c, value):
        """(r, c) 위치에 값을 저장한다."""
        self._check_index(r, c)
        self.data[r][c] = to_number(value, where=f"({r + 1}행 {c + 1}열)")

    def format(self, indent="  "):
        return "\n".join(
            indent + " ".join(format_number(v) for v in row) for row in self.data
        )

    def __repr__(self):
        return f"Matrix({self.size}×{self.size})"


# ---------------------------------------------------------------------------
# MAC 연산 · 판정 · 시간 측정
# ---------------------------------------------------------------------------
EPSILON = 1e-9         # 동점 판정 허용오차: |a - b| < EPSILON 이면 동점
EPSILON_TEXT = "1e-9"
REPEAT = 10            # 성능 측정 반복 횟수 (최소 기준 10회)


def mac(pattern, filt):
    """같은 위치끼리 곱한 값을 모두 누적한다. 곱셈 횟수 = N²."""
    if pattern.size != filt.size:
        raise SchemaError(
            f"크기 불일치: 패턴 {pattern.size}×{pattern.size}, "
            f"필터 {filt.size}×{filt.size}"
        )
    n = pattern.size
    a = pattern.data
    b = filt.data
    total = 0.0
    for r in range(n):
        for c in range(n):
            total += a[r][c] * b[r][c]
    return total


def is_tie(score_a, score_b, eps=EPSILON):
    return abs(score_a - score_b) < eps


def decide(score_a, score_b, label_a, label_b, eps=EPSILON):
    """점수가 큰 쪽의 라벨을 돌려준다. 차이가 eps 미만이면 UNDECIDED."""
    if is_tie(score_a, score_b, eps):
        return UNDECIDED
    return label_a if score_a > score_b else label_b


def measure_avg_ms(func, *args, repeat=REPEAT):
    """func(*args)를 repeat회 호출한 평균 시간(ms).

    - 첫 호출(워밍업)은 측정에서 뺀다
    - 측정 중에는 GC를 꺼서(timeit 과 같은 방식) 가비지 컬렉션으로 시간이 튀는 것을 줄인다
    """
    func(*args)
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        elapsed = 0.0
        for _ in range(repeat):
            start = time.perf_counter()
            func(*args)
            elapsed += time.perf_counter() - start
    finally:
        if gc_was_enabled:
            gc.enable()
    return elapsed / repeat * 1000.0

# ---------------------------------------------------------------------------
# 라벨 정규화 (표준 라벨: Cross, X)
# ---------------------------------------------------------------------------
# data.json 은 expected 에 '+' / 'x', 필터 키에 'cross' / 'x' 를 쓴다.
# 내부 비교와 출력은 모두 표준 라벨(Cross/X)로 통일한다.
LABEL_ALIASES = {
    "+": LABEL_CROSS,
    "cross": LABEL_CROSS,
    "x": LABEL_X,
}


def normalize_label(raw):
    """여러 표기를 표준 라벨로 바꾼다. 해석할 수 없으면 None."""
    if not isinstance(raw, str):
        return None
    return LABEL_ALIASES.get(raw.strip().lower())


# ---------------------------------------------------------------------------
# 공통 출력 도우미 / 모드 등록
# ---------------------------------------------------------------------------
LINE = "#" + "-" * 40
MODES = {}   # 메뉴 번호 -> (이름, 실행 함수)


def register_mode(key, name, runner):
    """메뉴에 모드를 등록한다. runner 는 명령행 인자(args)를 받는 함수."""
    MODES[key] = (name, runner)


def section(title):
    print()
    print(LINE)
    print(f"# {title}")
    print(LINE)


def format_score(score):
    """점수를 repr 그대로 출력해 부동소수점 오차가 눈에 보이게 한다."""
    return repr(score)


# ---------------------------------------------------------------------------
# 모드 1: 사용자 입력 (3×3)
# ---------------------------------------------------------------------------
MODE1_SIZE = 3
FORMAT_ERROR = "입력 형식 오류: 각 줄에 {n}개의 숫자를 공백으로 구분해 입력하세요."


def parse_row(line, n):
    """한 줄 문자열을 숫자 n개로 파싱한다. 실패하면 SchemaError."""
    tokens = line.split()
    if not tokens:
        raise SchemaError("빈 줄이 입력되었습니다 (행 누락)")
    if len(tokens) != n:
        raise SchemaError(f"열 개수 불일치: {len(tokens)}개 입력됨")
    values = []
    for token in tokens:
        try:
            number = float(token)
        except ValueError:
            raise SchemaError(f"숫자 파싱 실패: {token!r}") from None
        values.append(to_number(number))
    return values


def read_matrix(title, n=MODE1_SIZE, reader=None):
    """n줄을 한 줄씩 입력받아 Matrix로 저장한다. 잘못된 줄은 그 줄만 다시 입력받는다."""
    reader = reader or input   # 호출할 때 input 을 찾아야 테스트에서 바꿔 끼울 수 있다
    print(f"{title} ({n}줄 입력, 공백 구분)")
    rows = []
    while len(rows) < n:
        line = reader()
        try:
            rows.append(parse_row(line, n))
        except SchemaError as e:
            print(f"  {FORMAT_ERROR.format(n=n)} ({e})")
            print(f"  → {len(rows) + 1}번째 줄을 다시 입력하세요.")
    return Matrix.from_rows(rows)


def print_decision_ab(score_a, score_b):
    decision = decide(score_a, score_b, "A", "B")
    if decision == UNDECIDED:
        print(f"판정: 판정 불가 (|A-B| < {EPSILON_TEXT})")
    else:
        print(f"판정: {decision}")
    return decision


def run_user_mode(reader=None):
    section("[1] 필터 입력")
    filter_a = read_matrix("필터 A", reader=reader)
    print()
    filter_b = read_matrix("필터 B", reader=reader)
    print()
    print("✓ 필터 저장 완료")
    print("필터 A:")
    print(filter_a.format())
    print("필터 B:")
    print(filter_b.format())

    section("[2] 패턴 입력")
    pattern = read_matrix("패턴", reader=reader)

    section("[3] MAC 결과")
    score_a = mac(pattern, filter_a)
    score_b = mac(pattern, filter_b)
    avg_ms = measure_avg_ms(lambda: (mac(pattern, filter_a), mac(pattern, filter_b)))
    print(f"A 점수: {format_score(score_a)}")
    print(f"B 점수: {format_score(score_b)}")
    print(f"|A-B|: {abs(score_a - score_b):.3e}")
    print(f"연산 시간(평균/{REPEAT}회, A·B 필터 MAC 2회): {avg_ms:.4f} ms")
    decision = print_decision_ab(score_a, score_b)
    return {"score_a": score_a, "score_b": score_b, "decision": decision}


register_mode("1", "사용자 입력 (3x3)", lambda args: run_user_mode())


# ---------------------------------------------------------------------------
# 모드 2: data.json 로드 / 스키마 검증 / 판정
# ---------------------------------------------------------------------------
FILTER_KEY_RE = re.compile(r"^size_(\d+)$")
PATTERN_KEY_RE = re.compile(r"^size_(\d+)_(\d+)$")

# 실패 원인 분류 (데이터/스키마 vs 수치 비교 vs 판정 로직)
CAUSE_SCHEMA = "데이터/스키마"
CAUSE_NUMERIC = "수치 비교·동점"
CAUSE_LOGIC = "판정 불일치"


class CaseFailure(Exception):
    """케이스 단위 FAIL. 프로그램을 멈추지 않고 사유와 원인 분류를 남긴다."""

    def __init__(self, cause, message):
        super().__init__(message)
        self.cause = cause


class CaseResult:
    def __init__(self, case_id):
        self.case_id = case_id
        self.cross_score = None
        self.x_score = None
        self.decision = None
        self.expected = None
        self.raw_expected = None
        self.passed = False
        self.cause = None
        self.reason = ""


def load_json(path):
    """(data, 오류 메시지)를 돌려준다. 실패해도 예외를 밖으로 던지지 않는다."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f), None
    except FileNotFoundError:
        return None, f"파일을 찾을 수 없습니다: {path}"
    except json.JSONDecodeError as e:
        return None, f"JSON 문법 오류: {e.msg} (line {e.lineno}, col {e.colno})"
    except (OSError, UnicodeDecodeError) as e:
        return None, f"파일을 읽을 수 없습니다: {e}"


def parse_filter_entry(n, entry):
    """size_N 필터 항목 → {Cross: Matrix, X: Matrix}. 필터 키도 라벨 정규화한다."""
    if not isinstance(entry, dict):
        raise SchemaError("필터 항목이 객체(dict)가 아닙니다")
    pair = {}
    for raw_label, rows in entry.items():
        label = normalize_label(raw_label)
        if label is None:
            raise SchemaError(f"알 수 없는 필터 라벨 {raw_label!r}")
        if label in pair:
            raise SchemaError(f"필터 라벨 중복: {raw_label!r} → {label}")
        try:
            matrix = Matrix.from_rows(rows)
        except SchemaError as e:
            raise SchemaError(f"{label} 필터 형식 오류: {e}") from None
        if matrix.size != n:
            raise SchemaError(
                f"{label} 필터 크기 불일치: 키는 {n}×{n}, 실제 {matrix.size}×{matrix.size}"
            )
        pair[label] = matrix
    missing = [label for label in (LABEL_CROSS, LABEL_X) if label not in pair]
    if missing:
        raise SchemaError(f"필터 누락: {', '.join(missing)}")
    return pair


def load_filters(raw):
    """filters 섹션 → ({N: {Cross, X}}, {키 또는 N: 오류 사유})"""
    filters, errors = {}, {}
    if not isinstance(raw, dict):
        errors["filters"] = "'filters' 항목이 없거나 객체(dict)가 아닙니다"
        return filters, errors
    for key, entry in raw.items():
        match = FILTER_KEY_RE.match(str(key))
        if not match:
            errors[key] = f"필터 키 형식 오류 (size_N 형식이어야 함): {key!r}"
            continue
        n = int(match.group(1))
        try:
            filters[n] = parse_filter_entry(n, entry)
        except SchemaError as e:
            errors[n] = str(e)
    return filters, errors


def _evaluate(result, entry, filters, filter_errors):
    match = PATTERN_KEY_RE.match(str(result.case_id))
    if not match:
        raise CaseFailure(CAUSE_SCHEMA, "패턴 키 형식 오류 (size_N_idx 형식이어야 함)")
    n = int(match.group(1))
    if not isinstance(entry, dict):
        raise CaseFailure(CAUSE_SCHEMA, "패턴 항목이 객체(dict)가 아닙니다")
    result.raw_expected = entry.get("expected")
    result.expected = normalize_label(result.raw_expected)
    if "input" not in entry:
        raise CaseFailure(CAUSE_SCHEMA, "'input' 필드 누락")
    try:
        pattern = Matrix.from_rows(entry["input"])
    except SchemaError as e:
        raise CaseFailure(CAUSE_SCHEMA, f"input 형식 오류: {e}") from None
    if pattern.size != n:
        raise CaseFailure(
            CAUSE_SCHEMA,
            f"크기 불일치: 키는 {n}×{n}인데 input은 {pattern.size}×{pattern.size}",
        )
    if n not in filters:
        reason = filter_errors.get(n)
        message = f"size_{n} 필터 사용 불가: {reason}" if reason else f"size_{n} 필터가 없습니다"
        raise CaseFailure(CAUSE_SCHEMA, message)

    pair = filters[n]
    for label, filt in pair.items():
        if filt.size != pattern.size:
            raise CaseFailure(
                CAUSE_SCHEMA,
                f"필터/패턴 크기 불일치: {label} 필터 {filt.size}×{filt.size}, "
                f"패턴 {pattern.size}×{pattern.size}",
            )

    result.cross_score = mac(pattern, pair[LABEL_CROSS])
    result.x_score = mac(pattern, pair[LABEL_X])
    result.decision = decide(result.cross_score, result.x_score, LABEL_CROSS, LABEL_X)

    if "expected" not in entry:
        raise CaseFailure(CAUSE_SCHEMA, "'expected' 필드 누락")
    if result.expected is None:
        raise CaseFailure(CAUSE_SCHEMA, f"expected 라벨 해석 불가: {result.raw_expected!r}")
    if result.decision == UNDECIDED:
        diff = abs(result.cross_score - result.x_score)
        raise CaseFailure(
            CAUSE_NUMERIC,
            f"동점(UNDECIDED) 처리 규칙에 따라 FAIL (|Cross-X|={diff:.2e} < {EPSILON_TEXT})",
        )
    if result.decision != result.expected:
        raise CaseFailure(CAUSE_LOGIC, f"판정 {result.decision} ≠ expected {result.expected}")
    result.passed = True


def evaluate_case(case_id, entry, filters, filter_errors):
    """패턴 1건을 평가한다. 어떤 오류가 나도 CaseResult(FAIL)로 돌려준다."""
    result = CaseResult(case_id)
    try:
        _evaluate(result, entry, filters, filter_errors)
    except CaseFailure as e:
        result.cause, result.reason = e.cause, str(e)
    except Exception as e:  # 마지막 안전망: 프로그램이 비정상 종료되지 않도록
        result.cause = CAUSE_SCHEMA
        result.reason = f"예기치 못한 오류: {type(e).__name__}: {e}"
    return result


def print_case(result):
    print(f"--- {result.case_id} ---")
    if result.cross_score is None:
        print("Cross 점수: 계산 불가")
        print("X 점수: 계산 불가")
    else:
        print(f"Cross 점수: {format_score(result.cross_score)}")
        print(f"X 점수: {format_score(result.x_score)}")
    decision = result.decision or "-"
    if result.expected:
        expected = result.expected
    elif result.raw_expected is not None:
        expected = f"{result.raw_expected!r}(해석 불가)"
    else:
        expected = "-"
    status = "PASS" if result.passed else f"FAIL [{result.cause}]"
    print(f"판정: {decision} | expected: {expected} | {status}")
    if not result.passed:
        print(f"  사유: {result.reason}")


def print_filter_status(filters, filter_errors):
    for n in sorted(filters):
        print(f"✓ size_{n:<3} 필터 로드 완료 ({', '.join(filters[n])})")
    for key, reason in filter_errors.items():
        name = f"size_{key}" if isinstance(key, int) else str(key)
        print(f"✗ {name:<8} 필터 로드 실패: {reason}")


def analyze_patterns(patterns, filters, filter_errors):
    results = []
    if not isinstance(patterns, dict):
        print("✗ 'patterns' 항목이 없거나 객체(dict)가 아닙니다.")
        return results
    for case_id, entry in patterns.items():
        result = evaluate_case(case_id, entry, filters, filter_errors)
        print_case(result)
        results.append(result)
    return results


def run_json_mode(path):
    section("[1] 필터 로드")
    data, error = load_json(path)
    if error:
        print(f"✗ {error}")
        print("  data.json 위치를 확인한 뒤 다시 실행하세요. (--data 옵션으로 경로 지정 가능)")
        return None
    if not isinstance(data, dict):
        print("✗ 최상위 구조가 객체(dict)가 아닙니다.")
        return None

    filters, filter_errors = load_filters(data.get("filters"))
    print_filter_status(filters, filter_errors)

    section("[2] 패턴 분석 (라벨 정규화 적용)")
    results = analyze_patterns(data.get("patterns"), filters, filter_errors)
    return {"filters": filters, "filter_errors": filter_errors, "results": results}


register_mode("2", "data.json 분석", lambda args: run_json_mode(args.data))


# ===========================================================================
# 실행 흐름 — 이 구역은 항상 파일 맨 아래에 둔다 (새 코드는 이 구분선 '위'에 추가)
# ===========================================================================
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")


def choose_mode(reader=None):
    reader = reader or input
    print()
    print("[모드 선택]")
    for key in sorted(MODES):
        print(f"{key}. {MODES[key][0]}")
    while True:
        print("선택: ", end="", flush=True)
        choice = reader().strip()
        if choice in MODES:
            return choice
        print(f"  잘못된 선택입니다. {', '.join(sorted(MODES))} 중에서 입력하세요.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Mini NPU Simulator")
    parser.add_argument("--mode", choices=sorted(MODES), help="모드 바로 실행 (메뉴 생략)")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="data.json 경로")
    return parser.parse_args(argv)


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        # 콘솔 인코딩이 ✓, × 같은 문자를 지원하지 않아도 비정상 종료하지 않도록
        sys.stdout.reconfigure(errors="replace")
    args = parse_args(argv)
    print("=== Mini NPU Simulator ===")
    try:
        mode = args.mode or choose_mode()
        _, runner = MODES[mode]
        runner(args)
    except (EOFError, KeyboardInterrupt):
        print("\n입력이 종료되어 프로그램을 마칩니다.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())