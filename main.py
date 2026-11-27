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