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