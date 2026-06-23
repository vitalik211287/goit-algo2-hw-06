"""Task 2. Compare exact unique IP counting with HyperLogLog.

Usage:
    python task_2.py
    python task_2.py path/to/lms-stage-access.log

The log file is intentionally not included in the repository archive because it
can be too large for LMS upload.
"""

from __future__ import annotations

import hashlib
import ipaddress
import math
import re
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, Tuple


IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def extract_ip_from_line(line: str) -> str | None:
    """Return a valid IPv4 address from a log line or None for invalid lines."""
    match = IP_PATTERN.search(line)
    if not match:
        return None

    ip = match.group(0)
    try:
        ipaddress.ip_address(ip)
    except ValueError:
        return None

    return ip


def load_ip_addresses(file_path: str | Path) -> Iterator[str]:
    """Load valid IP addresses from a log file lazily.

    The function reads the file line by line, so it is suitable for large files.
    Incorrect lines and invalid IP addresses are ignored.
    """
    path = Path(file_path)

    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            ip = extract_ip_from_line(line)
            if ip is not None:
                yield ip


def count_unique_exact(items: Iterable[str]) -> int:
    """Count unique elements exactly using set."""
    return len(set(items))


class HyperLogLog:
    """Small HyperLogLog implementation for approximate cardinality counting."""

    def __init__(self, p: int = 14) -> None:
        if not isinstance(p, int) or not 4 <= p <= 16:
            raise ValueError("p must be an integer from 4 to 16")

        self.p = p
        self.m = 1 << p
        self.registers = [0] * self.m
        self.alpha = self._get_alpha()

    def _get_alpha(self) -> float:
        if self.m == 16:
            return 0.673
        if self.m == 32:
            return 0.697
        if self.m == 64:
            return 0.709
        return 0.7213 / (1 + 1.079 / self.m)

    @staticmethod
    def _hash_64(item: str) -> int:
        digest = hashlib.sha1(item.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False)

    @staticmethod
    def _rho(value: int, bits: int) -> int:
        """Count leading zeros in the remaining bits plus one."""
        if value == 0:
            return bits + 1
        return bits - value.bit_length() + 1

    def add(self, item: str) -> None:
        x = self._hash_64(str(item))
        register_index = x & (self.m - 1)
        remaining_bits = 64 - self.p
        w = x >> self.p
        self.registers[register_index] = max(
            self.registers[register_index],
            self._rho(w, remaining_bits),
        )

    def count(self) -> float:
        indicator = sum(2.0 ** -register for register in self.registers)
        estimate = self.alpha * self.m * self.m / indicator

        # Small-range correction: linear counting.
        empty_registers = self.registers.count(0)
        if estimate <= 2.5 * self.m and empty_registers:
            return self.m * math.log(self.m / empty_registers)

        # Large-range correction for 64-bit hashes.
        two_64 = 2.0**64
        if estimate > two_64 / 30:
            return -two_64 * math.log(1 - estimate / two_64)

        return estimate


def count_unique_hyperloglog(items: Iterable[str], p: int = 14) -> float:
    """Estimate the number of unique elements with HyperLogLog."""
    hll = HyperLogLog(p=p)
    for item in items:
        hll.add(item)
    return hll.count()


def measure_time(function, *args, **kwargs) -> Tuple[float, float]:
    """Run function and return result with execution time."""
    start = time.perf_counter()
    result = function(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def print_results_table(exact_count: int, exact_time: float, hll_count: float, hll_time: float) -> None:
    """Print comparison results as a table."""
    print("Результати порівняння:")
    print(f"{'':<28}{'Точний підрахунок':>20}{'HyperLogLog':>15}")
    print(f"{'Унікальні елементи':<28}{float(exact_count):>20.1f}{hll_count:>15.1f}")
    print(f"{'Час виконання (сек.)':<28}{exact_time:>20.6f}{hll_time:>15.6f}")


def main() -> None:
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("lms-stage-access.log")

    if not log_path.exists():
        print(f"Файл '{log_path}' не знайдено.")
        print("Покладіть lms-stage-access.log поруч зі скриптом або передайте шлях аргументом:")
        print("python task_2.py path/to/lms-stage-access.log")
        return

    # For a fair time comparison both methods receive their own lazy iterator.
    exact_count, exact_time = measure_time(count_unique_exact, load_ip_addresses(log_path))
    hll_count, hll_time = measure_time(count_unique_hyperloglog, load_ip_addresses(log_path), 14)

    print_results_table(exact_count, exact_time, hll_count, hll_time)

    if exact_count:
        error = abs(hll_count - exact_count) / exact_count * 100
        print(f"\nПохибка HyperLogLog: {error:.2f}%")


if __name__ == "__main__":
    main()
