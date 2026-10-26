### Fixed

- The benchmark reads a number with no digit before the decimal point (`.5`) as 0.5, not as 5, so `is_correct`
  accepts such an answer.
