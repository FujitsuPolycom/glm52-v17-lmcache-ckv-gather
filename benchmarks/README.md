# Selected validation results

Hardware: 4x RTX PRO 6000 Blackwell 96GB, TP4/DCP4/MTP3.

## CKV-gather prefill

| Context | Stock v17 | CKV gather | Gain |
| --- | ---: | ---: | ---: |
| 8K | 2,140 tok/s | 3,450 tok/s | +61.2% |
| 64K | 2,108 tok/s | 3,224 tok/s | +52.9% |
| 128K | 2,084 tok/s | 3,032 tok/s | +45.5% |

The matched 12-cell decode matrix was effectively flat at -1.23% geometric
mean, consistent with decode remaining on the stock v17 path.

## Long-context IPC diagnosis

| Prompt | TTFT | Prefill | Result |
| --- | ---: | ---: | --- |
| 196,610 tokens | 68.937 s | 2,852 tok/s | Request and transition passed |
| 258,048 tokens | 97.015 s | 2,660 tok/s | Request completed; later CUDA fault during store-future polling |

The second result is a reproducer, not a pass. The staged 84-chunk multi-future
fix has not yet been rerun on GPU hardware.

