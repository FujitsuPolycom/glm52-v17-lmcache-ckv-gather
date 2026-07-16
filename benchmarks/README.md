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
| 258,048 tokens, pre-fix | 97.015 s | 2,660 tok/s | Request completed; later CUDA fault |
| 258,048 tokens, exp.2 #1 | 90.869 s | 2,840 tok/s | Pass + 8/8 follow-ups |
| 258,048 tokens, exp.2 #2 | 91.583 s | 2,818 tok/s | Pass + 8/8 follow-ups |
| 258,048 tokens, post-reboot | 90.423 s | 2,854 tok/s | Pass + 4/4 follow-ups |

Each exp.2 run drained 2,016 LMCache objects totaling 9,562,226,688 bytes. The
three-run mean TTFT was 90.958 seconds, and all 20 immediate unique follow-ups
returned HTTP 200 without a CUDA error or engine restart.
