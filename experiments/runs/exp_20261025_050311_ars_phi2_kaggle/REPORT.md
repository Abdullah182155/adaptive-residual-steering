# ARS Benchmark Report



- created: 2026-10-25T05:03:11.029498+00:00 | git: b51a81e

- model: microsoft/phi-2



## Deployment contract



- s*_fixed: `[13, 15, 22]` | theta*_abs: `1.01`

- gate adopted: `passthrough` | envelope cap: `0.0041589075699448586`



## Primary: 0-shot (n=100)



### accuracy



| Arm | Strict acc (%) | Lenient acc (%) |

|---|---|---|

| Baseline | 0.00 | 33.00 |

| Fixed-control | 0.00 | 41.00 |

| Router-only | 0.00 | 37.00 |

| Full ARS | 0.00 | 37.00 |

| Raw ARS | 0.00 | 41.00 |



### paired strict



| Arm | Delta vs baseline (pp) | 95% CI | (b, c) | McNemar p | Holm p |

|---|---|---|---|---|---|

| Fixed-control | +0.00 | [+0.00, +0.00] | (0, 0) | 1.0000 | 1.0000 |

| Router-only | +0.00 | [+0.00, +0.00] | (0, 0) | 1.0000 | 1.0000 |

| Full ARS | +0.00 | [+0.00, +0.00] | (0, 0) | 1.0000 | 1.0000 |

| Raw ARS | +0.00 | [+0.00, +0.00] | (0, 0) | 1.0000 | 1.0000 |



### paired lenient



| Arm | Delta vs baseline (pp) | 95% CI | (b, c) | McNemar p | Holm p |

|---|---|---|---|---|---|

| Fixed-control | +8.00 | [+2.00, +15.00] | (10, 2) | 0.0386 | 0.1157 |

| Router-only | +4.00 | [-1.00, +10.00] | (6, 2) | 0.2891 | 0.5781 |

| Full ARS | +4.00 | [-1.00, +10.00] | (6, 2) | 0.2891 | 0.5781 |

| Raw ARS | +8.00 | [+2.00, +14.00] | (9, 1) | 0.0215 | 0.0859 |



### routing



| Arm | Gate mode | Route entropy (bits) | Abstain rate | K distribution | Layer frequencies | Guard trip rate |

|---|---|---|---|---|---|---|

| Baseline | none | -0.000 | 1.000 | 0:1.00 1:0.00 2:0.00 3:0.00 4:0.00 | - | 0.0000 |

| Fixed-control | const | -0.000 | 0.000 | 0:0.00 1:0.00 2:0.00 3:1.00 4:0.00 | L13:1.00 L15:1.00 L22:1.00 | 0.0000 |

| Router-only | const | 0.995 | 0.000 | 0:0.00 1:0.00 2:0.46 3:0.54 4:0.00 | L13:0.46 L15:0.54 L18:0.54 L22:1.00 | 0.0000 |

| Full ARS | const | 0.995 | 0.000 | 0:0.00 1:0.00 2:0.46 3:0.54 4:0.00 | L13:0.46 L15:0.54 L18:0.54 L22:1.00 | 0.0000 |

| Raw ARS | learned | 0.995 | 0.000 | 0:0.00 1:0.00 2:0.46 3:0.54 4:0.00 | L13:0.46 L15:0.54 L18:0.54 L22:1.00 | 0.0000 |



Non-inferiority margin: 1.0 pp (CI lower bound > -margin).

- Fixed-control: delta +0.00 pp, non-inferior=True, significant (Holm)=False

- Router-only: delta +0.00 pp, non-inferior=True, significant (Holm)=False

- Full ARS: delta +0.00 pp, non-inferior=True, significant (Holm)=False

- Raw ARS: delta +0.00 pp, non-inferior=True, significant (Holm)=False



## Primary: 4-shot (n=100)



### accuracy



| Arm | Strict acc (%) | Lenient acc (%) |

|---|---|---|

| Baseline | 31.00 | 42.00 |

| Fixed-control | 35.00 | 47.00 |

| Router-only | 35.00 | 47.00 |

| Full ARS | 35.00 | 47.00 |

| Raw ARS | 33.00 | 48.00 |



### paired strict



| Arm | Delta vs baseline (pp) | 95% CI | (b, c) | McNemar p | Holm p |

|---|---|---|---|---|---|

| Fixed-control | +4.00 | [-3.00, +11.00] | (8, 4) | 0.3877 | 1.0000 |

| Router-only | +4.00 | [-3.00, +11.00] | (9, 5) | 0.4240 | 1.0000 |

| Full ARS | +4.00 | [-3.00, +11.00] | (9, 5) | 0.4240 | 1.0000 |

| Raw ARS | +2.00 | [-5.00, +9.00] | (8, 6) | 0.7905 | 1.0000 |



### paired lenient



| Arm | Delta vs baseline (pp) | 95% CI | (b, c) | McNemar p | Holm p |

|---|---|---|---|---|---|

| Fixed-control | +5.00 | [-2.00, +12.00] | (9, 4) | 0.2668 | 0.8005 |

| Router-only | +5.00 | [-2.00, +12.00] | (9, 4) | 0.2668 | 0.8005 |

| Full ARS | +5.00 | [-2.00, +12.00] | (9, 4) | 0.2668 | 0.8005 |

| Raw ARS | +6.00 | [+0.00, +13.00] | (9, 3) | 0.1460 | 0.5840 |



### routing



| Arm | Gate mode | Route entropy (bits) | Abstain rate | K distribution | Layer frequencies | Guard trip rate |

|---|---|---|---|---|---|---|

| Baseline | none | -0.000 | 1.000 | 0:1.00 1:0.00 2:0.00 3:0.00 4:0.00 | - | 0.0000 |

| Fixed-control | const | -0.000 | 0.000 | 0:0.00 1:0.00 2:0.00 3:1.00 4:0.00 | L13:1.00 L15:1.00 L22:1.00 | 0.0000 |

| Router-only | const | -0.000 | 0.000 | 0:0.00 1:0.00 2:1.00 3:0.00 4:0.00 | L13:1.00 L22:1.00 | 0.0000 |

| Full ARS | const | -0.000 | 0.000 | 0:0.00 1:0.00 2:1.00 3:0.00 4:0.00 | L13:1.00 L22:1.00 | 0.0000 |

| Raw ARS | learned | -0.000 | 0.000 | 0:0.00 1:0.00 2:1.00 3:0.00 4:0.00 | L13:1.00 L22:1.00 | 0.0000 |



Non-inferiority margin: 1.0 pp (CI lower bound > -margin).

- Fixed-control: delta +4.00 pp, non-inferior=False, significant (Holm)=False

- Router-only: delta +4.00 pp, non-inferior=False, significant (Holm)=False

- Full ARS: delta +4.00 pp, non-inferior=False, significant (Holm)=False

- Raw ARS: delta +2.00 pp, non-inferior=False, significant (Holm)=False



## Primary: 8-shot (n=100)



### accuracy



| Arm | Strict acc (%) | Lenient acc (%) |

|---|---|---|

| Baseline | 32.00 | 41.00 |

| Fixed-control | 46.00 | 51.00 |

| Router-only | 39.00 | 46.00 |

| Full ARS | 39.00 | 46.00 |

| Raw ARS | 39.00 | 45.00 |



### paired strict



| Arm | Delta vs baseline (pp) | 95% CI | (b, c) | McNemar p | Holm p |

|---|---|---|---|---|---|

| Fixed-control | +14.00 | [+7.00, +22.00] | (15, 1) | 5.19e-04 | 0.0021 |

| Router-only | +7.00 | [+2.00, +13.00] | (8, 1) | 0.0391 | 0.1172 |

| Full ARS | +7.00 | [+2.00, +13.00] | (8, 1) | 0.0391 | 0.1172 |

| Raw ARS | +7.00 | [+2.00, +13.00] | (8, 1) | 0.0391 | 0.1172 |



### paired lenient



| Arm | Delta vs baseline (pp) | 95% CI | (b, c) | McNemar p | Holm p |

|---|---|---|---|---|---|

| Fixed-control | +10.00 | [+2.00, +18.00] | (14, 4) | 0.0309 | 0.1235 |

| Router-only | +5.00 | [-1.00, +12.00] | (8, 3) | 0.2266 | 0.6797 |

| Full ARS | +5.00 | [-1.00, +12.00] | (8, 3) | 0.2266 | 0.6797 |

| Raw ARS | +4.00 | [-3.00, +11.00] | (8, 4) | 0.3877 | 0.6797 |



### routing



| Arm | Gate mode | Route entropy (bits) | Abstain rate | K distribution | Layer frequencies | Guard trip rate |

|---|---|---|---|---|---|---|

| Baseline | none | -0.000 | 1.000 | 0:1.00 1:0.00 2:0.00 3:0.00 4:0.00 | - | 0.0000 |

| Fixed-control | const | -0.000 | 0.000 | 0:0.00 1:0.00 2:0.00 3:1.00 4:0.00 | L13:1.00 L15:1.00 L22:1.00 | 0.0000 |

| Router-only | const | -0.000 | 0.000 | 0:0.00 1:0.00 2:1.00 3:0.00 4:0.00 | L13:1.00 L22:1.00 | 0.0000 |

| Full ARS | const | -0.000 | 0.000 | 0:0.00 1:0.00 2:1.00 3:0.00 4:0.00 | L13:1.00 L22:1.00 | 0.0000 |

| Raw ARS | learned | -0.000 | 0.000 | 0:0.00 1:0.00 2:1.00 3:0.00 4:0.00 | L13:1.00 L22:1.00 | 0.0000 |



Non-inferiority margin: 1.0 pp (CI lower bound > -margin).

- Fixed-control: delta +14.00 pp, non-inferior=True, significant (Holm)=True

- Router-only: delta +7.00 pp, non-inferior=True, significant (Holm)=False

- Full ARS: delta +7.00 pp, non-inferior=True, significant (Holm)=False

- Raw ARS: delta +7.00 pp, non-inferior=True, significant (Holm)=False
