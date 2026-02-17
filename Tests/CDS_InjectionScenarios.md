# CDS (Credit Delta Single) Outlier Injection Scenarios

8 outlier injection scenarios to be implemented for the CDS dataset.

## Common Rules

- Injections should not overwrite each other: if a trade/date was selected for one scenario, it should not be used for other injections.
- Scenarios are sorted starting with simpler cases. Injections should be added in the same order to avoid conflicts.
- Injections should be done only to dates in OOS (Out-of-Sample) records.
- RecordType field should be updated with the scenario name for changed records.
- Injections should be applied to a provided number of trades selected randomly.
- In most scenarios, the number of days is absolute. In one scenario, it is relative (given % of trades).
- Injections are applied to days selected according to the conditions below.

## Day Selection Conditions

- **Random day**: A random day is selected.
- **Consecutive days**: A few consecutive days are selected at a random point in time.
- **Last consecutive days**: Same as above but not random; should end at the end of OOS.
- **First consecutive days**: Same as above but not random; should start at the beginning of OOS.
- **Excluded**: This scenario is not applicable for the given trade type and must be omitted.

## Calculation Logic

Adjusted values for injections are calculated by provided formulas, which are determined by Trade Type.

### Formula Variables

- **Δ'** - New value to replace the original
- **Δ** - Original value
- **k** - Additive coefficient (e.g., ∈ {3, 6, 12}); used as a parameter for the injection function and defined at test level
- **α** - Relative coefficient (e.g., ∈ {0.25, 0.5, 1.0, 2.0}); used as a parameter for the injection function and defined at test level
- **MAD_global** - Credit Delta MAD for all observations in the dataset
- **MAD_tt** - Credit Delta MAD for the given trade type
- **Q01_tt and Q99_tt** - First and last quantiles for Credit Delta values for the given trade type
- **Scale_tt** - Multiplicative factor = MAD_tt. If MAD_tt is impossible to calculate (NaN), use MAD_global instead
- **Range_tt** - Relative multiplicator = Q99_tt − Q01_tt
- **Scale_trade** - Credit Delta MAD for the given trade
- **Range_trade** - Q99 - Q01 for the given trade
- **t** - Day variable
- **t-1** - Previous day

## Implementation Notes & Clarifications

### Formula & Variables
- **T in CD_Drift**: The variable T in the CD_Drift formula equals the number of days specified in the '# Days' column (e.g., T=15 for Basis, T=10 for Basket).
- **Index Formula Design**: Index trade types use α·Range formulas (instead of k·Scale used by other types) because Credit Delta values for Index trades are very small and require different scaling logic.
- **Scale Fallback**: If MAD_tt cannot be calculated (is NaN), fall back to MAD_global. If other scale/range values cannot be calculated, fail explicitly with an error.

### Trade & Date Selection
- **Trade Reuse**: Trades **can be reused** across different scenarios. The non-overlapping constraint applies only within a single scenario's trade selection.
- **Percentage Trades**: In scenarios like CD_TradeTypeWide_Shock with '50%' or '100%' as the # Trades value, the percentage refers to the number of trades **of that specific trade type** in the OOS dataset, not the global trade count.
- **Each Trade's Injection**: For consecutive day scenarios, each selected trade receives injections over one contiguous block of days (e.g., in CD_ClusterShock_3d, each trade is injected over 3 consecutive days as a single block).

### Recording & Execution
- **RecordType Update**: When updating the RecordType field for injected records, use just the scenario name (e.g., "CD_Drift", not "CD_Drift_Trade3").
- **Trade Counts**: The '# Trades' values are fixed requirements, not ranges or minimums.
- **Scenario Ordering**: While scenarios are ordered to avoid conflicts, the strict order is not mandatory. The goal is to avoid overlapping trades/dates across scenarios; applying them in the provided order is the simplest way to achieve this.
- **Randomness**: All random selections (trades, days) must be seeded/reproducible for testing purposes.

### Coverage
- **Exclusions**: Only Tranche in CD_Drift is excluded. All other trade type/scenario combinations must be implemented.

## Scenarios

### 1. CD_Drift
**Description**: Linear drift over T days (Gradual deterioration / creeping issue)

| Trade Type | # Trades | # Days | Formula | Condition |
|---|---|---|---|---|
| Basis | 1 | 15 | For i=0..T−1: Δ(t+i)' = Δ(t+i) + (i/(T−1))·k·Scale_trade | Last consecutive days |
| Basket | 1 | 10 | For i=0..T−1: Δ(t+i)' = Δ(t+i) + (i/(T−1))·k·Scale_trade | Last consecutive days |
| Index | 1 | 15 | Δ(t+i)' = Δ(t+i) + (i/(T−1))·α·Range_trade | Last consecutive days |
| SingleName | 1 | 15 | For i=0..T−1: Δ(t+i)' = Δ(t+i) + (i/(T−1))·k·Scale_trade | Last consecutive days |
| Tranche | 1 | 0 | For i=0..T−1: Δ(t+i)' = Δ(t+i) + (i/(T−1))·k·Scale_trade | Excluded |

### 2. CD_StaleValue
**Description**: Stuck values (Stuck output / frozen feed)

| Trade Type | # Trades | # Days | Formula | Condition |
|---|---|---|---|---|
| Basis | 1 | 5 | Δ(t)' = Δ(t−1) | First consecutive days |
| Basket | 1 | 5 | Δ(t)' = Δ(t−1) | First consecutive days |
| Index | 1 | 5 | Δ(t)' = Δ(t−1) | First consecutive days |
| SingleName | 1 | 5 | Δ(t)' = Δ(t−1) | First consecutive days |
| Tranche | 1 | 5 | Δ(t)' = Δ(t−1) | First consecutive days |

### 3. CD_ClusterShock_3d
**Description**: Short feed issue (fallback curve) for a few days

| Trade Type | # Trades | # Days | Formula | Condition |
|---|---|---|---|---|
| Basis | 1 | 3 | For i=0..2: Δ(t+i)' = Δ(t+i) + k·Scale_tt | Consecutive days |
| Basket | 1 | 3 | For i=0..2: Δ(t+i)' = Δ(t+i) + k·Scale_tt | Consecutive days |
| Index | 1 | 3 | Δ(t+i)' = Δ(t+i) + α·Range_tt | Consecutive days |
| SingleName | 1 | 3 | For i=0..2: Δ(t+i)' = Δ(t+i) + k·Scale_tt | Consecutive days |
| Tranche | 1 | 3 | For i=0..2: Δ(t+i)' = Δ(t+i) + k·Scale_tt | Consecutive days |

### 4. CD_TradeTypeWide_Shock
**Description**: Systemic curve roll/config change affecting a whole book

| Trade Type | # Trades | # Days | Formula | Condition |
|---|---|---|---|---|
| Basis | 50% | 1 | Δ' = Δ + k·Scale_tt | Random day |
| Basket | 100% | 1 | Δ' = Δ + k·Scale_tt | Random day |
| Index | 50% | 1 | Δ' = Δ + α·Range_tt | Random day |
| SingleName | 50% | 1 | Δ' = Δ + k·Scale_tt | Random day |
| Tranche | 100% | 1 | Δ' = Δ + k·Scale_tt | Random day |

### 5. CD_PointShock
**Description**: One-day bad snapshot / transient pricer glitch

| Trade Type | # Trades | # Days | Formula | Condition |
|---|---|---|---|---|
| Basis | 3 | 1 | Δ' = Δ + k·Scale_tt | Random day |
| Basket | 1 | 1 | Δ' = Δ + k·Scale_tt | Random day |
| Index | 2 | 1 | Δ' = Δ + α·Range_tt | Random day |
| SingleName | 3 | 1 | Δ' = Δ + k·Scale_tt | Random day |
| Tranche | 1 | 1 | Δ' = Δ + k·Scale_tt | Random day |

### 6. CD_SignFlip
**Description**: Sign convention bug

| Trade Type | # Trades | # Days | Formula | Condition |
|---|---|---|---|---|
| Basis | 3 | 1 | Δ' = −Δ | Random day |
| Basket | 1 | 1 | Δ' = −Δ | Random day |
| Index | 2 | 1 | Δ' = −Δ | Random day |
| SingleName | 3 | 1 | Δ' = −Δ | Random day |
| Tranche | 1 | 1 | Δ' = −Δ | Random day |

### 7. CD_ScaleError
**Description**: Unit / decimal / notional scale bug

| Trade Type | # Trades | # Days | Formula | Condition |
|---|---|---|---|---|
| Basis | 3 | 1 | Δ' = 100·Δ | Random day |
| Basket | 1 | 1 | Δ' = 100·Δ | Random day |
| Index | 2 | 1 | Δ' = 1000·Δ | Random day |
| SingleName | 3 | 1 | Δ' = 100·Δ | Random day |
| Tranche | 1 | 1 | Δ' = 100·Δ | Random day |

### 8. CD_SuddenZero
**Description**: Valuation error leading to 0 value

| Trade Type | # Trades | # Days | Formula | Condition |
|---|---|---|---|---|
| Basis | 3 | 1 | Δ' = 0 | Random day |
| Basket | 1 | 1 | Δ' = 0 | Random day |
| Index | 2 | 1 | Δ' = 0 | Random day |
| SingleName | 3 | 1 | Δ' = 0 | Random day |
| Tranche | 1 | 1 | Δ' = 0 | Random day |



