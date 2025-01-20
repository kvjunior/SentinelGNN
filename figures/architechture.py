+----------------------------------- SentinelGNN Architecture -----------------------------------+
|                                                                                               |
|  +----------------+     +-------------------+     +--------------------+     +----------------+|
|  | Input          |     | Feature           |     | Neural             |     | Anomaly        ||
|  | Processing     | --> | Engineering       | --> | Architecture       | --> | Detection      ||
|  +----------------+     +-------------------+     +--------------------+     +----------------+|
|         |                       |                          |                        |         |
|         v                       v                          v                        v         |
| +--------------+    +--------------------+    +----------------------+    +-----------------+ |
| |Raw Data      |    |Feature Extraction  |    |Temporal Processing   |    |Detection Engine | |
| |Processing    |    |Components          |    |Pipeline              |    |Components       | |
| |              |    |                    |    |                      |    |                 | |
| | * Data Load  |    | * Node Features    |    | * Edge Sampling      |    | * GMM Module    | |
| | * Validation |    |   - Account Props  |    |   - Time Windows     |    |   - Fitting     | |
| | * Cleaning   |    |   - Balance Hist   |    |   - Freq. Adaptive   |    |   - Scoring     | |
| | * Formatting |    |   - Smart Contract |    |   - Pattern Preserve |    |                 | |
| |              |    |                    |    |                      |    | * Energy Score  | |
| | * Blockchain |    | * Edge Features    |    | * Dual Stream GNN    |    |   - Threshold   | |
| |   Interface  |    |   - Transaction    |    |   - Structural Path  |    |   - Calibration | |
| |              |    |   - Gas & Time     |    |   - Temporal Path    |    |                 | |
| |              |    |   - Value Data     |    |   - Fusion Layer     |    | * Alert System  | |
| +--------------+    |                    |    |                      |    |   - Prioritize  | |
|         ^           | * Temporal Features|    | * Attention Mechanism|    |   - Classify    | |
|         |           |   - Block Info     |    |   - Multi-head      |    |   - Report      | |
|         |           |   - Market Data    |    |   - Scale-Adaptive  |    |                 | |
|    +----------+     |   - Network Stats  |    |   - Cross-temporal  |    +-----------------+ |
|    |Input Data|     +--------------------+    +----------------------+            |           |
|    |          |              |                          |                        v           |
|    | * Nodes  |              v                          v                 +--------------+   |
|    | * Edges  |    +--------------------+    +----------------------+     |Output Layer  |   |
|    | * Times  |    |Feature Integration |    |Model Components      |     |              |   |
|    | * Values |    |                    |    |                      |     | * Anomaly    |   |
|    +----------+    | * Normalization    |    | * GINE Convolution  |     |   Scores     |   |
|                    | * Embedding        |    | * Batch Norm        |     | * Confidence |   |
|                    | * Aggregation      |    | * Dropout          |     | * Categories |   |
|                    +--------------------+    +----------------------+     +--------------+   |
|                                                                                             |
+---------------------------------------------------------------------------------------------+