## 4/19

sample random subsets of sequences from each temperature
  likelihood
  diversity
    distinct-N
    SBERT (all-MiniLM-L6-v2)
    DPP (be careful with diagonals, normalization, etc)
  aggregation:
    BST (RLHFlow/Llama3.1-8B-ORM-Mistral-Data)
    MAJ
    MBR (self-BLEU + SBERT)

## 4/20: 802687fa1864a9037d5682635cd427ecf69dc394

with boot=1 and only partial data for t=1.0

| sample_size | maj_correct | best_correct | sem_mbr_correct | lex_mbr_correct |
|-------------|-------------|--------------|-----------------|-----------------|
| 4           | 0.463415    | 0.536585     | 0.452575        | 0.471545        |
| 8           | 0.547425    | 0.550136     | 0.493225        | 0.506775        |
| 16          | 0.590786    | 0.544715     | 0.482385        | 0.560976        |
| 32          | 0.623306    | 0.542005     | 0.463415        | 0.574526        |

| temp | maj_correct | best_correct | sem_mbr_correct | lex_mbr_correct |
|------|-------------|--------------|-----------------|-----------------|
| 0.7  | 0.58375     | 0.57         | 0.50125         | 0.57            |
| 1    | 0.523669    | 0.511834     | 0.439349        | 0.47929         |

|              |   maj_correct |   best_correct |   sem_mbr_correct |   lex_mbr_correct |
|--------------|---------------|----------------|-------------------|-------------------|
| cosine_sim   |      0.328344 |      0.191655  |         0.230321  |          0.294955 |
| dist_n       |     -0.258973 |     -0.151995  |        -0.156531  |         -0.255537 |
| dpp          |     -0.186473 |     -0.0747052 |        -0.0849543 |         -0.160496 |
| qual_dpp     |     -0.127362 |     -0.0219812 |        -0.0295819 |         -0.100558 |
| avg_log_prob |      0.35952  |      0.315766  |         0.307895  |          0.354459 |

note that this aggregates across settings, we really have |temperatures| x |samples| settings to evaluate

diversity and quality are entangled but even controlling for this we see that diversity negatively impacts aggregated accuracy

the plausible conclusion is that math problems may have a relatively unimodal output distribution so any diversity we see just comes from incorrect solutions—indeed at t=0.7, we see that increased diversity (DPP score) is negatively correlated with the number of correct solutions in the batch

| batch size | pearson's | spearman's |
|------------|-----------|------------|
| 4          | -0.4362   | -0.4297    |
| 8          | -0.4897   | -0.4753    |
| 16         | -0.5545   | -0.5333    |
| 32         | -0.5916   | -0.5852    |
