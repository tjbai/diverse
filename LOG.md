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

|           | maj_correct | best_correct | sem_mbr_correct | lex_mbr_correct |
|-----------|-------------|--------------|-----------------|-----------------|
| (4, 0.7)  | 0.5         | 0.575        | 0.49            | 0.515           |
| (8, 0.7)  | 0.575       | 0.575        | 0.505           | 0.555           |
| (16, 0.7) | 0.62        | 0.575        | 0.52            | 0.595           |
| (32, 0.7) | 0.64        | 0.555        | 0.49            | 0.615           |

|           | maj_correct | best_correct | sem_mbr_correct | lex_mbr_correct |
|-----------|-------------|--------------|-----------------|-----------------|
| (4, 1.0)  | 0.420118    | 0.491124     | 0.408284        | 0.420118        |
| (8, 1.0)  | 0.514793    | 0.52071      | 0.47929         | 0.449704        |
| (16, 1.0) | 0.556213    | 0.508876     | 0.43787         | 0.52071         |
| (32, 1.0) | 0.60355     | 0.526627     | 0.431953        | 0.526627        |

notice how columns 2 and 3 don't benfit very heavily from additional samples

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

note that this aggregates across settings, we really have |temperatures| x |samples| settings to evaluate.
these should also really be _pairwise_ correlation numbers because diversity and likelihood are correlated.
even with this, the correlations are largely negative between diversity and accuracy.

the plausible conclusion is that math problems may have a relatively unimodal output distribution so any diversity we see just comes from incorrect solutions—indeed at t=0.7, we see that increased diversity (DPP score) is negatively correlated with the number of correct solutions in the batch

| batch size | pearson's | spearman's |
|------------|-----------|------------|
| 4          | -0.4362   | -0.4297    |
| 8          | -0.4897   | -0.4753    |
| 16         | -0.5545   | -0.5333    |
| 32         | -0.5916   | -0.5852    |

## 4/20

it also seems plausible that our RM is not very good, which would help explain the `best_correct` results above.
(really I should train/fine-tune one myself on our samples)

## 4/20

how much can we prune best-of-N batches before seeing performance degradation?

i want to do something like k-means clusters but in an arbitrarily-defined metric space and compare those tradeoffs vs. naive subsampling. k-medoids seems to be the solution here because we want the centroids to be actual points in the set.

what we really want is to pick out all the modes in the (reward * likelihood) space, but our proxy is finding some metric where nearby sequences are likely to have the same reward

what if we use the encodings from our ORM? need to figure out whether this ORM is actually good though... scaling above would say it's not.

## 4/21: 6218d96c8772a553a659964348dfb4081b6198cb

|   sample_size |   maj_correct |   best_correct |   sem_mbr_correct |   lex_mbr_correct |
|---------------|---------------|----------------|-------------------|-------------------|
|             4 |      0.501375 |       0.562    |          0.497375 |          0.520375 |
|             8 |      0.5695   |       0.56875  |          0.50375  |          0.54075  |
|            16 |      0.603625 |       0.568875 |          0.507    |          0.56     |
|            32 |      0.62375  |       0.563875 |          0.50625  |          0.5775   |

|   temp |   maj_correct |   best_correct |   sem_mbr_correct |   lex_mbr_correct |
|--------|---------------|----------------|-------------------|-------------------|
|    0.3 |      0.57525  |       0.57975  |          0.530375 |          0.55975  |
|    0.5 |      0.598875 |       0.585625 |          0.540625 |          0.55675  |
|    0.7 |      0.585625 |       0.565375 |          0.497875 |          0.57275  |
|    1   |      0.5385   |       0.53275  |          0.4455   |          0.509375 |

| n   | temp | maj_correct | best_correct | sem_mbr_correct | lex_mbr_correct |
|-----|------|-------------|--------------|-----------------|-----------------|
| 4   | 0.3  | 0.5275      | 0.579        | 0.532           | 0.5505          |
| 4   | 0.5  | 0.534       | 0.5845       | 0.5285          | 0.534           |
| 4   | 0.7  | 0.5075      | 0.564        | 0.495           | 0.529           |
| 4   | 1.0  | 0.4365      | 0.5205       | 0.434           | 0.468           |
| 8   | 0.3  | 0.5735      | 0.5785       | 0.527           | 0.544           |
| 8   | 0.5  | 0.5975      | 0.5865       | 0.5415          | 0.565           |
| 8   | 0.7  | 0.5805      | 0.5795       | 0.4985          | 0.558           |
| 8   | 1.0  | 0.5265      | 0.5305       | 0.448           | 0.496           |
| 16  | 0.3  | 0.6         | 0.5795       | 0.5275          | 0.5645          |
| 16  | 0.5  | 0.624       | 0.5905       | 0.5425          | 0.558           |
| 16  | 0.7  | 0.6145      | 0.565        | 0.508           | 0.589           |
| 16  | 1.0  | 0.576       | 0.5405       | 0.45            | 0.5285          |
| 32  | 0.3  | 0.6         | 0.582        | 0.535           | 0.58            |
| 32  | 0.5  | 0.64        | 0.581        | 0.55            | 0.57            |
| 32  | 0.7  | 0.64        | 0.553        | 0.49            | 0.615           |
| 32  | 1.0  | 0.615       | 0.5395       | 0.45            | 0.545           |

## 4/22: 6218d96c8772a553a659964348dfb4081b6198cb

>>> tab(bucket(df, num_buckets=8, sample_size=4, metric='qual_dpp'))
|   maj_correct |   best_correct |   sem_mbr_correct |   lex_mbr_correct |       min |       max |      mean |   count |
|---------------|----------------|-------------------|-------------------|-----------|-----------|-----------|---------|
|      0        |       0        |          0        |          0        | -20.9764  | -18.798   | -19.7907  |       3 |
|      0        |       0        |          0        |          0        | -18.3201  | -18.0404  | -18.1802  |       2 |
|      0.285714 |       0.428571 |          0.285714 |          0.357143 | -16.4768  | -14.4102  | -15.1642  |      14 |
|      0.276596 |       0.340426 |          0.234043 |          0.276596 | -14.2472  | -12.1649  | -13.1623  |      47 |
|      0.591463 |       0.664634 |          0.634146 |          0.652439 | -12.1348  |  -9.95855 | -10.8818  |     164 |
|      0.652413 |       0.682957 |          0.638363 |          0.662187 |  -9.95304 |  -7.75302 |  -8.44445 |    1637 |
|      0.497964 |       0.562296 |          0.493282 |          0.517508 |  -7.75243 |  -5.54907 |  -6.66668 |    4912 |
|      0.313115 |       0.396721 |          0.320492 |          0.336885 |  -5.54901 |  -3.34514 |  -4.9875  |    1220 |

>>> tab(bucket(df, num_buckets=8, sample_size=4, metric='dpp'))
|   maj_correct |   best_correct |   sem_mbr_correct |   lex_mbr_correct |       min |       max |      mean |   count |
|---------------|----------------|-------------------|-------------------|-----------|-----------|-----------|---------|
|      1        |       0.8      |          0.8      |          1        | -14.0415  | -12.6051  | -13.3102  |       5 |
|      0.947368 |       0.947368 |          0.947368 |          0.947368 | -11.8895  | -11.028   | -11.395   |      19 |
|      0.959459 |       0.986486 |          0.972973 |          0.972973 | -10.9653  |  -9.45939 | -10.0721  |      74 |
|      0.822171 |       0.842956 |          0.815242 |          0.815242 |  -9.42085 |  -7.92014 |  -8.45205 |     433 |
|      0.700166 |       0.727223 |          0.683048 |          0.713418 |  -7.91967 |  -6.38988 |  -7.0209  |    1811 |
|      0.50647  |       0.568763 |          0.496238 |          0.523924 |  -6.38917 |  -4.85959 |  -5.60159 |    3323 |
|      0.277482 |       0.375787 |          0.291041 |          0.309443 |  -4.85865 |  -3.32901 |  -4.22383 |    2065 |
|      0.133829 |       0.193309 |          0.163569 |          0.156134 |  -3.32823 |  -1.7982  |  -2.92574 |     269 |

>>> tab(bucket(df, num_buckets=8, sample_size=4, metric='avg_log_prob'))
|   maj_correct |   best_correct |   sem_mbr_correct |   lex_mbr_correct |       min |        max |      mean |   count |
|---------------|----------------|-------------------|-------------------|-----------|------------|-----------|---------|
|      0        |       0        |          0        |          0        | -4.44985  | -3.96556   | -4.2033   |       3 |
|      0        |       0        |          0        |          0        | -3.879    | -3.59613   | -3.73756  |       2 |
|      0        |       0        |          0        |          0        | -3.16859  | -2.81572   | -2.96207  |       5 |
|      0.037037 |       0.185185 |          0.111111 |          0.111111 | -2.79119  | -2.2804    | -2.50248  |      27 |
|      0.12069  |       0.224138 |          0.137931 |          0.172414 | -2.21921  | -1.70898   | -1.95961  |      58 |
|      0.179775 |       0.247191 |          0.191011 |          0.224719 | -1.67926  | -1.15829   | -1.39358  |      89 |
|      0.121387 |       0.242775 |          0.148362 |          0.183044 | -1.15259  | -0.609072  | -0.785376 |     519 |
|      0.537755 |       0.593395 |          0.530903 |          0.552967 | -0.608525 | -0.0601202 | -0.234665 |    7297 |

tons of figures that i was trying to read like tealeaves...

realizing that this is basically just an exploration/exploitation tradeoff—arXiv:2504.13837 shows that post-RLVR models put everything on exploit while the base models are lower entropy, so they win when you sample more sequences.

the nice thing about sequential generation is that the model adaptively figures out what other things to exploit while maintaining feasibility.

## 4/22: 769d19b2c3d493e3aa2aeca7024101d2060ddce0

measure correlation between diversity and coverage (rather than some aggregated metric)—this or something like pass@k might be better moving forward because it's simple and (probably) correlates well with other things

| k  | Measure | Correlation | Partial Correlation |
|----|---------|-------------|---------------------|
| 4  | cos sim | 0.437       | 0.354               |
| 4  | dist-3  | -0.067      | 0.204               |
| 8  | cos sim | 0.508       | 0.382               |
| 8  | dist-3  | -0.356      | 0.034               |
| 16 | cos sim | 0.531       | 0.325               |
| 16 | dist-3  | -0.435      | 0.081               |
| 32 | cos sim | 0.565       | 0.355               |
| 32 | dist-3  | -0.526      | 0.010               |
