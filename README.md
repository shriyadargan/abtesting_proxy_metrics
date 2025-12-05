# abtesting_proxy_metrics
When running many A/B tests with weak treatment effects, naive regression-based methods for estimating relationships between metrics produce biased results due to correlated user-level noise. This project implements three sophisticated methods (JIVE, LIML, TC) that provide unbiased estimates using a synthetic dataset.
