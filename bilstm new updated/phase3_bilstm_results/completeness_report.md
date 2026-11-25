# Training-period completeness analysis

| Training period | n | Bin width | MAXC | MAXC interval | GFT | GFT interval | GFT success |
|---|---|---|---|---|---|---|---|
| before2000 | 1824 | 0.1 | 4.7 | [4.50, 6.00] | 4.40 | [4.30, 6.20] | 87% |
| before2000 | 1824 | 0.2 | 4.6000000000000005 | [4.60, 4.80] | 4.60 | [4.40, 6.20] | 76% |
| before2000 | 1824 | 0.3 | 4.5 | [4.50, 4.80] | 4.50 | [4.50, 6.00] | 98% |
| 2000_2009 | 222 | 0.1 | 4.0 | [3.90, 4.20] | 4.00 | [3.90, 4.01] | 98% |
| 2000_2009 | 222 | 0.2 | 4.2 | [3.80, 4.20] | not estimable | [4.00, 4.20] | 20% |
| 2000_2009 | 222 | 0.3 | 4.2 | [3.90, 4.20] | 4.20 | [3.90, 4.20] | 99% |
| 2010_2014 | 735 | 0.1 | 3.3000000000000003 | [3.20, 3.50] | 4.20 | [3.10, 4.20] | 98% |
| 2010_2014 | 735 | 0.2 | 3.6 | [3.40, 3.60] | 3.60 | [3.20, 4.20] | 94% |
| 2010_2014 | 735 | 0.3 | 3.3 | [3.30, 3.60] | 4.20 | [3.30, 4.20] | 100% |
| 2015_2019 | 669 | 0.1 | 3.6 | [3.50, 4.00] | 3.90 | [3.80, 4.00] | 94% |
| 2015_2019 | 669 | 0.2 | 3.6 | [3.60, 4.00] | 4.00 | [3.80, 4.40] | 68% |
| 2015_2019 | 669 | 0.3 | 3.5999999999999996 | [3.60, 3.90] | 3.90 | [3.90, 4.50] | 97% |
| 2020_2022 | 740 | 0.1 | 3.4000000000000004 | [2.90, 3.80] | 3.80 | [3.40, 4.00] | 100% |
| 2020_2022 | 740 | 0.2 | 3.8000000000000003 | [3.20, 3.80] | 3.60 | [3.60, 3.83] | 98% |
| 2020_2022 | 740 | 0.3 | 3.5999999999999996 | [3.30, 3.90] | 3.90 | [3.60, 4.20] | 100% |

Conditional catalog-based estimates only: original retained ML magnitudes, training-period records before2023, no threshold-based data filtering. MAXC is the modal bin; GFT selects the first95% fit threshold (90% fallback), requiring100 tail events. Bin centers are integer multiples of width; deterministic half-bin ties go upward. These are adaptations, not a claimed exact reproduction of the reference's search grid or bootstrap.

Intervals are95% percentile ranges from200 moving-block resamples of20 consecutive events. GFT intervals condition on successful fits; success fractions quantify failures. They do not include magnitude calibration, spatial heterogeneity or unknown reporting bias and are not formal detection-completeness confidence bounds. Small samples or failed fits are marked unavailable. Bin-width variation is a required sensitivity check; rounded ML heaping can move modes.

Method reference: [Mignan & Woessner (2012), sections6.1–6.2 and appendix](https://www.corssa.org/export/sites/corssa/.galleries/articles-pdf/Mignan-Woessner-2012-CORSSA-Magnitude-of-completeness.pdf_2063069299.pdf). MAXC can underestimate completeness for heterogeneous catalogs. No current2026 Mc estimate or causal network-completeness conclusion is made.
