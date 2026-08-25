## Part 3: Project

### Part A: Supabase Setup

The Supabase project is set up successfully, including the project URL, anon API key, and the weather_raw and weather_enriched tables.

### Part B: Cloud Cost Analysis

Scenario A: Lightweight compute - t3.micro, $.0104 * 160 = $1.66 / month

Scenario B: Heavy analytic workload

* Amazon EC2 (p3.2xlarge): $3.06 * 730 = $2233.80
* Amazon RDS (db.m5.large): $.171 * 730 = $124.83
* Amazon S3 Standard (1 TB): 1024 GB * $0.023 per GB = $23.55

### Reflection

Scenario A costs approximately $1.66 per month, while Scenario B costs $2382.18 per month.  The large difference was surprising, especially because the p3.2xlarge GPU instance accounts for most of Scenario B's cost.  While exploring the AWS Pricing Calculator, I found that changing the instance type, utilization, and deployment options can significantly affect the overall monthly cost.  The comparison shows that a GPU instance is worth the additional cost when a workload requires intensive GPU processing, such as machine-learning or large-scale analytics, but for lighter workloads, a less expensive CPU-based instance is a much more practical choice.