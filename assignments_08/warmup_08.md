## --- Part 1: Warmup - Cloud Concepts ---

### Cloud Concepts Q1:

Cloud computing allows you to rent computing infrastructure from a provider.
You pay only for what you use and can always scale up or down without buying
servers.

### Cloud Concepts Q2:

* Vertical scaling: making a machine bigger/powerful (adding more RAM, CPU, GPU).
Example: You might choose this when a single workload needs more
computing power.

* Horizontal scaling: more machines working in parallel.
Example: You might choose this when a workload has to be split across multiple
machines.

Scenario A: Horizontal scaling because you want to spread load across several servers.

Scenario B: Vertical scaling; you want to upgrade the resources of one machine.

Scenario C: Horizontal scaling, because multiple machines can process the files in parallel.

### Cloud Concepts Q3:

* Gmail = SaaS: You use the software through a browser, while Google manages the servers,infrastructure, and application.

* Azure Virtual Machine = IaaS: You rent servers and are responsible managing the operating system, software, and applications.

* AWS S3 (Simple Storage Service) = IaaS: It provides cloud infrastructure for storing data.
AWS manages the physical hardware.

* GitHub Codespace = PaaS:  It provides a ready to use cloud development environment.

* Snowflake = SaaS: It provides a fully managed data platform that you access without managing the underlying servers.

* Supabase = BaaS: It provides backend services such as databases and authentication so that clients don't need to build their own.

* IaaS (Infrastructure as a Service): Provide basic computing infrastructure such as virtual machines and storage.
For example, Azure Virtual Machines.
As the developer, I am responsible for managing the operating system, software, applications,
and data.

* PaaS (Platform as a Service): Provides a managed platform where I can build and run applications without managing the underlying infrastructure.
For example, GitHub Codespace.
As the developer, I am mainly responsible for my code and application, while the provider manages the infrastructure and platform.

* SaaS (Software as a Service): Is a software that is already built and provided over the internet. 
For example, Gmail.
As the developer, I am responsible mainly for using and configuring the software; the provider manages the application, infrastructure, and maintenance.

### Cloud Concepts Q4:
A fully-managed platform like Databricks or Snowflake is built on top of raw cloud. Instead of building and maintaining everything yourself with AWS or GCP services, the platform provides a more ready-to-use data environment.
Gains: easier setup and management, less infrastructure to maintain, integrated tooling, faster to get started with large-scale data processing or machine learning.
Give up: some control, flexibility and potentially higher costs.

### Cloud Concepts Q5:
1.) Small/simple workloads - Cloud costs and complexity may not be worth it.
2.) Strict security, data location, or hardware control requirements - Makes cloud use impractical.

## --- Part 2: Warmup - Cloud Landscape ---

### Cloud Landscape Q1:
1.) Amazon Web Services (AWS): the oldest and largest cloud market.
Widely used by startups, large enterprise, or a nonprofit with engineering staff.

2.) Google Cloud Platform (GCP): the strongest in data and machine learning.
Used by data-driven companies and organizations focused on machine learning.

3.) Microsoft Azure: strong in enterprise and Microsoft-based environment, making it popular with large businesses and organization already using Microsoft products.

### Cloud Landscape Q2:
1.) Access - Because it's easier for students to set up; students can create their own account within minutes without waiting for organizational approval.

2.) Better for learning - Supabase uses relational tables, which teaches database skills that are useful in many data jobs.

3.) Fits the pipeline - Supabase makes it easy to organize the raw and enriched data into separate tables and inspect each stage.

Reflection: I should choose a cloud tool based on the project's needs, ease of use, cost and how well it supports the skills and goals of the project.

### Cloud Landscape Q3:
1.) Object storage - Amazon S3.

2.) GPU compute - AWS EC2.

3.) Serverless compute - AWS Lambda.

4.) LLM API - OpenAI API

### Cloud Landscape Q4:
I could build a job-search data pipeline that collect job postings, stores them, cleans the data, and display the results in a dashboard. I could use AWS S3 for object storage and Supabase for a managed database.
Consolidating to one provider could make the project simpler to manage and integrate, but I would give up flexibility and the ability to choose the best tool for each job.
