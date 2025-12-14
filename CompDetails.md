co-located with ICSE 2026
Attending 
Tracks 
Organization 
 Search
Series 
Sign in
Sign up
 ICSE 2026 (series) /  MSR 2026 (series) /
Mining ChallengeMSR 2026
Call for Mining Challenge Papers
Call for Mining Challenge Proposals
Call for Mining Challenge Papers
Mining challenge PDF available here
AIDev dataset preprint available here

Update 2025-11-06: We updated the Zenodo link for the dataset and added a new FAQ section.

Update 2025-10-28: Paper Deadline is now Dec 23, 2025 (AoE) to avoid the holiday period.

Update 2025-10-14: The submission deadlines have been extended to provide authors with more time to prepare submissions.

Update 2025-09-29: Accepted papers will be published as Short Papers in the ACM Digital Library.

AI coding agents are rapidly reshaping the landscape of software engineering by autonomously developing features, fixing bugs, and writing tests. These tools, such as Claude Code, Cursor, Devin, GitHub Copilot, and OpenAI Codex, are no longer just assisting developers; they are becoming active AI teammates in the software development process. Yet, despite their growing presence, the research community lacks a comprehensive, large-scale understanding of how AI coding agents collaborate with developers in real-world projects: how they propose code changes, how developers respond, and what kinds of collaboration patterns emerge.

This year’s MSR Mining Challenge invites the global research community to explore unprecedented questions and present their insights using AIDev, the first large-scale, openly available dataset capturing agent-authored pull requests (Agentic-PRs) from real-world GitHub repositories:

Scale: 932,791 Agentic-PRs
Breadth: 116,211 repositories and 72,189 developers, across five AI agents (Claude Code, Cursor, Devin, GitHub Copilot, OpenAI Codex)
Depth: 33,596 curated Agentic-PRs from 2,807 popular repositories (over 100 stars), enriched with comments, reviews, commits, and related issues
Challenge
The AIDev dataset opens up rich and timely research directions around AI adoption, code quality, testing, review dynamics, risks, and human-AI collaboration in software engineering. Example research questions include (but are not limited to):

1) Adoption and Practices
i. Who adopts Coding Agents on GitHub (e.g., newcomers vs. experienced developers)?
ii. How do adoption patterns vary across repositories and ecosystems?
iii. What practices (e.g., PR size, task type, and commit granularity) correlate with the quality of Agentic-PRs?
iv. How can these practices inform concrete guidelines for developers to work with Agentic-PRs?

2) Code Patch Characteristics
i. How do Agentic-PRs change code (e.g., additions, deletions, files touched)?
ii. How consistent are their descriptions with the actual code changes?
iii. To what extent do Agentic-PRs introduce original code versus reusing existing snippets?
iv. What are the implications for maintainability?

3) Testing Behavior
i. How frequently do Coding Agents contribute tests? What types (e.g., unit, integration, end-to-end) are most common?
ii. What is the test-to-code churn ratio across ecosystems?
iii. When tests are missing in initial Agentic-PRs, do developers intervene to ensure reliable software testing (via follow-up commits or related PRs)?

4) Review Dynamics
i. What aspects of Agentic-PRs (e.g., correctness, style, security, testing) receive the most attention during review?
ii. To what extent do Coding Agents address review comments?
iii. Which comment types are challenging for agents to resolve?

5) Failure Patterns and Risks
i. What common failure patterns and code quality issues appear in Agentic-PRs? Why do they occur?
ii. How can we leverage these insights to reduce failure rates, optimize human–AI collaboration, and improve AI model training that prioritizes learning from mistakes?
iii. How well can early signals (e.g., PR description, touched paths, and patch characteristics) predict Agentic-PRs rejection or review effort?
iv. How frequently do Agentic-PRs introduce or mitigate security vulnerabilities?

We also suggest checking our preprint paper for more research questions and ideas: https://arxiv.org/abs/2507.15003

How to Participate in the Challenge
First, familiarize yourself with the AIDev dataset:

The details about the AIDev infrastructure and the data are provided in our preprint.
The dataset can be downloaded from either Hugging Face or Zenodo.
GitHub (example code & notebooks): https://github.com/SAILResearch/AI_Teammates_in_SE3.
An example Jupyter notebook demonstrating how to load and analyze the dataset is available here, you can also open it directly in Google Colab.
Use the dataset to answer your research questions, and report your findings in a four-page challenge paper that you submit to our challenge. If your paper is accepted, present your results at MSR 2026 in Rio de Janeiro, Brazil!

Submission
IMPORTANT: Accepted papers in Mining Challenge will be published as Short Papers in the ACM Digital Library. Starting 2026, all articles published by ACM will be made Open Access. This is greatly beneficial to the advancement of computer science and leads to increased usage and citation of research.

Most authors will be covered by ACM OPEN agreements by that point and will not have to pay Article Processing Charges (APC). Check if your institution participates in ACM OPEN.
Authors not covered by ACM OPEN agreements may have to pay APC; however, ACM is offering several automated and discretionary APC Waivers and Discounts.
A challenge paper should describe the results of your work by providing an introduction to the problem you address and why it is worth studying, the version of the dataset you used, the approach and tools you used, your results and their implications, and conclusions. Make sure your report highlights the contributions and the importance of your work. See also our open science policy regarding the publication of software and additional data you used for the challenge.

To ensure clarity and consistency in research submissions:

When detailing methodologies or presenting findings, authors should specify which snapshot/version of the AIDev dataset was utilized.
Given the continuous updates to the dataset, authors are reminded to be precise in their dataset references. This will help maintain transparency and ensure consistent replication of results.
All authors should use the official “ACM Primary Article Template”, as can be obtained from the ACM Proceedings Template page. LaTeX users should use the sigconf option, as well as the review (to produce line numbers for easy reference by the reviewers) and anonymous (omitting author names) options. To that end, the following LaTeX code can be placed at the start of the LaTeX document:

\documentclass[sigconf,review,anonymous]{acmart}
\acmConference[MSR 2026]{MSR '26: Proceedings of the 23rd International Conference on Mining Software Repositories}{April 2026}{Rio de Janeiro, Brazil}
Submissions to the Challenge Track can be made via the submission site by the submission deadline. We encourage authors to upload their paper info early (the PDF can be submitted later) to properly enter conflicts for anonymous reviewing. All submissions must adhere to the following requirements:

Submissions must not exceed the page limit (4 pages plus 1 additional page of references). The page limit is strict, and it will not be possible to purchase additional pages at any point in the process (including after acceptance).
Submissions must strictly conform to the ACM formatting instructions. Alterations of spacing, font size, and other changes that deviate from the instructions may result in desk rejection without further review.
Submissions must not reveal the authors’ identities. The authors must make every effort to honor the double-anonymous review process. In particular, the authors’ names must be omitted from the submission and references to their prior work should be in the third person. Further advice, guidance, and explanation about the double-anonymous review process can be found in the Q&A page for ICSE 2026.
Submissions should consider the ethical implications of the research conducted within a separate section before the conclusion.
The official publication date is the date the proceedings are made available in the ACM or IEEE Digital Libraries. This date may be up to two weeks prior to the first day of the ICSE 2026. The official publication date affects the deadline for any patent filings related to published work.
Purchases of additional pages in the proceedings are not allowed.
Any submission that does not comply with these requirements is likely to be desk rejected by the PC Chairs without further review. In addition, by submitting to the MSR Challenge Track, the authors acknowledge that they are aware of and agree to be bound by the following policies:

The ACM Policy and Procedures on Plagiarism and the IEEE Plagiarism FAQ. In particular, papers submitted to MSR 2026 must not have been published elsewhere and must not be under review or submitted for review elsewhere whilst under consideration for MSR 2026. Contravention of this concurrent submission policy will be deemed a serious breach of scientific ethics, and appropriate action will be taken in all such cases (including immediate rejection and reporting of the incident to ACM/IEEE). To check for double submission and plagiarism issues, the chairs reserve the right to (1) share the list of submissions with the PC Chairs of other conferences with overlapping review periods and (2) use external plagiarism detection software, under contract to the ACM or IEEE, to detect violations of these policies.
The authorship policy of the ACM and the authorship policy of the IEEE.
Upon notification of acceptance, all authors of accepted papers will be asked to fill a copyright form and will receive further instructions for preparing the camera-ready version of their papers. At least one author of each paper is expected to register and present the paper at the MSR 2026 conference. All accepted contributions will be published in the electronic proceedings of the conference.

The AIDev dataset can be cited as:

@article{li2025aidev,
title={{The Rise of AI Teammates in Software Engineering (SE) 3.0: How Autonomous Coding Agents Are Reshaping Software Engineering}}, 
author={Li, Hao and Zhang, Haoxiang and Hassan, Ahmed E.},
journal={arXiv preprint arXiv:2507.15003},
year={2025}
}
A preprint is available online: https://arxiv.org/abs/2507.15003

Submission Site
Papers must be submitted through HotCRP: https://msr2026-challenge.hotcrp.com/

Important Dates (AoE)
Abstract Deadline: Dec 18, 2025 (Optional, but encouraged to help us plan the review process)
Paper Deadline: Dec 23, 2025
Author Notification: Jan 15, 2026
Camera Ready Deadline: Jan 23, 2026
Open Science Policy
Openness in science is key to fostering progress via transparency, reproducibility and replicability. Our steering principle is that all research output should be accessible to the public and that empirical studies should be reproducible. In particular, we actively support the adoption of open data and open source principles. To increase reproducibility and replicability, we encourage all contributing authors to disclose:

the source code of the software they used to retrieve and analyze the data
the (anonymized and curated) empirical data they retrieved in addition to the AIDev dataset
a document with instructions for other researchers describing how to reproduce or replicate the results
Already upon submission, authors can privately share their anonymized data and software on archives such as Zenodo or Figshare (tutorial available here). Zenodo accepts up to 50GB per dataset (more upon request). There is no need to use Dropbox or Google Drive. After acceptance, data and software should be made public so that they receive a DOI and become citable. Zenodo and Figshare accounts can easily be linked with GitHub repositories to automatically archive software releases. In the unlikely case that authors need to upload terabytes of data, Archive.org may be used.

We recognise that anonymizing artifacts such as source code is more difficult than preserving anonymity in a paper. We ask authors to take a best effort approach to not reveal their identities. We will also ask reviewers to avoid trying to identify authors by looking at commit histories and other such information that is not easily anonymized. Authors wanting to share GitHub repositories may want to look into using https://anonymous.4open.science/ which is an open source tool that helps you to quickly double-blind your repository.

We encourage authors to self-archive pre- and postprints of their papers in open, preserved repositories such as arXiv.org. This is legal and allowed by all major publishers including ACM and IEEE and it lets anybody in the world reach your paper. Note that you are usually not allowed to self-archive the PDF of the published article (that is, the publisher proof or the Digital Library version). Please note that the success of the open science initiative depends on the willingness (and possibilities) of authors to disclose their data and that all submissions will undergo the same review process independent of whether or not they disclose their analysis code or data. We encourage authors who cannot disclose industrial or otherwise non-public data, for instance due to non-disclosure agreements, to provide an explicit (short) statement in the paper.

Best Mining Challenge Paper Award
As mentioned above, all submissions will undergo the same review process independent of whether or not they disclose their analysis code or data. However, only accepted papers for which code and data are available on preserved archives, as described in the open science policy, will be considered by the program committee for the best mining challenge paper award.

Best Student Presentation Award
Like in the previous years, there will be a public voting during the conference to select the best mining challenge presentation. This award often goes to authors of compelling work who present an engaging story to the audience. Only students can compete for this award.

FAQ
Q1. Can we augment AIDev with additional data for the challenge?
Yes. You are welcome and encouraged to “bring your own data” (BYOD) by integrating the AIDev dataset with information from other public, readily available sources (e.g., GitHub REST/GraphQL APIs, repository clones, ecosystem registries). Please document all sources and extraction steps. We urge participants to thoroughly consider the ethical implications of merging the AIDev dataset with other sources. The share or use of personally identifiable information (PII) is strictly prohibited.

Q2. Some PRs seem to have missing patch content. What happened and what should we do?
The dataset has been updated to include all available patches provided by GitHub API. However, the GitHub API may omit content for large patches. If you need the exact patch for these cases, you may need to clone the source repositories and obtain the commit diffs locally. When using patches, verify and comply with the original repository licenses.

Q3. Where is the data dictionary/table?
A data dictionary is available here: https://huggingface.co/datasets/hao-li/AIDev/blob/main/data_table.md

Q4. What do the commit_stats_additions/deletions vs. additions/deletions fields mean in pr_commit_details?
commit_stats_additions, commit_stats_deletions: Totals per commit (sum across all files touched in that commit).
additions, deletions: Per-file counts within that commit (for the specific file row).
Important Dates AoE (UTC-12h)
Thu 18 Dec 2025
Abstract Deadline [Papers]
Tue 23 Dec 2025
Paper Deadline [Papers]
Thu 15 Jan 2026
Author Notification [Papers]
Fri 23 Jan 2026
Camera Ready [Papers]
Mon 15 Sep 2025
Call for Challenge Papers Published [Challenge]
Thu 28 Aug 2025
Notification [Challenge]
Wed 20 Aug 2025
Deadline for Proposals [Challenge]
Mining Challenge - Program Committee
Hao Li
Hao LiMining Challenge Co-Chair
Queen's University
Canada
Haoxiang Zhang
Haoxiang ZhangMining Challenge Co-Chair
Queen's University
Canada
Ahmad Abdellatif
Ahmad AbdellatifCommittee Member
University of Calgary
Canada
An Ran Chen
An Ran ChenCommittee Member
University of Alberta
Canada
Zishuo Ding
Zishuo DingCommittee Member
The Hong Kong University of Science and Technology (Guangzhou)
China
Eduardo Figueiredo
Eduardo FigueiredoCommittee Member
Federal University of Minas Gerais
Brazil
Keheliya Gallaba
Keheliya GallabaCommittee Member
Centre for Software Excellence, Huawei Canada
Canada
Fatemeh Hendijani Fard
Fatemeh Hendijani FardCommittee Member
University of British Columbia, Okanagan
Canada
Yintong Huo
Yintong HuoCommittee Member
Singapore Management University, Singapore
Singapore
Yutaro Kashiwa
Yutaro KashiwaCommittee Member
Nara Institute of Science and Technology
Japan
Masanari Kondo
Masanari KondoCommittee Member
Kyushu University
Japan
Raula Gaikovina Kula
Raula Gaikovina KulaCommittee Member
The University of Osaka
Japan
Zhenhao Li
Zhenhao LiCommittee Member
York University
Canada
Jirat Pasuksmit
Jirat PasuksmitCommittee Member
Atlassian
Australia
Nimmi Rashinika Weeraddana
Nimmi Rashinika WeeraddanaCommittee Member
University of Calgary
Canada
Brittany Reid
Brittany ReidCommittee Member
Nara Institute of Science and Technology
Japan
Wannita Takerngsaksiri
Wannita TakerngsaksiriCommittee Member
Applied Artificial Intelligence Initiative, Deakin University
Australia
 MSR 2026
 contact form
using conf.researchr.org (v1.72.1)
 Support page
Tracks
Technical Papers
Industry Track
Data and Tool Showcase Track
FOSS Award
Junior PC
MSR Awards
Mining Challenge
Registered Reports
Tutorials
Vision and Reflection
Attending
Venue: Windsor Convention Center and Hotels
Registration
Visa and Travel Authorization Information for ICSE 2026 and its Co-Located Events
Official Travel Services
Social Events
Sustainability at ICSE 2026Sign Up