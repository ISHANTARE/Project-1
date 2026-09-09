# Base Papers - Literature Review Repository

This directory contains the full-text research papers referenced in **Review 2** (`Review_2_PPT.pptx`, Slide 5, Slide 6, and Slide 14) and the Project-I Technical Report.

---

## 📚 Summary Table of Base Papers

| # | Paper Title | Authors | Venue / Year | Local PDF | Direct Web Link / Source |
|---|---|---|---|---|---|
| **1** | **LinCE: A Centralized Benchmark for Linguistic Code-switching Evaluation** | Gustavo Aguilar, Sudipta Kar, Thamar Solorio | *LREC 2020* | [`1_Aguilar_LREC_2020.pdf`](file:///c:/Users/Admin/VIT_Projects/PROJECT-1/base_papers/1_Aguilar_LREC_2020.pdf) | [ACL Anthology](https://aclanthology.org/2020.lrec-1.223.pdf) \| [arXiv:2005.04322](https://arxiv.org/abs/2005.04322) |
| **2** | **GLUECoS: An Evaluation Benchmark for Code-Switched NLP** | Simran Khanuja, Sandipan Dandapat, Anirudh Srinivasan, Sunayana Sitaram, Monojit Choudhury | *ACL 2020* | [`2_Khanuja_ACL_2020.pdf`](file:///c:/Users/Admin/VIT_Projects/PROJECT-1/base_papers/2_Khanuja_ACL_2020.pdf) | [ACL Anthology](https://aclanthology.org/2020.acl-main.329.pdf) \| [arXiv:2004.12376](https://arxiv.org/abs/2004.12376) |
| **3** | **Language Modeling for Code-Mixing: The Role of Linguistic Theory** | Adithya Pratapa, Gayatri Bhat, Monojit Choudhury, Sunayana Sitaram, Sandipan Dandapat, Kalika Bali | *NAACL-HLT 2018* | [`3_Pratapa_NAACL_HLT_2018.pdf`](file:///c:/Users/Admin/VIT_Projects/PROJECT-1/base_papers/3_Pratapa_NAACL_HLT_2018.pdf) | [ACL Anthology](https://aclanthology.org/N18-1139.pdf) |
| **4** | **"Why Should I Trust You?": Explaining the Predictions of Any Classifier (LIME)** | Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin | *ACM SIGKDD 2016* | [`4_Ribeiro_KDD_2016.pdf`](file:///c:/Users/Admin/VIT_Projects/PROJECT-1/base_papers/4_Ribeiro_KDD_2016.pdf) | [arXiv:1602.04938](https://arxiv.org/pdf/1602.04938.pdf) \| [ACM DL](https://dl.acm.org/doi/10.1145/2939672.2939778) |
| **5** | **Lexically Constrained Decoding for Sequence Generation Using Grid Beam Search** | Chris Hokamp, Qun Liu | *ACL 2017* | [`5_Hokamp_ACL_2017.pdf`](file:///c:/Users/Admin/VIT_Projects/PROJECT-1/base_papers/5_Hokamp_ACL_2017.pdf) | [ACL Anthology](https://aclanthology.org/P17-1141.pdf) |
| **6** | **Fast Lexically Constrained Decoding with Dynamic Beam Allocation for Neural Machine Translation** | Matt Post, David Vilar | *NAACL-HLT 2018* | [`6_Post_NAACL_2018.pdf`](file:///c:/Users/Admin/VIT_Projects/PROJECT-1/base_papers/6_Post_NAACL_2018.pdf) | [ACL Anthology](https://aclanthology.org/N18-1119.pdf) \| [arXiv:1804.06609](https://arxiv.org/abs/1804.06609) |
| **7** | **Language Identification and Analysis of Code-Switched Social Media Text** | Deepthi Mave, Suraj Maharjan, Thamar Solorio | *NAACL-HLT 2018* | [`7_Mave_NAACL_2018.pdf`](file:///c:/Users/Admin/VIT_Projects/PROJECT-1/base_papers/7_Mave_NAACL_2018.pdf) | [ACL Anthology](https://aclanthology.org/N18-3006.pdf) |
| **8a** | **HinGE: A Dataset for Generation and Evaluation of Code-Mixed Hinglish Text** | Vivek Srivastava, Mayank Singh | *ACL-IJCNLP 2021* | [`Srivastava_HinGE_ACL_2021.pdf`](file:///c:/Users/Admin/VIT_Projects/PROJECT-1/base_papers/Srivastava_HinGE_ACL_2021.pdf) | [ACL Anthology](https://aclanthology.org/2021.acl-long.339.pdf) \| [arXiv:2107.03666](https://arxiv.org/abs/2107.03666) |
| **8b** | **PHINC: A Parallel Hinglish Social Media Code-Mixed Corpus for Machine Translation** | Vivek Srivastava, Mayank Singh | *W-NUT @ EMNLP 2020* | [`Srivastava_PHINC_EMNLP_2020.pdf`](file:///c:/Users/Admin/VIT_Projects/PROJECT-1/base_papers/Srivastava_PHINC_EMNLP_2020.pdf) | [ACL Anthology](https://aclanthology.org/2020.wnut-1.6.pdf) \| [arXiv:2009.11710](https://arxiv.org/abs/2009.11710) |
| **Bonus** | **Toward a Typological Model of Code-Switching** | Shana Poplack | *Linguistics (1980)* | *Paywalled / Institutional* | [De Gruyter (DOI: 10.1515/ling.1980.18.7-8.581)](https://doi.org/10.1515/ling.1980.18.7-8.581) \| [Semantic Scholar](https://www.semanticscholar.org/paper/Sometimes-I'll-start-a-sentence-in-English-y-en-a-Poplack/70624c9c64ef5ddb5a59d9f5daeebcceef9ec2ff) |

---

## 🎯 Role in LAPS Project Architecture

1. **Aguilar et al. (LinCE, 2020)**: Grounding dataset for token-level language identification, providing benchmark data for Spanish-English and Hindi-English code-mixing.
2. **Khanuja et al. (GLUECoS, 2020)**: Highlights degradation of multilingual foundation models (mBERT, XLM-RoBERTa) when processing code-switched text, motivating the need for pre-generation guardrails.
3. **Pratapa et al. (2018)**: Investigates linguistic constraints (Equivalence Constraint Theory, Matrix Language Frame); demonstrates limitations of purely syntactic rule-based approaches for dynamic loanword borrowing.
4. **Ribeiro et al. (LIME, 2016)**: Foundation of explainable machine learning; inspires LAPS's token-level feature attribution and tabular explainability dashboard.
5. **Hokamp & Liu (2017)**: Formulates Grid Beam Search (GBS) for lexically constrained decoding in sequence-to-sequence translation models.
6. **Post & Vilar (2018)**: Introduces Dynamic Beam Allocation (DBA) to reduce constrained decoding computational overhead from exponential grid search down to constant-time allocations.
7. **Mave et al. (2018)**: Token-level biLSTM language identification on informal social media text; establishes that upstream LID is necessary but insufficient without loanword borrowing propensity scoring.
8. **Srivastava & Singh (2020, 2021)**: Seminal empirical studies on Hinglish code-mixed machine translation and generation, demonstrating that standard models struggle with informal Hinglish and fail on over-translation.
