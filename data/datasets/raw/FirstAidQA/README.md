---
language:
  - en
license: cc-by-4.0
pretty_name: FirstAidQA
size_categories:
  - "1K<n<10K"
task_categories:
  - question-answering
tags:
  - text
  - medical
  - healthcare
  - first-aid
  - emergency-response
  - synthetic-data
  - instruction-tuning
  - low-resource
  - offline-ai
  - safety-critical
  - medical-question-answering
  - medical-nlp
  - emergency-medicine
  - synthetic-medical-data
  - small-language-models
  - edge-ai
  - low-connectivity
---

# FirstAidQA: A Synthetic First-Aid and Emergency-Response Question-Answering Dataset

> **Medical safety notice:** FirstAidQA is intended for research and educational purposes. It is not a substitute for professional medical advice, emergency services, certified first-aid training, or clinical judgment. Models trained on this dataset may produce incomplete, outdated, or unsafe responses.

## Dataset Summary

FirstAidQA is an English-language synthetic question-answering dataset covering practical first-aid and emergency-response scenarios.

The current Hugging Face release contains **5,550 question-answer pairs**. It was developed to support research on instruction-tuning and fine-tuning Large Language Models (LLMs) and Small Language Models (SLMs), particularly for offline, low-bandwidth, edge-device, and resource-constrained environments.

The dataset was generated using **ChatGPT-4o-mini** through structured, role-based prompting and prompt-based in-context learning. Generation was grounded in content selected from the *New Vital First Aid: First Aid Book*, published by Vital First Aid Training Services Pty Ltd in 2019.

The associated paper was accepted at the **5th Muslims in Machine Learning Workshop (MusIML), co-located with NeurIPS 2025**.

## Important Version Note

The current Hugging Face repository contains **5,550 rows**, while version 1 of the associated paper reports **5,500 question-answer pairs**.

Users should treat the Hugging Face repository row count as the size of the currently released artifact. The maintainers intend to reconcile this discrepancy in a future dataset or paper revision.

## Associated Publication

**Title:** FirstAidQA: A Synthetic Dataset for First Aid and Emergency Response in Low-Connectivity Settings

**Authors:**

1. Mushfiqur Rahman Mushfique
2. Saiyma Sittul Muna
3. Rezwan Islam Salvi
4. Ajwad Abrar

**Affiliation:** Islamic University of Technology, Dhaka, Bangladesh

**Venue:** 5th Muslims in Machine Learning Workshop, co-located with NeurIPS 2025

**arXiv:** [arXiv:2511.01289](https://arxiv.org/abs/2511.01289)

**DOI:** [10.48550/arXiv.2511.01289](https://doi.org/10.48550/arXiv.2511.01289)

#**Hugging Face downloads:** 1,847 all time  
*Reported on 6 August 2026*
(updated 6 Aug 2026)

## Dataset Details

| Property                            | Description                                         |
| ----------------------------------- | --------------------------------------------------- |
| Dataset name                        | FirstAidQA                                          |
| Language                            | English                                             |
| Modality                            | Text                                                |
| Format                              | JSON                                                |
| Current number of rows              | 5,550                                               |
| Number of splits                    | One                                                 |
| Split name                          | `train`                                             |
| Data file                           | `firstaidqa_v1.json`                                |
| Primary task                        | Question answering                                  |
| Additional use                      | Instruction tuning and text generation              |
| Generation model                    | ChatGPT-4o-mini                                     |
| Source material                     | *New Vital First Aid: First Aid Book* (2019)        |
| License                             | Creative Commons Attribution 4.0 International      |
| Contains real patient records       | No real patient records were intentionally included |
| Personally identifiable information | None intentionally included                         |

## Dataset Coverage

The dataset covers practical and situational first-aid topics, including:

* General emergency procedures and scene safety
* DRSABCD and initial casualty assessment
* Cardiopulmonary resuscitation
* Choking, drowning, and overdose-related emergencies
* Road traffic accidents
* Moving and transporting casualties
* Spinal precautions
* First-aid equipment and improvised tools
* Bleeding and wound management
* Burns and scalds
* Fractures and soft-tissue injuries
* Head injuries
* Asthma and breathing emergencies
* Animal bites
* Temperature-related emergencies
* Family and community emergency preparedness
* Patient examination and monitoring
* Adult, pediatric, and elderly emergency scenarios

## Dataset Structure

Each record contains two string fields:

* `question`: A natural-language first-aid or emergency-response question.
* `answer`: A generated response describing relevant actions, precautions, or procedural guidance.

### Example

```json
{
  "question": "Can a human crutch be used for a person with a shoulder injury?",
  "answer": "No, the human crutch should not be used if the person has an injured arm, hand, or shoulder on the side you're supporting. In such cases, you should stand on the opposite side of the injury to avoid causing further pain or damage."
}
```

## Data Splits

The current release contains one split:

| Split   | Number of examples |
| ------- | -----------------: |
| `train` |              5,550 |

The repository does not currently provide separate validation or test sets.

Researchers conducting model evaluation should create and document their own non-overlapping validation and test partitions. Deduplication and semantic-similarity checks are recommended before constructing these partitions because multiple questions may cover closely related first-aid concepts.

## Loading the Dataset

Using the Hugging Face `datasets` library:

```python
from datasets import load_dataset

dataset = load_dataset("i-am-mushfiq/FirstAidQA")

print(dataset)
print(dataset["train"][0])
```

To access the individual fields:

```python
question = dataset["train"][0]["question"]
answer = dataset["train"][0]["answer"]

print(question)
print(answer)
```

## Dataset Creation

### Curation Rationale

FirstAidQA was created to address the limited availability of practical first-aid question-answering data suitable for training smaller, domain-focused language models.

Many existing medical QA datasets focus on clinical examinations, biomedical literature, or diagnostic reasoning. FirstAidQA instead emphasizes practical, situational, and procedural first-aid knowledge intended for research on systems operating in low-connectivity or resource-constrained environments.

### Source Material

The dataset was grounded in:

> Vital First Aid Training Services Pty Ltd. *New Vital First Aid: First Aid Book*. Australia, reprinted edition, November 2019.

Relevant sections of the source material were selected and divided into context-preserving chunks. These chunks covered first-aid procedures, casualty assessment, emergency response, equipment use, and condition-specific guidance.

### Generation Process

The dataset-generation process included the following stages:

1. Selection of first-aid-relevant source sections.
2. Text cleaning and contextual chunking.
3. Creation of structured, role-based generation prompts.
4. Inclusion of topic-specific source context within the prompts.
5. Generation of question-answer pairs using ChatGPT-4o-mini.
6. Iterative generation in batches to expand scenario coverage.
7. Review for duplication, relevance, contextual diversity, and formatting.
8. Filtering and refinement of generated records.
9. Human and medical-expert evaluation of a randomly selected sample.

The prompts instructed the generation model to:

* Base answers on the supplied context.
* Produce detailed and actionable responses.
* Cover different responder perspectives.
* Include diverse emergency settings.
* Avoid repeating previously generated questions.
* Return the results in structured JSON format.

Approximately 100 question-answer pairs were generated for each major topic through multiple batches.

## Quality Assurance

### Context Grounding

ChatGPT-4o-mini was instructed to generate answers using the supplied source chunks. This was intended to reduce unsupported generation and keep responses aligned with the selected first-aid material.

Context grounding does not guarantee factual correctness. Generated content may still contain hallucinations, omissions, misinterpretations, or unsafe recommendations.

### Filtering

Source chunks were selected based on their relevance to realistic first-aid scenarios. Generated records were subsequently reviewed and filtered for relevance, duplication, contextual diversity, formatting, and apparent safety issues.

### Medical-Expert Evaluation

A randomly selected sample of **200 question-answer pairs** was evaluated by **three medical professionals**.

Each pair was scored on a five-point scale using the following criteria:

| Criterion                    | Mean score |
| ---------------------------- | ---------: |
| Clarity                      |    4.2 / 5 |
| Relevance                    |    4.7 / 5 |
| Specificity and completeness |    4.0 / 5 |
| Safety and accuracy          |    3.7 / 5 |

These results apply only to the evaluated sample and should not be interpreted as complete medical validation of every record in the dataset.

The comparatively lower safety-and-accuracy score is particularly important. Models trained using this dataset require additional expert review, safety testing, guideline verification, refusal behavior, and human oversight before any real-world use.

## Intended Uses

Appropriate research uses include:

* Instruction-tuning or fine-tuning language models for first-aid QA
* Research on lightweight and offline-capable language models
* Small Language Model and edge-device experimentation
* Retrieval-augmented first-aid research
* Dataset-quality and synthetic-data research
* Safety evaluation of medical or emergency-response assistants
* Educational research involving first-aid question answering
* Development of prototypes with qualified human oversight
* Comparison of model performance in low-connectivity settings

## Out-of-Scope and Prohibited Uses

The dataset should not be used:

* As a substitute for emergency services or professional medical care
* As the sole source of medical or first-aid guidance
* For autonomous diagnosis, triage, or treatment
* In clinical decision-support systems without regulatory, clinical, and institutional review
* In high-stakes environments without qualified human supervision
* To delay contacting emergency services
* For general medical advice unrelated to first aid or emergencies
* To train systems that represent their answers as guaranteed to be medically correct
* To deploy an emergency-response chatbot without extensive safety evaluation
* To infer information about real patients or individuals
* To create deceptive medical content

## Limitations and Risks

### Partial Expert Validation

Only a sample of 200 records was evaluated by medical professionals. The complete dataset has not been independently clinically validated on a record-by-record basis.

### Source-Date Limitations

The grounding material was published in 2019. First-aid recommendations, resuscitation standards, public-health guidance, and local emergency procedures may change over time.

Users should compare the content against current recommendations from appropriate medical authorities before conducting safety-critical research or deployment.

### Language Limitations

The dataset is available only in English. It has not been validated for translated use, multilingual deployment, local dialects, or culturally specific emergency communication.

## Licensing Information

The released FirstAidQA dataset is licensed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

Users may share and adapt the dataset under the terms of that license, including the attribution requirement.

The *New Vital First Aid: First Aid Book* is third-party source material and remains subject to its own copyright and licensing terms. The CC BY 4.0 license attached to FirstAidQA does not grant rights to reproduce or redistribute the underlying source book.

Users are responsible for ensuring that their use of the dataset and related source materials complies with applicable copyright, medical, privacy, and regulatory requirements.

## Dataset Authors and Contributors

| Name                       | Affiliation                                  | Contribution                                               |
| -------------------------- | -------------------------------------------- | ---------------------------------------------------------- |
| Mushfiqur Rahman Mushfique | Islamic University of Technology, Bangladesh | Dataset author, repository maintainer, and paper co-author |
| Saiyma Sittul Muna         | Islamic University of Technology, Bangladesh | Dataset author and paper co-author                         |
| Rezwan Islam Salvi         | Islamic University of Technology, Bangladesh | Dataset author and paper co-author                         |
| Ajwad Abrar                | Islamic University of Technology, Bangladesh | Dataset author and paper co-author                         |

### Author Profiles

#### Mushfiqur Rahman Mushfique

* **Affiliation:** Islamic University of Technology, Bangladesh
* **Hugging Face:** [@i-am-mushfiq](https://huggingface.co/i-am-mushfiq)
* **ORCID:** [0009-0000-6633-1767](https://orcid.org/0009-0000-6633-1767)
* **Institutional email:** [mushfique2@iut-dhaka.edu](mailto:mushfique2@iut-dhaka.edu)
* **GitHub:** [i-am-mushfiq](https://github.com/i-am-mushfiq)
* **ResearchGate:** [Md Mushfique](https://www.researchgate.net/profile/Md-Mushfique)
* **LinkedIn:** [i-am-mushfiq](https://www.linkedin.com/in/i-am-mushfiq/)

#### Saiyma Sittul Muna

* **Affiliation:** Islamic University of Technology, Bangladesh
* **Hugging Face:** 
* **ORCID:** 
* **Institutional email:**`
* **GitHub:** 
* **ResearchGate:** 
* **LinkedIn:**

#### Rezwan Islam Salvi

* **Affiliation:** Islamic University of Technology, Bangladesh
* **Hugging Face:**
* **ORCID:** 
* **Institutional email:** 
* **GitHub:** 
* **ResearchGate:**
* **LinkedIn:**

#### Ajwad Abrar

* **Affiliation:** Islamic University of Technology, Bangladesh
* **Hugging Face:** 
* **ORCID:** 
* **Institutional email:**
* **GitHub:** 
* **ResearchGate:**
* **LinkedIn:** 

### Repository Maintenance

* **Repository maintainer:** Mushfiqur Rahman Mushfique
* **Dataset repository:** [i-am-mushfiq/FirstAidQA](https://huggingface.co/datasets/i-am-mushfiq/FirstAidQA)
* **Project contact:** [mushfique2@iut-dhaka.edu](mailto:mushfique2@iut-dhaka.edu)
* **Issue reporting:** Please use the Hugging Face repository’s Community section to report data-quality, safety, licensing, or documentation issues.

## Citation

### Paper Citation

```bibtex
@misc{muna2025firstaidqa,
  title         = {FirstAidQA: A Synthetic Dataset for First Aid and Emergency Response in Low-Connectivity Settings},
  author        = {Saiyma Sittul Muna and Rezwan Islam Salvi and Mushfiqur Rahman Mushfique and Ajwad Abrar},
  year          = {2025},
  eprint        = {2511.01289},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  doi           = {10.48550/arXiv.2511.01289}
}
```

### Dataset Citation

```bibtex
@dataset{muna2025firstaidqa_dataset,
  author    = {Saiyma Sittul Muna and Rezwan Islam Salvi and Mushfiqur Rahman Mushfique and Ajwad Abrar},
  title     = {FirstAidQA},
  year      = {2025},
  publisher = {Hugging Face},
  version   = {1.0},
  url       = {https://huggingface.co/datasets/i-am-mushfiq/FirstAidQA},
  note      = {Synthetic question-answering dataset for first aid and emergency response}
}
```

When using the dataset, please cite both the associated paper and the Hugging Face dataset release.

## Acknowledgments

The authors acknowledge the three medical professionals who participated in the expert evaluation of the sampled question-answer pairs.

The dataset was grounded in the *New Vital First Aid: First Aid Book*, published by Vital First Aid Training Services Pty Ltd.

## Reporting Issues

Users are encouraged to report:

* Potentially dangerous first-aid instructions
* Medical inaccuracies
* Hallucinated information
* Duplicate records
* Licensing concerns
* Formatting problems
* Missing safety context
* Bias or underrepresented scenarios

Reports may be submitted through the Hugging Face repository’s Community section.

## Disclaimer

FirstAidQA is provided for research and educational purposes without any guarantee of completeness, currentness, medical accuracy, or suitability for a particular use.

The dataset creators and contributors do not provide medical services through this dataset. In an actual emergency, users should contact the appropriate local emergency service and seek assistance from qualified medical professionals.
