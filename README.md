[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)
[![GitHub issues](https://img.shields.io/github/issues/hasan-sayeed/BioAlloyRAG)](https://github.com/hasan-sayeed/BioAlloyRAG/issues)
[![GitHub Discussions](https://img.shields.io/github/discussions/hasan-sayeed/BioAlloyRAG)](https://github.com/hasan-sayeed/BioAlloyRAG/discussions)
[![Last Committed](https://img.shields.io/github/last-commit/hasan-sayeed/BioAlloyRAG)](https://github.com/hasan-sayeed/BioAlloyRAG/commits/main/)
<!-- These are examples of badges you might also want to add to your README. Update the URLs accordingly.
[![Built Status](https://api.cirrus-ci.com/github/<USER>/BioAlloyRAG.svg?branch=main)](https://cirrus-ci.com/github/<USER>/BioAlloyRAG)
[![ReadTheDocs](https://readthedocs.org/projects/BioAlloyRAG/badge/?version=latest)](https://BioAlloyRAG.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/BioAlloyRAG/main.svg)](https://coveralls.io/r/<USER>/BioAlloyRAG)
[![PyPI-Server](https://img.shields.io/pypi/v/BioAlloyRAG.svg)](https://pypi.org/project/BioAlloyRAG/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/BioAlloyRAG.svg)](https://anaconda.org/conda-forge/BioAlloyRAG)
[![Monthly Downloads](https://pepy.tech/badge/BioAlloyRAG/month)](https://pepy.tech/project/BioAlloyRAG)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/BioAlloyRAG)
-->

# BioAlloyRAG

**BioAlloyRAG:** A Retrieval-Augmented Generation System for Biomedical Alloy Literature

BioAlloyRAG is a specialized Retrieval-Augmented Generation (RAG) system designed to enable efficient retrieval and generation of insights from biomedical alloy literature. Tailored specifically for this domain, BioAlloyRAG facilitates the exploration of research papers, technical documents, and datasets relevant to the study and application of biomedical alloys.

Adapted from the [RAGSkeleton] framework, BioAlloyRAG inherits its modular and customizable architecture, allowing users to swap components like embedding models, vector databases, and LLMs to align with their specific data and research needs. However, unlike RAGSkeleton, BioAlloyRAG is purpose-built for working with biomedical alloy literature, making it a focused tool for researchers and engineers in this field.

By leveraging BioAlloyRAG, users can rapidly access domain-specific knowledge, streamline literature reviews, and generate concise, accurate responses from a wealth of specialized data. This system is an essential resource for advancing research and innovation in the biomedical alloy domain.

## Running the RAG System

For developers and contributors who want to work with the source code or customize the setup.

### 1. Clone the Repository

First, clone the repository and navigate to the project root directory:

```bash
git clone https://github.com/hasan-sayeed/BioAlloyRAG.git
cd BioAlloyRAG
```

### 2. Set Up the Environment

- Create a Conda Environment:

   ```bash
   conda env create -f environment.yml
   conda activate BioAlloyRAG
   ```

- BioAlloyRAG relies on Hugging Face for both the embedding model and the generative language model. Log in to Hugging Face from the terminal to access the necessary models:

   ```bash
   huggingface-cli login
   ```

   Enter your Hugging Face access token when prompted. You can obtain an API token by signing up at [Hugging Face] and navigating to your account settings.

   **Note:** Some models, like Meta LLaMA, may require additional permission from the owner on Hugging Face. To use these models, request access through the model's Hugging Face page, and you’ll be notified when access is granted.

### 3. Run the RAG System

To run the RAG system directly from the source, use the -m flag with Python to specify the module path. This will invoke the __main__.py entry point, which manages command-line arguments and initiates the chatbot.

```bash
python -m bioalloyrag --data_path /path/to/your/pdf/folder --load_mode api --model_name "meta-llama/Llama-3.2-3B-Instruct" --api_token <your_huggingface_api>
```

- `--data_path`: Path to a directory of PDF files for creating a vector database. If omitted, the system will use the existing knowledge base (if available) or prompt you to provide a path. If you want to ground your RAG on a different set of documents, simply provide the new directory path here, and the system will create a fresh knowledge base.

- `--load_mode`: Specify `local` to use a model hosted on your system, or `api` to use Hugging Face's API. `local` mode is suitable if you have the necessary computational resources, while `api` mode is useful if you prefer not to host the model locally or lack the computational resources.

- `--model_name`: Name of the language model to use. Default is "meta-llama/Llama-3.2-3B-Instruct". Any model available on Hugging Face can be specified here, allowing you to choose models best suited to your requirements.

- `--api_token`: Required if using the Hugging Face API (`--load_mode` api).

**Note:** With the API, you can opt for larger models that might otherwise be challenging to run locally. However, keep in mind that the Hugging Face Free API has a model size limit of 10GB. If you need to use larger models, consider a paid API plan or explore model optimization techniques.

### Usage

When you start BioAlloyRAG, you’ll be welcomed by a chatbot interface where you can ask questions. The system will retrieve relevant information from the knowledge base and generate responses grounded in the PDF documents you provided.

To exit, type `exit`.


## Project Organization

```
├── AUTHORS.md              <- List of developers and maintainers.
├── CHANGELOG.md            <- Changelog to keep track of new features and fixes.
├── CONTRIBUTING.md         <- Guidelines for contributing to this project.
├── Dockerfile              <- Build a docker container with `docker build .`.
├── LICENSE.txt             <- License as chosen on the command-line.
├── README.md               <- The top-level README for developers.
├── configs                 <- Directory for configurations of model & application.
├── data
│   ├── external            <- Data from third party sources.
│   ├── interim             <- Intermediate data that has been transformed.
│   ├── processed           <- The final, canonical data sets for modeling.
│   └── raw                 <- The original, immutable data dump.
├── docs                    <- Directory for Sphinx documentation in rst or md.
├── environment.yml         <- The conda environment file for reproducibility.
├── models                  <- Trained and serialized models, model predictions,
│                              or model summaries.
├── notebooks               <- Jupyter notebooks. Naming convention is a number (for
│                              ordering), the creator's initials and a description,
│                              e.g. `1.0-fw-initial-data-exploration`.
├── pyproject.toml          <- Build configuration. Don't change! Use `pip install -e .`
│                              to install for development or to build `tox -e build`.
├── references              <- Data dictionaries, manuals, and all other materials.
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures             <- Generated plots and figures for reports.
├── scripts                 <- Analysis and production scripts which import the
│                              actual PYTHON_PKG, e.g. train_model.
├── setup.cfg               <- Declarative configuration of your project.
├── setup.py                <- [DEPRECATED] Use `python setup.py develop` to install for
│                              development or `python setup.py bdist_wheel` to build.
├── src
│   └── bioalloyrag        <- Actual Python package where the main functionality goes.
├── tests                   <- Unit tests which can be run with `pytest`.
├── .coveragerc             <- Configuration for coverage reports of unit tests.
├── .isort.cfg              <- Configuration for git hook that sorts imports.
└── .pre-commit-config.yaml <- Configuration of pre-commit git hooks.
```

## Feedback

Any questions, comments, or suggestions are welcome! This project is a flexible foundation for RAG-based applications, and we’re open to improvements that can make it even more useful across various domains.

<!-- pyscaffold-notes -->

## Note

This project has been set up using [PyScaffold] 4.6 and the [dsproject extension] 0.7.2.

[MTEB leaderboard]: https://huggingface.co/spaces/mteb/leaderboard
[Hugging Face]: https://huggingface.co/
[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[Jupyter]: https://jupyter.org/
[nbstripout]: https://github.com/kynan/nbstripout
[Google style]: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[PyScaffold]: https://pyscaffold.org/
[dsproject extension]: https://github.com/pyscaffold/pyscaffoldext-dsproject
[RAGSkeleton]: https://github.com/hasan-sayeed/RAGSkeleton
