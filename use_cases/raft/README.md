# SFTA: Accelerating Fine-Tuning of Small Language Models for Optimized Deployment

Welcome to SFTA, a comprehensive toolkit designed to accelerate the fine-tuning of Small Language Models (SLMs), enabling faster, more efficient deployment tailored to domain-specific needs.

## Why SFTA?

In the landscape of AI and machine learning, generic Large Language Models (LLMs) often struggle with domain-specific terminology and can be prone to generating hallucinated information. SFTA addresses these challenges by empowering presales and delivery teams to quickly prototype domain-specific SLMs for customized copilots and Retrieval-Augmented Generation (RAG) scenarios. By providing tools for synthetic data generation, fine-tuning, evaluation, and deployment, SFTA facilitates rapid time-to-market, reduces costs, and broadens the range of potential use cases.

## Why Choose Small Language Models (SLMs)?

SLMs are lightweight and resource-efficient, making them suitable for applications that require privacy-sensitive data handling on premises or need to be deployed at the edge, such as in remote locations or on devices with limited computational resources. In contrast, LLMs are larger, more resource-intensive, and reliant on cloud infrastructure, making them less suitable for these specialized environments.

## Key Components of SFTA

### 1. SLM RAFT
RAFT (Retrieval Aware Fine-Tuning) combines the approaches of closed-book and open-book for conversational AI. More details about RAFT can be found [here](https://techcommunity.microsoft.com/t5/ai-ai-platform-blog/raft-a-new-way-to-teach-llms-to-be-better-at-rag/ba-p/4084674).
In this repository, we propose to use SLMs in the context of RAFT. The process involves:

- **Synthetic Data Generation**: Creating domain-specific QA pairs to fine-tune models effectively (uses assets from https://github.com/ShishirPatil/gorilla/tree/main/raft).
- **Contextual Retrieval**: Using prompt engineering to fetch relevant chunks of information, ensuring the model has access to the necessary context.
- **Domain-Specific Training**: Fine-tuning SLMs to generate accurate, informed responses tailored to specialized fields.

### 2. LLM RAG
For scenarios where LLMs are still preferred, the RAG method enables:

- **Document Retrieval**: Extracting relevant chunks of information from a large set of documents.
- **Prompt Engineering**: Generating answers by dynamically integrating the retrieved information into the LLM's response generation process.

### 3. SFT: Supervised Fine-Tuning
Supervised Fine-Tuning (SFT) involves training models on QA datasets to improve their performance in answering questions. It can also incorporate additional documents to enable the model to generate well-informed responses, similar to having reference material available during a test.

### 4. Synthetic Data Pipeline
Our synthetic data pipeline simplifies the handling of various data formats, such as PDF documents. Users can upload their data to a blob or local storage, and the pipeline will process it, generating high-quality synthetic data that can be used to fine-tune SLMs.

## Conceptual Approach

SFTA provides a generalized pipeline for fine-tuning SLM models, which can also be adapted for LLM + RAG models. This flexible approach allows teams to:

- **Upload Data**: Simply upload your data, whether it's in a PDF format or another supported type, to blob or local storage.
- **Automated Data Handling**: Our pipeline automatically processes the data, creating synthetic QA pairs and preparing it for model training.
- **Model Training and Deployment**: Use our scripts to fine-tune your SLM or LLM models, and deploy them quickly with minimal configuration.
- **Evaluation and Benchmarking**: Evaluate the performance of both SLM and LLM models using our benchmarking tools. This allows you to choose the best approach for your specific requirements.

## How SLM RAFT is Different

SLM RAFT stands out by combining several techniques to fine-tune SLMs effectively:

- **Customized Training**: SLMs are fine-tuned with domain-specific QA pairs, ensuring that the model is well-adapted to the specialized language and requirements of the target domain.
- **Context-Aware Responses**: By retrieving relevant context through prompt engineering, SLM RAFT ensures that the model's responses are both accurate and informed.
- **Efficient Deployment**: SLMs, being smaller and more resource-efficient, are perfect for edge deployments, where larger models may not be feasible.

## Getting Started

1. **Clone the Repository**: Download the SFTA toolkit from GitHub.
2. **Prepare Your Data**: Upload your data to blob or local storage. The toolkit supports various formats, including PDF.
3. **Run the Synthetic Data Pipeline**: Use our scripts to generate synthetic QA pairs and prepare your data for fine-tuning.
4. **Fine-Tune Your Model**: Select your model, whether it's an SLM or LLM, and use our fine-tuning scripts to adapt it to your domain-specific needs.
5. **Deploy and Evaluate**: Deploy your model using our deployment scripts and evaluate its performance using the provided benchmarking tools.

## Conclusion

SFTA empowers teams to build, fine-tune, and deploy domain-specific SLMs quickly and efficiently. By offering a streamlined workflow for data preparation, model training, and evaluation, SFTA reduces the time and resources required to bring specialized AI solutions to market. Whether you're developing custom copilots or enhancing RAG capabilities, SFTA provides the tools you need to succeed