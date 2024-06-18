# Implementation of a local LLM-based contextual chatbot with LangChain and Nvidia AI Endpoints for healthcare sector 
* This project aims to propose a chatbot designed according to privacy by design principles to avoid sending sensitive data outside the device in healthcare sector.
  
## Description

* The use case I have chosen involves providing a computer with this chatbot to patients in a hospital emergency room to give them a quick recommendation based on their symptoms and direct them to the appropriate service.
* This chatbot can be used for other purposes by modifying the data source that serves as a reference for the questions. The LLM model can also be adjusted according to the needs and computational capabilities of the machine.
* This chatbot uses LangChain framework and Nvidia technologies (e.g. Nvidia AI Endpoints for Embeddings and Nvidia CUDA for inference)
* This chatbot takes its context and responses by vector similarity search to give adequate responses.
* The medical departement is chosen via mapping according to symptoms. 

## Getting Started

### Dependencies

You will have to install on your computer the following tools: 
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
* [cuDNN](https://developer.nvidia.com/cudnn)
* Visual Studio - Development with C++ for desktop (make sure to chose all the elements checked by default) 

You will need to pip install the following libraries:
* langchain langchain-community langchainhub
* beautifulsoup4
* pycuda
* llama-cpp-python (but I used another fork of this library giving information about CUDA usage through verbose = True, please find below in Installing section more details)

You will need an Nvidia API key for the Nvidia Embeddings :
* Create an Nvidia NGC account (https://catalog.ngc.nvidia.com/)
* Go at the top right, click on your account
* Click on Setup
* Generate an API key

### Installing

I coded my project in a virtual environement:
```
python3 -m venv venv
```
```
venv\Scripts\activate
```
I installed all the libraries above but for llama I used the following : 
```
python -m pip install llama-cpp-python==0.2.26 --prefer-binary --extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu122
```
It's a llama-cpp wheel that I found on this github : [llama-cpp-python cuBLAS wheels](https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels)
I chose this version because I had issues with Wheels installation and I wanted to be sure that I was using CUDA for inference, and I found exactly what I wanted thanks to this github page. As you will see, you have to know if your CPU uses AVX, and chose the right version according to your python and CUDA Toolkit versions you have on your computer.

## Datasets and model choice
For the dataset I used to make the vector store, I chose these links to a github proposing medical Q&As in json format: 
* [eHealthforumsQAs](https://github.com/LasseRegin/medical-question-answer-data/blob/master/ehealthforumQAs.json)
* [icliniqQAs](https://github.com/LasseRegin/medical-question-answer-data/blob/master/icliniqQAs.json)

For the model, I chose quantized Mistral-7b found on Hugging Face :

### Executing program

```
python3 doc_chatbot.py
```
You will find at the end of the output messages in the CLI, an URL that will display the chatbot in your main browser.

## Help

To verify your CUDA version, you can execute this prompt in CLI.
```
nvidia-smi
```

## Authors

Selim Salem  
(https://www.linkedin.com/in/selimsalem/)

## Version History

* 0.1
    * Initial Release

## License

This project is licensed under the MIT License - see the LICENSE.md file for details

## Acknowledgments

Please find below documentations I used to write my code : 
* [Nvidia Embeddings](https://nvidia.github.io/GenerativeAIExamples/latest/notebooks/10_RAG_for_HTML_docs_with_Langchain_NVIDIA_AI_Endpoints.html))
* [LangChain QA](https://python.langchain.com/v0.2/docs/tutorials/local_rag/)
* [Medical Q&A Dataset](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)
