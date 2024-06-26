# Implementation of a local LLM-based contextual chatbot with RAG using LangChain and Nvidia AI Endpoints for healthcare sector 
* This project aims to propose a chatbot designed according to privacy by design principles to avoid sending sensitive data outside the device in healthcare sector.
  
## Description

* The use case I have chosen involves providing a computer with this chatbot to patients in a hospital emergency room to give them a quick recommendation based on their symptoms and direct them to the appropriate service.
* This chatbot can be used for other purposes by modifying the data source used for retrieval to answer the questions. The LLM model can also be quantized and fine tuned according to the needs and computational capabilities of the machine.
* I tried this chatbot with quantized versions of Mistral-7b and Llama2-13b
* This chatbot uses LangChain framework (perform the vector store indexing through FAISS and run the model through LLAMA.cpp with RAG among other things), Nvidia technologies (i.e. Nvidia AI Endpoints for Embeddings and Nvidia CUDA for inference) and Gradio for the front-end.
* The chatbot provides the answers to hard coded questions by vector similarity search of the inputs (i.e. patient’s symptoms in our case) and are augmented with the model’s knowledge.
* This chatbot takes its context and responses by vector similarity search to give adequate responses (i.e.: quick recommendations and right medical department) which are augmented with the informations carried by the model.

## Video Presentation

[![Watch the video](https://img.youtube.com/vi/OjoA3c8PRKA/0.jpg)](https://www.youtube.com/watch?v=OjoA3c8PRKA)

## Getting Started

### Dependencies

You will have to install on your computer the following tools: 
* [Python](https://www.python.org/downloads/)
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
* [cuDNN](https://developer.nvidia.com/cudnn)
* Visual Studio - Development with C++ for desktop (make sure to install all the elements checked by default) 

You will need to pip install the following libraries:
* langchain langchain-community langchainhub
* beautifulsoup4
* pycuda
* llama-cpp-python (but I used another fork of this library giving information about CUDA usage through verbose = True, please find below in Installing section more details)
* gradio

You will need an Nvidia API key for the Nvidia Embeddings :
* Create an Nvidia NGC account (https://catalog.ngc.nvidia.com/)
* Go at the top right, click on your account
* Click on Setup
* Generate an API key

### Installing

I coded my project in a virtual environement in Visual Studio Code:
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
I chose this version because I had issues with Wheels installation and I wanted to be sure that I was using CUDA for inference, and I found exactly what I wanted thanks to this github page. As you will see, you have to know if your CPU uses AVX, and chose the right version according to your python and CUDA Toolkit version you have on your computer.

You can download the json files and use them locally with JSONLoader.

## Datasets and model choice
Concerning the datasets I used, to make the vector store, I chose these links to a github proposing medical Q&As in json format: 
* [eHealthforumsQAs](https://github.com/LasseRegin/medical-question-answer-data/blob/master/ehealthforumQAs.json)
* [icliniqQAs](https://github.com/LasseRegin/medical-question-answer-data/blob/master/icliniqQAs.json)
And I generated with ChatGPT 4o, a json with symptoms and the medical department where to go. You will find the file named hospital_departement.json in this github.

For the model, I tried quantized Mistral-7b and Llama-13b from Hugging Face :
* [Mistral-7b](bhttps://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF)
* [Llama-13b](https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF)

I recommend to fine tune this model on your dataset before using it for inference. I followed this tutorial to do so :
[Fine tuning](https://rentry.org/cpu-lora#appendix-a-hardware-requirements)

## GPU offloading
When you configure your model, you can choose through gpu_layers, how many layers of the model you can offload to the GPU. I offload half (22/44) because when I offload them all, the GPU was too overloaded probably because it's too short in RAM. I tested only 1 layer on the GPU and the rest on the CPU, and it was slower. The inference took 4 minutes in my case because I used a laptop configuration (RTX 2060M). For production, a desktop with an RTX 30 or 40 series with at least 12 Gb of VRAM or a professional GPU such as Quadro RTX would be more appropriate and way faster. 

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

My hardware specifications are :
CPU : Intel Core i7-1165G7 (Quad-Core 1.2 GHz - 2.8 GHz / 4.7 GHz Turbo - 8 Threads - Cache 12 Mo - TDP 28 W) 
RAM : 16 Go DDR4 3200 Mhz
GPU : Nvidia RTX 2060 M (6 Go V-RAM)
