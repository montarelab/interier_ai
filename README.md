The solution is tasked to generate interier design based on user's textual preferences and source image. As input the program takes text + image and outputs multiple images representing different parts of user's appartments + **JSON** report file with details. The details include token usages, costs, recorded automated metrics, and durations of each operation. 

This `README.md` has the following sections:
* **solution-outline** - the explaination what was built and how
* **how-to-run** - the commands to run the solution
* **implementation** - the details of development
* **future-work** - here I reflect on what can improved in the future

## Solution outline
<img src="images/diagram.png"/>

### Input data
The flow starts by gathering inputs. API takes an image from user (potentially multiple images in the future see **Future Work** section). Additionally, budget information, style preset, appartment information extracted from the textual prompt (see **Future Work** section for more details). 

### LLM Planner
All aggregated information is passed to the AI workflow implemented using **LangGraph**. The entrance node is **LLM Planner** that augments prompt making it more appropriate for the system and determines how many images it needs to generate. Current solution is able to generate multiple images of the appartments from different rooms, such that **LLM Planner** determines its number and writes the detailed explanation for each of them.

### Reference image generation
It's important ensure that the pictures follow the same style of appartments giving the consistent final result. However, it's worth to mention that the image, given by user, might not be that reference appartment picture, because it can be whatever the user wants to put in context: drawings, plans, screenshots, and ideas.

That's why generation of the first canonical reference appartment image is essential.

### Parallel image generation
After the reference is prepared, the workflow concurrently generates the rest of images for other parts of the plan.

### Evaluation
The system calculcates automated metrics **from every single generated image** using LLM-as-a-Judge.

Even though, exploring of eval metrics is an iterative process, I picked initial list based on my intuition:
* **Budget correspondece** - how generated interier is realistic for the budget given by a user;
* **General prompt correspondence** - how much the picture matches the prompt;
* **Realism** - how much it's real to do it;
* **Satisfaction** - how pleasent it is;
* **Usability** - how usable it is for living;

More details are included at **Future Work** section.

### Report 

I store all the generated images with semantically proper names for them as well as the request and the report. The report includes durations in seconds for every LLM generation including planning, image generations, and evaluations, also the plan, and all of the evaluation results. it also contains token usage information for each independent AI model with transformed from it costs using **LiteLLM**.

### Complexity

Because of the solution's workflow required complexity, the **Graph architecture** is the best option. I built it using **LangGraph**. However, I came to this decision iteratively after hours of planning and development. Modules such as LLM planner and LLM evaluator didn't come at the first day. Remembering the first version, the solution was built in a single service and was much more simple.

### RAG & Storage
I did not implement **RAG (Retrieval-augmented-generation)**, nor use **vector databases** because they're architecturally redundant points to provide high-quality interier design images. The images and results are stored locally in the file system as a good starting point.

### Caching

For the reasongs of time, the caching was not imlemented. Though, the concept is simple here: every request is hashing and stored to the cache storage (e.g. **Redis** or **PostgreSQL**) along with the API's response, where the hash is a DB index / key. Once the request with the similar request hash arrives, the system takes the prepared response and responds it saving costs on AI generation.

## How to run

1. Clone **Git** repo:
    ```
    git clone git@github.com:montarelab/interior_ai.git
    ```

2. Open the folder:
    ```
    cd interior_ai
    ```

3. Install packages:
    ```
    uv sync
    ```
4. Create `.env` file for environment variables:
    ```
    cp .env.example .env
    ```
5. Fill missing enviroment variables in `.env`.

6. Run **Docker daemon**.

7. Run app using **Docker compose**:
    ```
    docker compose up
    ```

- (Optionally) Run app locally:
    ```
    uv run python -m src.main
    ```

## Implementation

The solution is containerized using **Docker** to support running in different environments. It's imporatant to state API's host and port inside environment variables. Additionally, you can update models used for evluation, planning, image generation, and add LLM API keys. 


### Libraries used
* `fastapi` - API development;
* `langchain`, `langgraph`, and `jinja2` - workflow graph construction, template rendering;
* `litellm` - LLM automatic costs calculation from token usage metadata;
* `openai` and `google-genai` - AI API of **OpenAI** and **Google**, because **LangChain**/**LiteLLM**/**others** are either poor, or not-asynchronous, or not working for image generation as good as they are for trivial chat and text API;
* `aiofiles` and `pydantic` - productivity: async file operations and validated models development;


### API Design
* `/generate` - POST endpoint that generates interier images and report based on image and textual prompt as input, and responds with path for the results;
* `/evaluate` - POST endpoint that evaluates the system based on the given evaluation data: AI models to use, tasks, versions of tasks, and responds with the path for the results.

### Dataset and tasks
* Dataset is used for evaluation. There are 5 **tasks** inside the dataset and each one corresponds to the unique interier style. 
* There are 6 files inside each **task-folder**: source reference image and 5 versionized prompts. 
* Each task input simulates user's prompt. Even though the prompts are written with **Markdown**, they are structured and contain all the details exlicitly.
* The prompt version determines how specific the user's request is: 1 - specific, 5 - vague.

### Project structure
* `src` - all the **Python** modules necessary for API.
    * `main.py` - main executive point with **FastAPI** app;
    * `graph.py` - main app's AI workflow built with **LangGraph**;
    * `models.py` - **Pydantic** models used for development;
    * `settings.py` - structured, validatable environment variables accessor;
    * `utils.py` - extra functions including token usages compouning, data type converters, and prompt render functionality;
* `dataset` - includes 5 **tasks** for the system for each unique style. (see **Dataset and tasks**);
* `prompts` - **Jinja2** prompt templates used for AI. They include `plan`, `img_gen`, and `eval` for the corresponding stages of workflow. Additionally, there is `dataset_gen.jinja` that I used to generate the initial dataset of prompts for my own evaluation based on **Pinterest** interier pictures. No parts of the workflow use it, but I used it my own with `gpt-5.2` and added here to keep and version once it will be needed;
* `jobs` - system results. Each folder inside is an exact run of `/generate` or `/evaluate` determined as request's datatime + `UUID`. THere are both request and response stored inside;
* `images` - images used in **README.md**.

### LLM models support
Current solution supports only **Google** and **OpenAI** models because there are no unified API for any **any-to-image** AI model provider, while writing my own would be too expensive. I chose these providers as they have flagship models (see **Future work** section).


### Model preferences
* Image generation: `gemini-2.5 (Nano Banana)` - gave me the most pleasent results. It is not that fast, but the quality of details is high. Additionally it gives much more realistic outputs and less halucinations than `gpt-image-1-mini` and `gpt-image-1.5`. Moreover, it's not that long and expensive as `gemini-3-pro-image-preview`.
* Planner LLM and Eval LLM: `gpt-5-mini` + minimal reasoning - ensured the stages are processing fast. For planning and evaluation light and fast models can be used, because the tasks are notheavy on LLM's thinking capabilities.

Overall, the time for the prefered generation (`gemini-2.5` + `gpt-5-mini`) request can take around **2.5 minutes** or **175 seconds**:
* 35s for planning;
* 55 sec for 1st image;
* 55 sec for the rest of images;
* 30 sec for evaluation.

## Future work

* Application input may take only image or only text. Additionally, it can take multiple images. Such freedom would enhance app's flexibility.
* It's impossible to trust that a user would provide clear budget, style preset, appartment information. It would be beneficial to upgrade UX by detecting improper for our system text inputs (For example add this detection to the **LLM Planner** that can stop the interier design generation workflow and return to the user with the clarifying questions). As the result, once user provides the clear rewrited textual details, he will be much more satisfied with the results.
* As the given evaluation metrics were selected based on my intuition and in the scope of the first steps, more metrics will be inevitably discovered. Evaluation metrics exploration is a very iterative process. More tests need to be done and more discoveries.
* Support of multiple model providers including **Alibaba Qwen**. Because of the system's design, writing adapters to the current system is simple.