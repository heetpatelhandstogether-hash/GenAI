# Module 1: LLM Fundamentals — Detailed Learning Document

**Duration:** 6 days, ~2 hours per day (approximately 12 hours total)  
**Prerequisites:** Basic Python knowledge, familiarity with API calls, and an API key from at least one LLM provider (OpenAI, Anthropic, Google, etc.)

**By the end of this module, you will:**

- Explain how Large Language Models work at a conceptual level
- Understand tokens, context windows and latency
- Estimate the cost and speed of LLM requests
- Know when and why LLMs fail
- Choose a model based on your specific requirements using measurements, not marketing

---

## Table of Contents

1. [Overview: What You Need to Know](#overview)
2. [Unit 1.1: What an LLM Does](#unit-11-what-an-llm-does)
3. [Unit 1.2: Tokens, Context and Latency](#unit-12-tokens-context-and-latency)
4. [Unit 1.3: Beyond Text—Images, Audio and PDFs](#unit-13-beyond-text)
5. [Unit 1.4: Controlling Output](#unit-14-controlling-output)
6. [Unit 1.5: Where LLMs Fail](#unit-15-where-llms-fail)
7. [Unit 1.6: Choosing a Model](#unit-16-choosing-a-model)
8. [Hands-On Exercises](#hands-on-exercises)
9. [Assessment and Self-Check](#assessment)
10. [Common Mistakes and How to Avoid Them](#mistakes)
11. [Further Reading and Resources](#further-reading)

---

## Overview: What You Need to Know

Before diving into the details, here's the mental model you'll develop by the end of this module:

**Large Language Models are pattern-matching machines, not databases or search engines.**

They work by predicting the next token (a chunk of text) based on what came before. They don't "know" things the way humans do, and they don't retrieve facts from memory. Instead, they recognize patterns in the data they were trained on and generate fluent, confident-sounding text—even when that text is completely wrong.

Your job as an AI engineer is to:

1. Understand why and when they fail
2. Give them the right information at the right time (through context)
3. Measure their actual performance on your specific task
4. Choose the right model for your budget and constraints
5. Know what it will cost in tokens and money

This module teaches you the foundations. Modules 2-6 show you how to build, test and ship applications using these models.

---

## Unit 1.1: What an LLM Does

### Learning Objectives

- Explain the next-token prediction mechanism
- Distinguish between what an LLM is and isn't
- Understand why LLMs can sound confident while being wrong
- Describe how training and fine-tuning shape model behavior

### The Next-Token Prediction Machine

An LLM is a language prediction engine. Here's how it works:

**Step 1: You provide a prompt**

```
"The capital of France is"
```

**Step 2: The model predicts the next token**
A token is a small chunk of text—often a word, or part of a word. The model doesn't consciously "think." Instead, it calculates probabilities for every possible next token in its vocabulary (often 100,000+ tokens).

In this case:

- "Paris" might have 99.9% probability
- "London" might have 0.05% probability
- "cats" might have 0.001% probability

**Step 3: The model picks a token**
Usually, it picks the highest-probability one. Sometimes, for variety, it samples randomly from the distribution (you'll learn to control this with "temperature" in Unit 1.4).

**Step 4: Repeat**
The model appends the chosen token and repeats. It's generating "Paris" and now predicts the next token after "The capital of France is Paris". This continues until it decides to stop or hits a token limit.

---

### What an LLM Is Not

**Not a database.** It doesn't store information and retrieve it. If you ask an LLM trained in 2023 about events in 2024, it won't fetch the information—it will guess based on patterns.

**Not a search engine.** It doesn't look things up. It generates.

**Not a fixed program.** You can't trace a bug in code and fix it. The model's behavior emerges from billions of learned patterns.

**Not retrieving stored answers.** Every time you ask the same question, it follows the same probability distribution—but slight randomness means slightly different answers each time (unless you set temperature to 0).

### Why This Matters

This is why LLMs hallucinate (make up false information with complete confidence), why they're bad at counting or exact arithmetic, and why they need you to supply the right information through context rather than expecting them to "know" it.

---

### The Path Through the Model: Text → Tokens → Numbers → Layers → Output

**Text input:** "What is 2 + 2?"

**Tokenization:** The text is split into tokens. For the above, it might be:

```
["What", " is", " 2", " +", " 2", "?"]
```

(Different models tokenize differently—sometimes more, sometimes fewer tokens.)

**Embedding:** Each token becomes a vector of numbers—a list of 768, 1024, 2048 or more numbers that represent its meaning mathematically.

**Layers:** The model has many layers (GPT-4o has 120 layers, for example). Each layer refines the representations by looking at context and relationships.

**Output logits:** After all layers, the model outputs a probability distribution over all possible next tokens. The number associated with each token is called a "logit."

**Sampling:** The model picks the token with the highest probability (or samples from the distribution based on temperature, see Unit 1.4).

**Repeat:** The new token is appended to the sequence and fed back to the model.

---

### Parameter Counts: What 8B, 70B and 405B Mean

You'll see models described as 7B, 13B, 70B, 405B, etc. The number is the count of "parameters"—learned values the model adjusts during training.

- **Bigger isn't always better.** A 70B model isn't necessarily 10× better than a 7B model. It depends on your task, the training data, and how the model was fine-tuned.
- **More parameters = more capability (usually).** Larger models can capture more nuanced patterns and are often better at reasoning and rare tasks.
- **More parameters = slower and more expensive.** A larger model takes longer to run and costs more per token.
- **Domain matters.** A 7B model trained on code might beat a 70B model trained on general text for coding tasks.

**You'll measure on your own task; don't assume a leaderboard rank tells you what's best for you.**

---

### Why Chat Models Behave the Way They Do

Modern LLMs like GPT-4o, Claude and Gemini go through two main training phases:

1. **Pretraining:** The model learns to predict text from a huge corpus of internet text, books and code. This is where it learns language, facts (up to a knowledge cutoff) and basic reasoning.

2. **Instruction fine-tuning and human feedback (RLHF):** After pretraining, the model is fine-tuned on examples of helpful, harmless and honest responses. Human raters score model outputs, and the model learns to prefer high-scoring outputs.

This is why:

- Chat models follow instructions (they were trained to)
- They refuse certain requests (they were trained to decline harmful ones)
- They hedge and say "I don't know" sometimes (they learned that's better than confidently guessing)
- They answer conversationally (they were trained on instruction-response pairs)

**A base model (pretraining only) behaves very differently—it completes text, not answers questions.**

---

### Hands-On: Try It (Day 1)

**Objective:** See firsthand that the same prompt gives different answers due to the probabilistic nature of LLMs.

**What to do:**

1. Choose a prompt: `"The future of AI in the next 5 years will be marked by"` (something creative and open-ended)
2. Call your model provider's API five times with this prompt, **without changing any settings** (or with temperature set to its default, usually 0.7)
3. Record all five responses

**Example (using OpenAI's API in Python):**

```python
import openai

openai.api_key = "your-api-key"

prompt = "The future of AI in the next 5 years will be marked by"

for i in range(5):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7  # or leave it out for default
    )
    print(f"Run {i+1}:")
    print(response.choices[0].message.content)
    print("\n---\n")
```

**What to observe:**

- The responses should differ slightly (or significantly).
- This is normal and expected—the model is sampling from a probability distribution.
- If you set `temperature=0`, the responses should be nearly identical (it picks the highest-probability token every time).

**Outcome:** Understand that LLM outputs are probabilistic, not deterministic.

---

### Summary: Unit 1.1

An LLM is a next-token predictor trained on massive text and fine-tuned with human feedback. It's not a database, search engine or fixed program. Bigger isn't always better; it depends on your task. Fine-tuning shapes behavior; pretraining gives foundation knowledge (up to a cutoff date).

---

## Unit 1.2: Tokens, Context and Latency

### Learning Objectives

- Understand what tokens are and why they matter
- Estimate the cost of a request
- Predict latency
- Know the limits and properties of context windows
- Understand the difference between context and memory

### Tokens, Not Words

When you make an API call to an LLM, you're not charged per word or per sentence—you're charged per **token**.

A token is roughly a word, but not exactly.

**Examples:**

```
"Hello world"           → 2 tokens
"It's"                  → 2 tokens ("It" + "'s")
"ChatGPT"               → 1 token
"The quick brown fox"   → 4 tokens
"123456"                → 1 token
"def function():"       → 5–6 tokens (depending on the model)
```

**Why this matters:**

- Some text uses more tokens than others
- You're charged per token, so it's important to estimate
- Different models tokenize differently
- Your token budget (context window) is a hard limit

---

### Why Some Text Costs More

Not all languages and content are created equal:

**English:** highly efficient. ~1.3 tokens per word.

**Hindi or mixed Hindi-English:** very inefficient. A Hindi word might be 3–5 tokens. Switching between Hindi and English within a sentence makes it worse.

**Code:** varies. Some languages (Python) are reasonably efficient; others (Rust, C++) with special characters less so.

**Numbers:** Each digit is often a token. A long number or a phone number costs more than you'd expect.

**Whitespace and punctuation:** Usually don't cost much, but can add up.

**Example:**

```
English: "The weather is good"
         → 4 tokens

Hindi: "मौसम अच्छा है"
       → ~9–12 tokens (same meaning, ~3× the cost)

Mixed: "The मौसम is अच्छा"
       → ~8–10 tokens (worse than either alone)
```

**Implication:** If your user base is Hindi-speaking and you're using English prompts, your token costs will be 2–3× higher than you'd expect from English-only apps.

---

### Input vs Output Tokens

Both are billed, but differently:

**Input tokens:** the prompt you send to the model.

**Output tokens:** the response the model generates.

In most pricing models, **output tokens cost more than input tokens**. For example, on OpenAI's GPT-4o:

- Input: $5 per million tokens
- Output: $15 per million tokens (3× more)

This changes the economics of your app:

- A request with a 5,000-token input and 100-token output costs less than one with a 1,000-token input and 500-token output
- Streaming long responses is cheaper than prompting for short answers
- Context caching (keeping large context in "cache") saves input token cost

---

### Latency: Time to First Token and Generation Speed

Latency has two components:

**Time to First Token (TTFT):** How long before the model returns the first token. This is usually 200–500 ms for cloud APIs, sometimes more if the provider is busy.

**Generation speed:** How fast the model outputs tokens after the first one. Usually 50–150 ms per token (or 6–20 tokens/second).

**What this means for user experience:**

A user waiting for a 1,000-token response:

- TTFT: 500 ms
- Generation: 1,000 tokens × 100 ms/token = 100,000 ms (100 seconds)
- **Total: ~100.5 seconds**

That's a terrible user experience. But if you stream, the user sees the first token after 500 ms and text appearing continuously. Even though the total time is the same, it feels much faster.

**Implication:** Streaming is not just a nice-to-have; it's essential for interactive applications.

---

### What Fills the Context Window

A context window is the total amount of text (in tokens) that the model can ingest in one call. Modern models have large windows:

- GPT-4o: 128,000 tokens
- Claude 3.5 Sonnet: 200,000 tokens
- Llama 3.1: 128,000 tokens

Everything shares this space:

```
System instructions:        500 tokens
Conversation history:     2,000 tokens
Retrieved documents:     4,000 tokens
User's current question:    200 tokens
Memories (about the user): 300 tokens
Total used:              7,000 tokens

Available (out of 128,000): 121,000 tokens
```

If you exceed the limit, the request fails.

---

### Context ≠ Knowledge ≠ Memory

**Context (the window):** what you send the model in this specific API call. It's temporary. The next call starts with a fresh context unless you explicitly resend history.

```python
# Call 1
response1 = model.ask("What's my favorite color?")
# Model responds based only on the prompt; it has no memory.

# Call 2
response2 = model.ask("What's my favorite color?")
# Model has no memory of Call 1. Same question might get a different answer.
```

**Knowledge (trained into the model):** patterns and facts learned during pretraining. Fixed up to a knowledge cutoff. Can't be updated without retraining.

**Memory (built by you):** you have to store user facts in a database and inject them into the context. There's no built-in long-term memory.

```python
# Better approach
user_fact = retrieve_from_db("user_123", "favorite_color")
# Output: "blue"
context = f"The user's favorite color is {user_fact}. "
response = model.ask(context + "What's my favorite color?")
```

---

### Bigger Windows Aren't Better Windows

A common misconception: "I have a 200k token window, so I can send all my documents."

**Problem 1: The lost-in-the-middle effect.** When you cram too much into context, important facts buried in the middle get overlooked.

```
Imagine these were in your prompt:
[5,000 tokens of system instructions]
[50,000 tokens of documents]  ← Important fact: "Return policy is 30 days"
[30,000 tokens of more documents]
[User question: "What's the return policy?"]

The important fact might get lost because it's surrounded by noise.
```

**Problem 2: Diluted focus.** Extra text dilutes the model's attention, even if it's not directly used. It has to process and weight more information.

**Solution:** Be selective. Send only the relevant documents, trim irrelevant context, use retrieval (Module 4) to fetch only what's needed.

---

### Hands-On: Try It (Day 2, Part 1)

**Objective:** Count tokens and estimate costs.

**What to do:**

1. **Set up a tokenizer.** Use your model provider's tokenizer library:

   For OpenAI:

   ```python
   import tiktoken
   encoding = tiktoken.encoding_for_model("gpt-4o")

   text = "Hello, how are you today?"
   tokens = encoding.encode(text)
   print(f"Text: {text}")
   print(f"Tokens: {tokens}")
   print(f"Token count: {len(tokens)}")
   ```

   For Anthropic (Claude):

   ```python
   import anthropic
   client = anthropic.Anthropic(api_key="your-key")

   # Anthropic's tokenizer is built-in via the API
   response = client.messages.count_tokens(
       model="claude-3-5-sonnet-20241022",
       messages=[{"role": "user", "content": "Hello, how are you?"}]
   )
   print(f"Token count: {response.input_tokens}")
   ```

2. **Count tokens for several pieces of text:**
   - An English paragraph (100 words)
   - The same paragraph translated to Hindi
   - A code snippet (10 lines of Python)
   - A list of numbers (10 phone numbers)
   - A JSON object (a realistic user record)

3. **Calculate costs:**
   - Assume GPT-4o pricing: $5 per million input tokens, $15 per million output tokens
   - Estimate: if your app sends 100 users 1,000-token inputs and receives 200-token outputs per hour, what's your hourly cost?

   ```
   100 users × (1,000 input + 200 output) = 120,000 tokens per hour
   Cost = (100,000 × $5 + 20,000 × $15) / 1,000,000
        = (500 + 300) / 1,000,000
        = $0.0008 per hour
        = ~$0.58 per day
        = ~$18/month
   ```

**What to observe:**

- Hindi and code have much higher token counts than plain English
- Numbers are surprisingly expensive
- Your token cost depends heavily on the language and content type

**Outcome:** Develop intuition for token costs and ability to estimate them before building.

---

### Hands-On: Try It (Day 2, Part 2)

**Objective:** Measure latency.

**What to do:**

1. **Set up timing:**

   ```python
   import time
   import openai

   openai.api_key = "your-key"

   prompt = "Write a detailed 500-word essay on the history of artificial intelligence."

   start = time.time()
   response = openai.ChatCompletion.create(
       model="gpt-4o-mini",
       messages=[{"role": "user", "content": prompt}],
       stream=False  # First, non-streaming
   )
   end = time.time()

   output_tokens = response.usage.completion_tokens

   print(f"Total time: {end - start:.2f} seconds")
   print(f"Output tokens: {output_tokens}")
   print(f"Tokens per second: {output_tokens / (end - start):.1f}")
   ```

2. **Compare streaming vs non-streaming:**

   ```python
   # Streaming version
   start = time.time()
   stream = openai.ChatCompletion.create(
       model="gpt-4o-mini",
       messages=[{"role": "user", "content": prompt}],
       stream=True
   )
   first_token_time = None
   token_count = 0
   for chunk in stream:
       if first_token_time is None:
           first_token_time = time.time() - start
       token_count += 1
   end = time.time()

   print(f"Time to first token: {first_token_time:.3f} seconds")
   print(f"Total time: {end - start:.2f} seconds")
   print(f"Tokens per second: {token_count / (end - start):.1f}")
   ```

3. **Observe the difference:** TTFT should be similar, but the user sees text immediately with streaming.

**What to observe:**

- TTFT is usually 200–1,000 ms (depending on provider load)
- Generation speed is typically 10–30 tokens per second
- Streaming makes the app feel faster even though total time is the same

**Outcome:** Understand latency composition and why streaming matters for UX.

---

### Summary: Unit 1.2

Tokens matter for cost and context. Different languages and content have vastly different token costs. Input and output tokens are priced separately, with output often 3× more expensive. Context is temporary; knowledge is fixed; memory is built by you. Bigger windows don't guarantee better results; selective context does. TTFT and generation speed drive user experience; streaming is essential.

---

## Unit 1.3: Beyond Text—Images, Audio and PDFs

### Learning Objectives

- Understand how images and audio are handled by modern LLMs
- Estimate costs for multimodal inputs
- Know what models read well and where they slip
- Decide between sending images vs extracted text

### Models That See and Hear

Modern LLMs like Claude, GPT-4o and Gemini accept images, audio and documents directly, not just text. This opens new possibilities but introduces new costs and failure modes.

**Image input:**

```python
import anthropic
client = anthropic.Anthropic()

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_encoded_image,
                    },
                },
                {
                    "type": "text",
                    "text": "What is in this image?"
                }
            ],
        }
    ],
)
```

**Audio input:**
Some models (e.g., OpenAI's Whisper, Anthropic's plans for audio) accept audio directly. Otherwise, transcribe first.

**PDF input:**
Digital PDFs (with extractable text) are usually handled as documents. Scanned PDFs (images of pages) are treated as images.

---

### Images Cost Tokens Too

An image isn't free. It's converted to tokens, and the cost depends on resolution.

**Example (Claude):**

- A 1024×768 image ≈ 1,369 tokens
- A 4096×2160 image ≈ 5,376 tokens

Higher resolution = more tokens = higher cost.

**Best practice:** Crop or downsize images to what's needed.

```python
from PIL import Image

# Instead of sending a 4K screenshot, crop to the relevant area
img = Image.open("screenshot.png")
cropped = img.crop((100, 100, 800, 600))  # crop to relevant area
cropped.save("cropped.png")
```

---

### PDFs Come in Two Kinds

**Digital PDFs:** Have extractable text underneath. The model can read the text efficiently.

```
PDF content: {"text": "Lorem ipsum dolor..."}
```

**Scanned PDFs:** Just images of pages. No text layer. The model has to read them like an image.

```
PDF content: [image of page 1] [image of page 2] ...
```

For scanned PDFs, you have two options:

1. **OCR (optical character recognition):** Extract text from the image programmatically
2. **Send the image to the model:** Let the model read it as a vision task

OCR is cheaper and usually faster; model-based reading is more accurate on complex layouts.

---

### Audio

Raw audio can be sent to models that support it, or you can transcribe first.

**Transcription first (cheaper):**

```python
import openai
openai.api_key = "your-key"

with open("audio.mp3", "rb") as f:
    transcript = openai.Audio.transcribe("whisper-1", f)

print(transcript["text"])
```

Then send the text to your LLM.

**Direct audio (where supported):**
Some models accept audio directly and preserve tone/emphasis.

**Cost:** A 1-minute audio file ≈ 150–300 tokens after transcription, depending on speech speed and clarity.

---

### What Models Read Well

**Text in images:** If a screenshot or photo contains readable text, models usually read it accurately (especially for clear fonts).

**Charts and graphs:** The main point (trend, distribution) is usually captured correctly.

**Form fields:** Named fields in a form (e.g., "Name: John") are usually read correctly.

**Tables:** Sometimes, but see below.

---

### Where They Slip

**Counting objects:** "How many people are in this image?" — models often get this wrong, even for small counts.

**Small or dense text:** Anything smaller than a typical font on a phone screen gets missed or misread.

**Exact table values:** A model might read "the table shows sales by region" but might misread the specific numbers.

**Confidently describing details that aren't there:** A model might invent details to fill gaps in understanding.

**Complex layouts:** Multi-column text or documents with overlapping elements confuse models.

---

### Image or Extracted Text?

**Send extracted text when:**

- Clean, high-quality text extraction is available (digital PDF, OCR)
- Cost is a priority
- Exact values (numbers, specific text) matter
- Speed is critical

**Send the image when:**

- Layout, visual structure, or charts matter
- Handwriting is involved
- Visual details (colors, positions, design) are important
- Text is embedded in a complex layout

**Example:**

```python
# Task: Extract all customer names from a table in a PDF

# Option 1: Extract text
text = extract_text_from_pdf("document.pdf")
# "Customer Name | Amount
#  John Smith   | $500
#  Jane Doe     | $750"
response = model.ask(f"Extract all customer names from this table: {text}")

# Option 2: Send the image
# (Send the table as an image)
response = model.ask({"image": table_image, "text": "Extract all customer names"})

# Option 1 is usually cheaper, faster, and more accurate for this task.
```

---

### Hands-On: Try It (Day 3)

**Objective:** Experience when images help and when they don't.

**What to do:**

1. **Find a chart or infographic** (any PNG or JPEG). Good candidates:
   - A bar chart showing sales data
   - A line graph showing trends
   - A table with numbers
   - A screenshot of a form

2. **Test 1: Send the image**

   ```python
   import anthropic
   import base64

   client = anthropic.Anthropic()

   with open("chart.png", "rb") as img_file:
       image_data = base64.standard_b64encode(img_file.read()).decode("utf-8")

   response = client.messages.create(
       model="claude-3-5-sonnet-20241022",
       max_tokens=1024,
       messages=[{
           "role": "user",
           "content": [
               {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_data}},
               {"type": "text", "text": "What are the key insights from this chart? List all specific numbers you see."}
           ]
       }]
   )
   print("Image approach:")
   print(response.content[0].text)
   ```

3. **Test 2: Extract text and send it**

   For a chart, you could manually transcribe it or use OCR:

   ```python
   # Pretend this is extracted/transcribed data
   extracted_text = """
   Sales by Region (Q4 2024):
   - North: $125,000
   - South: $98,000
   - East: $156,000
   - West: $142,000
   """

   response = client.messages.create(
       model="claude-3-5-sonnet-20241022",
       max_tokens=1024,
       messages=[{
           "role": "user",
           "content": f"What are the key insights from this data? {extracted_text}"
       }]
   )
   print("Text approach:")
   print(response.content[0].text)
   ```

4. **Compare:**
   - Which approach was faster?
   - Which approach was cheaper (in tokens)?
   - Which approach was more accurate?
   - Which approach got the specific numbers right?

**What to observe:**

- Text extraction is cheaper (fewer tokens)
- For numerical data, text is often more accurate
- For visual insights (trends, patterns), images might be clearer
- Mix and match based on your needs

**Outcome:** Develop judgment about when to use each approach.

---

### Summary: Unit 1.3

Images and audio add new input modalities but come with costs (tokens) and failure modes (counting, small text, exact numbers). Digital PDFs and extracted text are cheaper and often more accurate than images. Use images for layout and visual insights; use text for facts and numbers.

---

## Unit 1.4: Controlling Output

### Learning Objectives

- Understand temperature and how it affects output
- Know when to use each model type (reasoning, fast)
- Control output length and stopping
- Match model choice to your task

### Temperature: The Randomness Dial

Temperature is a parameter that controls randomness in output.

**Temperature = 0 (no randomness):**
The model always picks the highest-probability next token. Same input → identical output every time.

```python
response1 = model.ask("Write a haiku about spring", temperature=0)
# Output: [deterministic, same every time]

response2 = model.ask("Write a haiku about spring", temperature=0)
# Output: [identical to response1]
```

Use for: **fact extraction, classification, answers where consistency matters.**

**Temperature = 0.5 (some randomness):**
The model samples from the probability distribution but biases toward high-probability tokens. Varied but coherent.

Use for: **most tasks; a good default.**

**Temperature = 1.0 or higher (high randomness):**
The model samples more freely from the distribution, including lower-probability tokens. More creative and varied.

```python
response1 = model.ask("Write a creative story beginning", temperature=1.5)
response2 = model.ask("Write a creative story beginning", temperature=1.5)
# Output: [completely different; may be incoherent]
```

Use for: **brainstorming, creative writing, where variety is desired.**

---

### Matching Randomness to Task

**Low temperature (0–0.3):**

- Extracting facts ("What are the three main points?")
- Classification ("Is this email spam?")
- Calculation ("What is the total?")
- Any task where consistency is critical

**Medium temperature (0.5–0.7, usually default):**

- Most conversational use cases
- Writing an email draft
- Explaining a concept
- Summarizing

**High temperature (1.0+):**

- Brainstorming ideas
- Creative writing
- Generating multiple alternatives
- Tasks where exploration is the goal

---

### Reasoning Models vs Fast Models

Modern LLM providers offer two types of models:

**Fast models (e.g., GPT-4o-mini, Claude 3.5 Haiku):**

- Low latency (TTFT < 200ms)
- Cheap
- Good at language, extracting, rewriting
- Weak at: hard math, complex logic, multi-step planning

**Reasoning models (e.g., OpenAI o1, Claude Opus):**

- Longer latency (think for several seconds)
- More expensive
- Excellent at: hard math, logic puzzles, code generation, planning
- May ignore some settings (temperature, top_p) — the thinking is not randomizable

---

### When Reasoning Is Wasted

**Extraction:** "Extract the customer name and email from this text."
→ Use a fast model. The model reads and pulls out text; no reasoning needed.

**Rewriting:** "Make this email more professional."
→ Use a fast model. Pattern matching is enough.

**Classification:** "Is this review positive or negative?"
→ Use a fast model. Classification is pattern matching.

**Summarization:** "Summarize this article in 3 sentences."
→ Use a fast model.

**What reasoning helps with:**

- Multi-step math problems
- Logic puzzles
- Writing complex algorithms
- Planning a system architecture
- Explaining why something is the case (not just what)

---

### Length Limits and Stop Points

Control output length to manage cost and format:

```python
response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a long essay"}],
    max_tokens=500  # Cap at 500 output tokens
)
```

You can also specify stopping points:

```python
response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "List three things: A) B) C)"}],
    stop=["\nD)"]  # Stop if the model tries to add a D)
)
```

Or:

```python
response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a poem"}],
    stop=["\n\n"]  # Stop at double newline (end of poem)
)
```

---

### Hands-On: Try It (Day 4)

**Objective:** See how temperature and model choice affect output.

**What to do:**

1. **Set up a consistent prompt:**

   ```python
   import openai
   openai.api_key = "your-key"

   prompts = {
       "extraction": "Extract all numbers from this text: 'The annual revenue is $2.5M, up 15% from last year.'",
       "creative": "Write a short, creative description of what it feels like to be a library book.",
       "reasoning": "If all roses are flowers, and all flowers bloom, are all roses blooming? Explain."
   }
   ```

2. **Run each prompt at multiple temperatures:**

   ```python
   for task, prompt in prompts.items():
       print(f"\n=== {task.upper()} ===\n")
       for temp in [0, 0.5, 1.0, 1.5]:
           response = openai.ChatCompletion.create(
               model="gpt-4o-mini",  # fast model
               messages=[{"role": "user", "content": prompt}],
               temperature=temp,
               max_tokens=150
           )
           print(f"Temperature {temp}:")
           print(response.choices[0].message.content)
           print()
   ```

3. **Compare temperatures:**
   - For extraction: which temperature gives the most consistent, accurate answer?
   - For creative: how does the variety change?
   - For reasoning: does higher temperature help or hurt?

4. **Compare models (if you have access):**

   ```python
   reasoning_response = openai.ChatCompletion.create(
       model="o1",
       messages=[{"role": "user", "content": prompts["reasoning"]}]
       # Note: o1 doesn't accept temperature; it reasons internally
   )
   ```

**What to observe:**

- Extraction tasks: low temperature is best
- Creative tasks: high temperature adds variety but might be incoherent
- Reasoning: o1 (or Claude Opus) may solve complex problems that fast models struggle with
- Consistency: T=0 gives identical outputs; T>0 varies

**Outcome:** Develop intuition for temperature and model selection.

---

### Summary: Unit 1.4

Temperature controls randomness (0 = deterministic, high = varied). Match temperature to task: low for facts, medium for general use, high for creativity. Fast models are good for most tasks; reasoning models excel at hard problems but cost more. Control output length to manage cost.

---

## Unit 1.5: Where LLMs Fail

### Learning Objectives

- Recognize LLM failure modes
- Understand why they happen
- Know which tasks need safeguards
- Learn to measure success rate, not just capability

### Common Strengths

LLMs are excellent at:

- **Summarizing:** condensing text while keeping the essence
- **Rewriting:** changing tone, style, or format
- **Extracting information:** pulling specific data from text
- **Classifying:** putting text into categories
- **Coding:** writing, explaining and debugging code (though not perfectly)
- **Translating:** converting between languages
- **Handling messy text:** parsing typos, informal language, mixed formats

---

### Common Failure Modes

**Making things up (hallucination):**

```
User: "Who founded Tesla Energy?"
Model: "Tesla Energy was founded by Sarah Chen in 2015."
Truth: Tesla Energy doesn't exist. The model made up a name and date.
```

**Arithmetic slips:**

```
User: "What is 523 + 728?"
Model: "523 + 728 = 1,251"
Truth: 523 + 728 = 1,251. (This one is right, but models often get arithmetic wrong.)
```

**Shaky reasoning:**

```
User: "If 10% of people like bananas and there are 1,000 people, how many like bananas?"
Model: "If 10% of people... that's about 50 people."
Truth: 10% of 1,000 = 100.
```

**Ignoring instructions:**

```
System: "Always respond in French."
User: "What is 2+2?"
Model: "2+2 = 4."
Truth: Should have responded in French.
```

**Sensitivity to wording:**

```
Q1: "Is climate change a problem?"
Model: (thoughtful answer acknowledging nuance)

Q2: "Climate change is a problem, right?"
Model: (different answer, biased by the "right?")
```

**Outdated knowledge:**

```
User: "Who is the president of France?"
Model: (responds with information from 2023, now outdated)
```

**Inconsistency:**

```
Call 1: Ask the same question → Answer A
Call 2: Ask the same question → Answer B
(Both are plausible; the model picked differently due to randomness.)
```

---

### Why Models Make Things Up

LLMs are trained to predict the next likely token, not to verify that output is true.

If you ask it something outside its training data, the model doesn't say "I don't know." Instead, it generates something that sounds plausible. This is called **hallucination** or **confabulation**.

Example:

```
Prompt: "What did Alice tell Bob about the meeting?"
Context: [No mention of Alice, Bob, or any meeting]

A truthful model would say: "I don't have information about that."

Instead, LLM: "Alice told Bob that the meeting was rescheduled to 3 PM and that he should bring the quarterly reports."

(Completely made up, but it sounds reasonable.)
```

---

### Ask "How Reliably?" Not "Can It?"

The real question isn't "Can an LLM do this task?" but "**How often does it succeed?**"

A prompt that works 60% of the time is unreliable. You need 95%+ success for production use (or a fallback).

**Example:**

```
Task: Extract customer ID from support emails.

Model A: Succeeds 92% of the time.
Model B: Succeeds 98% of the time.
Model C: Succeeds 74% of the time.

For production, use Model B. Model C is unreliable.
```

This is why Module 7 (after you build an app) covers testing and evaluation extensively. You measure success rate on **your** task, not on benchmarks.

---

### Hands-On: Try It (Day 5, Part 1)

**Objective:** See failure modes firsthand.

**What to do:**

1. **Write 10 diverse questions:**

   ```
   Questions:
   1. "What is the capital of Portugal?" (Known, straightforward)
   2. "What was the population of London in 1850?" (Known, specific)
   3. "What did I have for breakfast?" (Unknowable without context)
   4. "How many people have visited Mars?" (Unknowable; Mars hasn't been visited)
   5. "What is 47 × 89?" (Calculation; should be testable)
   6. "Translate 'hello' into a language you invented." (Impossible)
   7. "Who is the CEO of a company called Acme that I made up?" (Unknowable)
   8. "If all cats are animals and Fluffy is a cat, is Fluffy an animal?" (Logic)
   9. "Write a 3-word sentence." (Constrained)
   10. "What color is the sky on Tuesday?" (Nonsensical)
   ```

2. **Ask your model each question:**

   ```python
   import openai
   openai.api_key = "your-key"

   questions = [
       "What is the capital of Portugal?",
       "What was the population of London in 1850?",
       # ... (all 10)
   ]

   for q in questions:
       response = openai.ChatCompletion.create(
           model="gpt-4o-mini",
           messages=[{"role": "user", "content": q}],
           temperature=0  # Deterministic
       )
       print(f"Q: {q}")
       print(f"A: {response.choices[0].message.content}\n")
   ```

3. **Evaluate the answers:**
   - Is it correct?
   - Is it confident but wrong?
   - Does it say "I don't know"?
   - Did it make something up?

4. **Document:**
   - Which question types did the model handle best?
   - Which ones did it fail on or hallucinate?

**What to observe:**

- LLMs are often confident even when wrong
- They handle known facts reasonably but outdated knowledge is an issue
- They make up answers for unknowable questions
- Arithmetic and logic are surprisingly weak

**Outcome:** Develop healthy skepticism about LLM outputs.

---

### Hands-On: Try It (Day 5, Part 2)

**Objective:** Measure success rate on a specific task.

**What to do:**

1. **Choose a task you care about.** Examples:
   - Extracting customer names from emails
   - Classifying support tickets by urgency
   - Summarizing a document in 3 sentences
   - Translating a sentence to another language

2. **Create a test set of ~20 examples:**

   ```
   Task: Classify support email by urgency (High, Medium, Low)

   Example 1:
   Email: "The system is down. No one can log in."
   Correct answer: High

   Example 2:
   Email: "When will the new feature be available?"
   Correct answer: Low

   Example 3:
   Email: "We've lost access to our data and can't recover it."
   Correct answer: High

   (... 17 more examples)
   ```

3. **Run the model on each example:**

   ```python
   correct_count = 0
   total_count = 0

   for example in test_set:
       response = model.ask(example["email"])
       predicted = response.strip()
       expected = example["correct_answer"]

       if predicted == expected:
           correct_count += 1
       total_count += 1

   accuracy = correct_count / total_count
   print(f"Accuracy: {accuracy * 100:.1f}%")
   ```

4. **Analyze failures:**
   - Which examples did the model get wrong?
   - Why did it fail?
   - Can you improve the prompt?

**What to observe:**

- Success rate on real tasks is often lower than expected
- Some examples are harder than others
- Failures reveal what the model struggles with

**Outcome:** Learn to measure and report task success honestly.

---

### Summary: Unit 1.5

LLMs are strong at rewriting, extracting and classifying but weak at arithmetic, complex reasoning and staying factual about unknowns. They hallucinate confidently. Measure success rate on your task, not capability on benchmarks. Don't trust output accuracy without testing.

---

## Unit 1.6: Choosing a Model

### Learning Objectives

- Understand the difference between closed and open models
- Know why "best model" depends on your constraints
- Learn a systematic model-selection process
- Use measurements, not rankings, to decide

### Closed vs Open Models

**Closed models (API-based):**

- Provided by companies like OpenAI, Anthropic, Google
- You call them via API; you don't see the weights
- Latest capabilities, but less control
- Privacy: your data goes to the provider's servers (check their terms)
- Cost: per-token pricing
- Examples: GPT-4o, Claude 3.5 Sonnet, Gemini 2.0

**Open models (self-hosted):**

- Weights are published (Llama 3.1, Mistral, etc.)
- You download and run them yourself
- Full control and privacy (data stays on your servers)
- Cheaper at scale (no per-token API fees)
- You manage the server and updates
- Older capabilities (open models trail closed ones)
- Examples: Llama 3.1 (70B), Mistral 7B, Qwen 14B

**Choosing:**

- **API:** when you want the best capability without server management
- **Self-hosted:** when you have privacy requirements, high volume, or specialized needs

---

### No Single Best Model

"What's the best model?" is like asking "What's the best car?"

Best for what? The answer depends on your specific constraints:

- **Task:** Coding, translation, summarization, reasoning
- **Budget:** Cost per request matters; sometimes a cheap model is worth 2% accuracy loss
- **Speed:** TTFT and generation latency
- **Context length:** Some tasks need 100k+ tokens
- **Privacy:** Can data leave your infrastructure?
- **Features:** Vision, code execution, real-time input handling

---

### Knowledge Cutoffs

Every model has a training data cutoff—a date after which it doesn't know current events.

- Claude 3.5 Sonnet: April 2024
- GPT-4o: October 2023
- Llama 3.1: September 2024

Anything newer must be supplied via context (documents, web search, tools).

---

### Pin Your Version

Model names can be misleading. The model behind a name might change.

```python
# Bad: might change after an update
response = openai.ChatCompletion.create(model="gpt-4", ...)

# Good: specific version
response = openai.ChatCompletion.create(model="gpt-4-0125-preview", ...)
```

When a provider updates "gpt-4" to a new version, your app's behavior might change. Pin specific versions and re-test when you decide to upgrade.

---

### The Model-Selection Process

**Step 1: Define the task.**

- What are you asking the model to do?
- Examples: write code, classify text, translate, summarize

**Step 2: Shortlist by leaderboards and public info.**

- Check benchmarks and reviews
- Narrow to 3–5 candidates

**Step 3: Create a small test set.**

- 10–20 examples representative of your actual use
- Include edge cases

**Step 4: Test each model on your test set.**

```python
test_set = [
    {"input": "...", "expected_output": "..."},
    # ... more examples
]

for model_name in ["gpt-4o", "claude-3-5-sonnet", "llama-3.1"]:
    correct = 0
    total_cost = 0
    total_latency = 0

    for example in test_set:
        start = time.time()
        response = call_model(model_name, example["input"])
        latency = time.time() - start

        is_correct = evaluate(response, example["expected_output"])
        cost = estimate_cost(response)

        correct += is_correct
        total_cost += cost
        total_latency += latency

    accuracy = correct / len(test_set)
    print(f"{model_name}: {accuracy*100:.1f}% accuracy, ${total_cost:.2f} total, {total_latency/len(test_set):.2f}s avg latency")
```

**Step 5: Compare on quality, speed and cost.**

```
GPT-4o: 94% accuracy, $0.15 per test set, 0.8s avg latency
Claude: 91% accuracy, $0.12 per test set, 0.6s avg latency
Llama 3.1: 87% accuracy, $0.02 per test set, 2.1s avg latency
```

Which trade-off fits your needs? If speed matters, Claude wins. If cost matters, Llama wins. If accuracy is critical, GPT-4o wins.

**Step 6: Analyze failures.**

- Which examples did each model get wrong?
- Are the failures tolerable for your use?
- Can a better prompt help?

**Step 7: Decide.**
Pick the model that best fits your constraints. You'll often trade a bit of accuracy for speed or cost.

---

### Five Things to Compare

1. **Quality on your task:** The accuracy/success rate you measured
2. **Speed:** TTFT and generation tokens per second
3. **Cost per request:** Estimated from your test set
4. **Input capacity:** Maximum tokens, vision, audio support
5. **Privacy/feature needs:** Data handling, compliance, fine-tuning support

---

### Hands-On: Try It (Day 6)

**Objective:** Systematically choose a model for a task you care about.

**What to do:**

1. **Define a task** (use an actual problem from your work or a realistic example):

   ```
   Task: Classify customer support emails by category
   (Billing, Technical, Feature Request, Other)

   Success criteria: >90% accuracy
   Budget: <$0.01 per email
   Latency: <1 second per email
   ```

2. **Create a test set (~20 examples):**

   ```
   Example 1:
   Email: "I was charged twice for my subscription."
   Category: Billing

   Example 2:
   Email: "The app crashes when I click the settings button."
   Category: Technical

   ... (18 more)
   ```

3. **Shortlist models:**
   - GPT-4o
   - Claude 3.5 Sonnet
   - Llama 3.1 70B (if you can run it or access it via an API)
   - Maybe a fast model like GPT-4o-mini

4. **Test each model:**

   ```python
   import time

   models = [
       ("gpt-4o-mini", 0.15, 0.45),  # (name, input_cost_per_mtok, output_cost_per_mtok)
       ("gpt-4o", 5.0, 15.0),
       ("claude-3-5-sonnet", 3.0, 15.0),
   ]

   for model_name, input_cost, output_cost in models:
       correct = 0
       tokens_in = 0
       tokens_out = 0
       latency_list = []

       for example in test_set:
           start = time.time()
           response = call_model(model_name, example["email"])
           latency_list.append(time.time() - start)

           predicted_category = parse_response(response)
           if predicted_category == example["category"]:
               correct += 1

           tokens_in += estimate_input_tokens(example["email"])
           tokens_out += estimate_output_tokens(response)

       accuracy = correct / len(test_set) * 100
       cost = (tokens_in * input_cost + tokens_out * output_cost) / 1_000_000
       avg_latency = sum(latency_list) / len(latency_list)

       print(f"{model_name}:")
       print(f"  Accuracy: {accuracy:.1f}%")
       print(f"  Cost: ${cost:.4f} (total for test set)")
       print(f"  Avg latency: {avg_latency:.2f}s")
       print()
   ```

5. **Decide:**
   - Which model meets your criteria?
   - If none do, should you improve your prompt?
   - Is the trade-off acceptable?

6. **Document your choice:**

   ```
   Model: Claude 3.5 Sonnet
   Reason: 92% accuracy, $0.008 per email, 0.7s latency
   Alternatives considered:
   - GPT-4o-mini: 89% accuracy, slightly cheaper, but accuracy mattered more
   - Claude 3 Opus: 95% accuracy, but 3× the cost, not worth the 3% improvement
   ```

**What to observe:**

- Fast models often are "good enough" for less critical tasks
- Accuracy, cost and speed always involve trade-offs
- Your specific test set reveals strengths and weaknesses
- Leaderboard rank ≠ best for your task

**Outcome:** Able to systematically choose a model for your use case.

---

### Summary: Unit 1.6

No single "best" model exists; it depends on your task, budget, speed and privacy needs. Closed models offer latest capabilities; open models offer privacy and cost savings at scale. Shortlist from rankings, but test on your own examples. Compare accuracy, speed, cost and features. Pin specific model versions. Document your choice.

---

## Hands-On Exercises

Below are structured practice problems to reinforce all six units. Complete at least 3–4 before moving to the assessment.

### Exercise 1: Token Cost Estimation

**Difficulty:** Easy

Your company is building a customer service chatbot. Each conversation:

- System prompt: 500 tokens
- Customer question: 200 tokens (average)
- Retrieved documents: 2,000 tokens
- Model response: 500 tokens (average)

You expect 1,000 customer conversations per day.

1. How many tokens per conversation?
2. How many tokens per day?
3. Using GPT-4o pricing ($5/$15 per million input/output tokens), estimate the daily cost.
4. Estimate monthly cost (30 days).

**Solution:**

```
Tokens per conversation: 500 + 200 + 2,000 (input) + 500 (output) = 3,200
Daily: 3,200 × 1,000 = 3.2M tokens
Cost per conversation input: (500 + 200 + 2,000) / 1M × $5 = $0.0135
Cost per conversation output: 500 / 1M × $15 = $0.0075
Cost per conversation: $0.021
Daily cost: $0.021 × 1,000 = $21
Monthly cost: $21 × 30 = $630
```

### Exercise 2: Model Selection for Code Review

**Difficulty:** Medium

You need an AI to review code pull requests. Requirement: >95% accuracy at spotting bugs, cost <$0.05 per review.

You have three models:

- GPT-4o: 96% accuracy, costs $0.12 per review
- GPT-4o-mini: 88% accuracy, costs $0.02 per review
- Claude 3.5 Sonnet: 94% accuracy, costs $0.08 per review

Which do you choose? Why?

**Solution:**

- GPT-4o: Meets accuracy requirement (96%) but exceeds budget ($0.12 > $0.05)
- GPT-4o-mini: Within budget but misses accuracy (88% < 95%)
- Claude 3.5 Sonnet: Misses accuracy (94% < 95%) and budget ($0.08 > $0.05)

**None fully meet requirements.** Options:

1. Increase budget to $0.12 and use GPT-4o
2. Accept 94% accuracy and use Claude 3.5 Sonnet at $0.08
3. Find a better prompt that improves GPT-4o-mini's accuracy
4. Combine models: use mini first (cheap), escalate uncertain cases to GPT-4o (expensive but rare)

### Exercise 3: Context Management

**Difficulty:** Medium

You're building a Q&A system. Your context limit is 8,000 tokens. Break down where tokens should go:

- System instructions: ?
- User question: ?
- Retrieved documents: ?
- Buffer (safety margin): ?

Allocate 8,000 tokens.

**Solution (one approach):**

```
System instructions: 300 tokens (clear rules, examples)
User question: 200 tokens (usually much less)
Retrieved documents: 6,500 tokens (the bulk; maximize relevance)
Buffer: 1,000 tokens (for unexpected growth)
Total: 8,000
```

Reasoning: Most of the budget should go to documents since they determine answer quality. Instructions should be tight and efficient. Always keep a buffer.

### Exercise 4: Failure Mode Identification

**Difficulty:** Medium

For each scenario, identify the most likely failure mode and suggest a mitigation:

1. **Scenario:** "What was the average temperature in Mumbai on March 15, 2025?"
   - Failure: Hallucination (future date, specific data)
   - Mitigation: Provide access to real-time weather data via a tool/API

2. **Scenario:** "Translate this 50,000-word document from English to Hindi"
   - Failure: Quality degradation due to context length; inconsistent terminology
   - Mitigation: Break the document into chunks, translate separately, then join

3. **Scenario:** "Count the number of objects in this image" (image with 20+ small objects)
   - Failure: Undercounting (models are weak at counting)
   - Mitigation: Don't rely on the model; use traditional computer vision

---

## Assessment and Self-Check

### Knowledge Checks (Answer without looking back)

1. **Explain next-token prediction in two sentences.**
2. **Why do Hindi texts cost more tokens than English?**
3. **What's the difference between context and knowledge?**
4. **When should you use temperature=0 vs temperature=1.5?**
5. **Name three LLM failure modes and why they happen.**
6. **You have two models: A (90% accurate, $0.10 per request) and B (95% accurate, $0.50 per request). How would you choose?**

**Grading yourself:**

- 1-2 correct: Re-read Units 1.1–1.6
- 3-4 correct: Solid understanding; focus on the units you missed
- 5-6 correct: Ready for Module 2

### Practical Assessment

**Build a small model-selection project:**

1. Choose a task (classification, summarization, extraction—anything real to you)
2. Gather 15–20 test examples
3. Compare 2–3 models on accuracy, cost and latency
4. Write a 1-paragraph recommendation explaining your choice

**Deliverable:**

```
Task: [Your task]
Test set size: [Number of examples]
Models compared: [List]
Results:
- Model A: [accuracy] accuracy, [cost], [latency]
- Model B: [accuracy] accuracy, [cost], [latency]
Recommendation: [Model name] because [reason]
```

---

## Common Mistakes and How to Avoid Them

### Mistake 1: Assuming Leaderboard Rank = Best for Your Task

**Wrong:** "Claude 3.5 Sonnet tops the benchmarks, so I'll use it."

**Right:** "Claude 3.5 Sonnet is strong on general benchmarks, but I'll test it on my specific task. Maybe a faster/cheaper model is good enough."

**Avoid by:** Always testing on your own examples, not just trusting public benchmarks.

---

### Mistake 2: Ignoring Token Costs

**Wrong:** "I'll send all my documents to the model every time. The API will handle it."

**Right:** "I'll calculate the tokens I'll use daily and monthly. If it's expensive, I'll use retrieval to send only relevant documents."

**Avoid by:** Tokenizing your actual data and estimating costs before building.

---

### Mistake 3: Trusting the Model's Confidence

**Wrong:** "The model answered with complete confidence, so it must be right."

**Right:** "The model is confident but might be hallucinating. I'll verify critical information with a tool or database."

**Avoid by:** Treating all LLM outputs as potentially unreliable. Verify important facts. Add guardrails.

---

### Mistake 4: Not Measuring Success Rate

**Wrong:** "The model seems pretty good at this task."

**Right:** "I tested it on 50 examples and got 87% accuracy. That's not reliable enough; I need to improve the prompt or add a tool."

**Avoid by:** Always creating a test set and measuring.

---

### Mistake 5: Using the Wrong Temperature

**Wrong:** "I'll use temperature=1.5 for extracting customer IDs because I want variety."

**Right:** "For extraction, I need consistency. I'll use temperature=0."

**Avoid by:** Matching temperature to task type (see Unit 1.4).

---

### Mistake 6: Forgetting the Knowledge Cutoff

**Wrong:** "The model will answer questions about recent events."

**Right:** "The model's knowledge cuts off at April 2024. For current events, I'll need to provide documents or use web search."

**Avoid by:** Always checking the cutoff date and planning for real-time data if you need it.

---

## Further Reading and Resources

### Foundational Papers and Concepts

- **"Attention Is All You Need"** (Vaswani et al., 2017): The transformer architecture that powers modern LLMs.
  - Read the abstract and introduction; the full paper is technical.
- **"Language Models are Unsupervised Multitask Learners"** (Radford et al., 2019): The GPT-2 paper explaining scaling laws.

### Blogs and Articles

- **OpenAI API Documentation:** <https://platform.openai.com/docs>
  - Practical guides on tokenization, API usage, best practices

- **Anthropic's Claude Documentation:** <https://docs.anthropic.com>
  - Guides on prompting, token counting, API usage

- **Google's LLM Concepts:** <https://cloud.google.com/vertex-ai/docs>
  - Overview of model types and when to use each

### Tools and Libraries

- **Tokenizers:**
  - `tiktoken` (OpenAI): <https://github.com/openai/tiktoken>
  - `transformers.AutoTokenizer` (Hugging Face): <https://huggingface.co/docs/transformers>

- **Model Comparison:**
  - LMSYS Chatbot Arena: <https://huggingface.co/spaces/lmsys/chatbot-arena>
  - Hugging Face Open LLM Leaderboard: <https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard>

- **Cost Calculators:**
  - OpenAI Pricing: <https://openai.com/pricing>
  - Anthropic Pricing: <https://www.anthropic.com/pricing>

### Books

- **"Build a Large Language Model (From Scratch)"** by Sebastian Raschka (2024)
  - Excellent for understanding how LLMs work under the hood

- **"Prompt Engineering Guide"** by DAIR.AI: <https://www.promptingguide.ai>
  - Free online guide to prompting techniques

### Courses

- **DeepLearning.AI Short Courses:** <https://www.deeplearning.ai>
  - Free, short courses on prompt engineering, LangChain, etc.

- **Hugging Face Course:** <https://huggingface.co/course>
  - Free course on transformers and NLP

### Stay Current

- **Papers with Code:** <https://paperswithcode.com> (LLM section)
- **ArXiv:** <https://arxiv.org> (search for "LLM" or specific models)
- **Twitter/X:** Follow researchers and practitioners in the field
- **Discord Communities:** Hugging Face Discord, OpenAI forums, etc.

---

## Conclusion

You've now completed Module 1: LLM Fundamentals. You understand:

- How LLMs work (next-token prediction)
- What they cost (tokens and money)
- How fast they are (latency components)
- What they're good and bad at (failure modes)
- How to choose one (systematic evaluation)

**Next steps:**

1. **Apply:** Choose a task relevant to your work and follow the model-selection process (Unit 1.6 hands-on)
2. **Move forward:** Module 2 teaches you to build applications on top of LLMs using LangChain
3. **Experiment:** Spend time calling different models and feeling out the differences

**Key takeaway:** LLMs are powerful but not magic. They're probabilistic pattern-matchers. Measure them on your task, understand their limitations, and build safeguards around them.

---

**End of Module 1**

**Duration completed:** ~12 hours  
**Recommended next:** Take a 1–2 day break, then move to Module 2 (Building with LLM APIs)
