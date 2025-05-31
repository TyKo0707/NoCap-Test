# GPT-2 Benchmark

![logo](img/logo.png)
Hey there! Are you interested in LLMs? Do you like experimenting with neural networks, implementing different ideas and testing them out? Would you like to do that for a living? Then you're in the right place!
This is an official open test for (not only) people interested in joining [BottleCapAI](https://www.bottlecapai.com).

This project is a fork of [Modded-NanoGPT](https://github.com/KellerJordan/modded-nanogpt) :heart:, rewritten with minimal changes to run on a single GPU (e.g. RTX 3090/4090)

---

## Objective

Train a langauge model on a subset of the **FineWeb** dataset to reach a validation loss of **≤ 3.3821** as fast as possible using **1 GPU**.

You can achieve this by:
- making your model faster (so that it sees more data in shorter time)
- making your training more efficient (so that in less steps your model makes better progress).

---

## What's the point?

We're not here to optimize learning rates, CUDA kernels, or torch.compile flags.  
We're here to explore **algorithmic ideas that might scale**.

This benchmark is meant for:
- People with limited hardware
- People with ideas and curiosity

You're encouraged to try new techniques to speed up language modeling such as but not exclusively:
- Modify the loss function
- Add auxiliary losses (multi-token prediction?)
- Modify the architecture (Mixture of Experts? Different attention?)
- Come up with a different training algorithm
- Modify the training data

You're **not** expected to:
- Just bump up the learning rate
- Beat everyone with hyperparameter magic
- Do 50 runs to grid search Adam betas
- Benchmark arcane PyTorch flags
- Copy speedups from [Modded-NanoGPT](https://github.com/KellerJordan/modded-nanogpt)
- Modify a specific hidden layer size to align better with the number of TensorCores on your GPU 

We're interested in your own ideas, not how well you can copy other's. These ideas should be general and work on different setups and not be hardcoded to a very specific one.

You have a budget of **5B** tokens available for training, but the baseline only uses **2.5B**, so you've got room to train on more data if you make your model faster, or on less but *better* data. 

The dataset is pre-tokenized so that you don't have to do that yourself (saves time) but if you want to explore the original text, you can decode it using the GPT-2 tokenizer (`tiktoken.get_encoding("gpt2")`).

---

## Running the current baseline 

To run the current baseline, run the following commands.
```bash
git clone https://github.com/BottleCapAI/modded-nanogpt && cd modded-nanogpt
pip install -r requirements.txt

# you can skip this if you don't want to use W&B, in which case you should remove the --log_wandb argument from run.sh
wandb login
wandb sync wandb/run-20250410_203158-64s1zc1w # synchronizes the baseline run to your W&B account for reference

python data/cached_fineweb10B.py
./run.sh
```

---

## Benchmarks

Here’s the current leaderboard. If the baseline improves, we’ll update it here for reference.

**Train a neural network to ≤ 3.3821 validation loss on FineWeb using 1 GPU.**

| # | Record time | Description                                                   | Date     | Log | Contributors |
| - | - |---------------------------------------------------------------|----------|-----| - |
1 | 5.401 hours | [baseline](https://github.com/KellerJordan/modded-nanogpt) | 11/04/25 | [log](pylog124M/14e37fbb-cc64-4185-a1a7-5ef956b56ac7.log)   | [contributors-of-modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt)

Note: The current baseline used one RTX 4090. It took 4768 steps/iterations and used in total 2.5B tokens.

![wandb_loss](img/wandb_loss.png)

## Rules

1. Reach validation loss <= 3.3821 in shorter time.
2. Do **not** introduce new datasets, but feel free to modify the current.


If you use a different GPU than RTX 4090, benchmark the baseline and compare your speedup to that result, for example, if the baseline takes 10 hours on your setup, but your solution takes only 8 hours, then thats your speedup that you can report to us! Keep the comparison fair, if you increase the learning rate for your solution, try increasing it also for the baseline.

## Submission

To submit your results, run:
```bash
git bundle create <first name>-<last name>.bundle --all
```
Then send us your .bundle file to hey@bottlecapai.com

Do **not** create a public pull request.

---
## Technical Notes

While this project is designed to run on **1 GPU**, there are a few things to keep in mind:

- Batch Size, Sequence Length and Gradient Accumulation:
  The current setup requires ~ 13GB of GPU memory, which might not be available to you (if you have no GPU we suggest using [Google Colab](!https://colab.research.google.com/)), in which case, you might need to tune down some hyperparameters. We recommend starting with validation batch size - this one will not affect performance but validation will take a bit longer. Next, you might tune down batch size which you might then compensate by increasing gradient accumulation to retain the same effective batch size, be careful about changing learning rate and other hyperparameters should you change effective batch size.

- **torch.compile Considerations:**  
  On some RTX cards, aggressive kernel auto-tuning via `torch.compile` can lead to shared memory issues. If you encounter errors or persistent warnings (e.g., about insufficient SMs for max autotune GEMM mode), you may have to **disable `torch.compile`** or adjust your model settings accordingly. Although this may lead to slightly slower performance, it typically resolves hardware compatibility issues.

- Multi-GPU Runs:
  This code should be ready for distributed training, if you happen to have access to multiple GPUs. In that case, make sure that Gradient Accumulation Steps is divisible by number of GPUs.

---

### Comment on the target metric

The target metric is cross-entropy loss on the FineWeb val set. The goal of the speedrun is to obtain a probability model of language which assigns a probability of at least `math.exp(-3.3821 * 1048576)` to the first 1,048,576 tokens of the FineWeb valset. Hence, we allow evaluation at any sequence length, so long as we still have a valid probability model of language on the **entire** validation set.

