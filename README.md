# Megatron-LLaMA: Easy, Fast and Affordable Training of Your Own LLaMA

As is known to all, LLaMA has become one of the greatest work in the open-source community of large language models (LLMs). LLaMA incorporates optimization techniques such as BPE-based tokenization, Pre-normalization, Rotary Embeddings, SwiGLU activation function, RMSNorm, and Untied Embedding. We have witnessed the outstanding results of LLaMA in both objective and subjective evaluations. LLaMA develops versions of 7B, 13B, 30B, and 65B/70B in model sizes. In the open-source community, there have been many successful variants based on LLaMA via continuous-training / supervised fine-tuning (such as Alpaca, Vicuna, WizardLM, Platypus, Minotaur, Orca, OpenBuddy, Linly, Ziya) and training from scratch (Baichuan, QWen, InternLM, OpenLLaMA). These works further demonstrate LLaMA's prominent capabilities in tasks such as long-context comprehension, long-context generation, code writing, mathematical problem solving, tool usage, etc.

However, it is often very expensive for developers to try out their own designs on LLaMA, as training or fine-tuning one's own LLM requires powerful computational resources. Typically, GPUs with large memory or distributed clusters composed of multi-GPU devices are essential for training LLMs. Megatron-LM is a distributed training solution that integrates tensor parallelism (TP), pipeline parallelism (PP), and sequence parallelism (SP). When training models with tens-of- or hundreds-of-billion parameters, it tends to take full advantage of the hardware resources. The resource utilization can reach far beyond the publicly available versions of LLaMA (implemented based on Huggingface and DeepSpeed). Nevertheless, native Megatron-LM would suffer from the communication bottleneck of the distributed optimizer during training at an extremely large scale.

Therefore, to facilitate the training of LLaMA-based models and reduce the cost on occupying hardware resources, Alibaba decides to release the internal optimized Megatron-LLaMA training framework to the community. Megatron-LLaMA makes the following contributions:

(i) **A standard implementation of LLaMA in Megatron-LLaMA**: It is easy to obtain the LLaMA code from Huggingface, which does not involve the various parallel methods provided by Megatron-LM. Megatron-LLaMA offers a standard implementation of LLaMA in Megatron-LM, allowing developers to configure the optimization techniques on demand. We will continue to release features such as Alibi and FlashAttention2 in the future.

(ii) **Efficient communication-computation parallelism**: Similar to [DeepSpeed ZeRO Stage 2](https://www.microsoft.com/en-us/research/blog/zero-2-deepspeed-shattering-barriers-of-deep-learning-speed-scale/), Megatron-LM implements [`DistributedOptimizer`](https://github.com/NVIDIA/Megatron-LM#distributed-optimizer) that partitions the gradient and optimizer state, significantly reducing GPU memory usage. However, the solution provided by Megatron-LM does not fully overlap GPU computation with communication, resulting in underutilization of hardware resources. Building upon the original `DistributedOptimizer` and ZeRO-Stage-2, Megatron-LLaMA proposes a novel approach for gradient and optimizer state sharding, achieving the following benefits without compromising precision: a) extremely high parallelism between communication and computation; b) highly efficient utilization of communication bandwidth; c) lower GPU memory usage. Consequently, Megatron-LLaMA enables higher training throughput on the same hardware configuration than the vanilla Megatron-LM.

(iii) **Utilities**: Megatron-LLaMA supplements several utilities and improves the checkpoint mechanism in Megatron-LM, including: a) Distributed checkpoint saving/restoring to speedup. This also provides abstract filesystem interfaces for easily integrating distributed file systems such as [HDFS](https://hadoop.apache.org/docs/r1.2.1/hdfs_design.html); b) Convenient interface for weight conversion from/to the HuggingFace format, facilitating the delivery to the downstream tasks after pretraining; c) Support for [Tokenizers](https://huggingface.co/docs/transformers/main/model_doc/llama#transformers.LlamaTokenizer) in HuggingFace transformers library.

**Megatron-LLaMA makes large-scale training of LLaMA models fast, affordable and scalable.**

**Efficiency and Affordability**: The Megatron-LM techniques make LLaMA training fast and affordable. Suppose that we train our own LLaMA-13b model on four 8xA100-80GB devices. The following table depicts the training cost and TFLOPS of DeepSpeed implentation and Megatron-LM implentation of LLama. According to the Azure pricing, Megatron-LLaMA saves $1037 compared to DeepSpeed when consuming 10 billion tokens.

|  | DeepSpeed (HF) | Megatron-LLaMA |
| ------ | ------ | ------ |
| Training cost | 49.7 hours ($5482) | 40.3 hours ($4445) |
| Training Model TFLOPS | 146 |180 |

**The global batch size is set to 2048 via gradient accumulation (GA).*

**We enable [FlashAttention](https://arxiv.org/abs/2205.14135) in the HF/DeepSpeed implementation.*

**Excellent Scalability**: The `OverlappedDistributedOptimizer` in Megatron-LLaMA introduces the high parallelism between computation and communication, regardless the number of gradient accumulation. We demonstrate the average tokens per second on each GPU in the following table when we try to reproduce the LLaMA training (with 8xA100-80GB devices and 4x200Gbps RDMA inter-bandwidth). Based on this metric, the scaling ratio of Megatron-LLaMA with OverlappedDistributedOptimizer can reach 0.85 when scaling from 32 GPUs to 512 GPUs, while Megatron-LLaMA with DistributedOptimizer can only achieve around 0.7.

|  | 256xA100 80GB | 512xA100 80GB |
| ------ | ------ | ------ |
| Megatron-LLaMA with OverlappedDistributedOptimizer | 1890 (23.9 days) | 1845 (12.2 days) |
| Megatron-LLaMA with DistributedOptimizer| 1630 (27.8 days) | 1430 (15.8 days) |


# OverlappedDistributedOptimizer

In the vanilla Megatron-LM, users can leverage [`DistributedOptimizer`](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/distrib_optimizer.md) to partition gradients and optimizer states to reduce GPU memory occupation. After accumulated all gradients in GA, `DistributedOptimizer` employs a `ReduceScatter` operation to scatter the gradients to the corresponding ranks. Each rank then updates the local parameters, and then collect the remaining parameters through an `AllGather` operation from all the other ranks. However, we observe a significant overhead on communication under small GA settings (over 50% time consumption without GA). 

To mitigate the overhead, we try to overlap the collective communication with computation, according to the partition strategy in DeepSpeed ZeRO Stage-2. This strategy fails to scale. It takes too many small `Reduce` operations at large scale, which makes it under-utilize the inter-connection bandwidth.

We abstract the above problem into two aspects:
1. Finding the room for overlapping communication with computation logically.
2. Implementing a partition strategy that fully utilizes the room for overlapping and inter-connection bandwidth, without introducing overhead in term of communication volume.

In this case, we propose `OverlappedDistributedOptimizer `, with a novel partition strategy of gradients and optimizer states. The design principles are summarized as follows:

* Common optimizers such as Adam and SGD update each value in the parameters independently. Therefore, it is not necessary to keep each parameter as a whole.
* Any single collective communication operation should commit a sufficient amount of data, making full use of the communication bandwidth.
* No extra communication volume or GPU memory-copy should be involved.


**Brief introduction to OverlappedDistributedOptimizer**

![](docs/shuffle_grad_bucket.png "The partition strategy in Megatron-LLaMA")
<center>Figure 1. The partition strategy in Megatron-LLaMA</center>

As shown in Figure 1, all parameters are assigned to their respective `Buckets` during the initialization of `OverlappedDistributedOptimizer`. All the model parameters within a `Bucket` are complete, with each parameter belonging to only one `Bucket`. Conceptually, each `Bucket` is divided equally into *P* (the number of ranks of the data parallel group) shards. Each rank would be responsible for one shard. The `Buckets` would be placed in a local queue (Local grad bucket queue) to ensure the communication order. During the training process, the data parallel groups exchange the required gradients at the `Bucket` level through collective communication.

![](docs/communicate_params.png "The communication mechanism in Megatron-LLaMA")
<center>Figure 2. The communication mechanism in Megatron-LLaMA</center>

`OverlappedDistributedOptimizer` incorporates an efficient communication mechanism over the `Buckets`. `OverlappedDistributedOptimizer` initializes a local buffer called `PartitionedParameter` with a size equal to the sum of sizes of all parameters that the current rank is responsible for. The respective parameters are taken from the pre-sharded model parameters and assigned to the `PartitionedParameter`. Besides, a buffer called `PartitionedGradient`, with the same size as `PartitionedParameter`, is created to store the gradients corresponding to the `PartitionedParameter`. Then, Megatron-LLaMA's communication mechanism mainly consists of the following three procedures:

a) As shown in Figure 2-(i), once a parameter's gradient is obtained, the gradient would be copied to the corresponding position in the Bucket. Once all gradients for the parameters in a Bucket are collected, a single `ReduceScatter` operation is performed to exchange the gradients, with the corresponding position in the `PartitionedGradient` as destination.

b) As shown in Figure 2-(ii), each rank updates `PartitionedParameter` by the `PartitionedGradient ` once all `ReduceScatter` operations are finished.

c) As shown in Figure 2-(iii), each rank re-constructs the full parameters from all the other ranks through `AllGather` with the logical `Bucket`.


Specifically, we reduce the memory copy and GPU memory occupation through the following approaches:

a. During the initialization of `OverlappedDistributedOptimizer`, a buffer called `ParameterBuffer` is allocated with the same size as the sum of all parameter sizes, and all model parameters are actually placed in `ParameterBuffer`. The destination addresses for re-constructing the full parameters via `AllGather` can directly reference to the corresponding positions in `ParameterBuffer`. It avoids the temporary memory allocation and reduces GPU memory copy. (This optimization is inspired by DeepSpeed).

b. Once copying gradients to the `Bucket` has been complete, the original space for gradients can be released, reducing GPU memory usage. Additionally, the memory for `Bucket` can also be released after the `ReduceScatter` operation. On top of this, we introduce a *Buffer Alternation* mechanism to avoid the issue of memory fragmentation caused by frequent memory allocation and deallocation.



## Using Megatron-LLaMA

### Launch a train task

You can use the same launching method as in [Megatron-LM Usage](./original_README.md#contents). Beyond that, we produce:

#### A. Weight conversion tool

This tool helps convert the format of paramters between Megatron-LLaMA/Megatron-LM and Huggingface format.

**HuggingFace to Megatron-LLaMA**
For LLaMa:
```
sh tools/checkpoint_conversion/hf_to_megatron.sh  
```
For Baichuan:
```
sh tools/checkpoint_conversion/baichuan_hf_to_megatron.sh  
```

**Megatron-LLaMA to HuggingFace**
For LLaMa:
```
sh tools/checkpoint_conversion/megatron_to_hf.sh
```
For Baichuan:
```
sh tools/checkpoint_conversion/baichuan_megatron_to_hf.sh  
```

#### B. Launching scripts

**Single-node launching**

```
sh examples/LLaMA/LLaMA_13_standalone.sh
```

**Distributed launching**


```
sh examples/LLaMA/LLaMA_13_slurm.sh
```

In particular, we recommend to increase the micro-batch size to fully occupy the GPU memory so that the hardware utilization will be maximized.

**Customized arguments in Megatron-LLaMA**

| Argument | Specification |
| ------ | ------ |
| `--overlapped-distributed-optimizer` | Enable the `OverlappedDistributedOptimizer`. Do not set `--use-distributed-optimizer` simultaneously. |
| `--reduce-bucket-size` | Set the size of the `Bucket` in `OverlappedDistributedOptimizer`. Default to 5e8. Larger `Bucket` indicates higher utilization of inter-DP group bandwidth; Smaller `Bucket` bring opportunity for better parallelism between communication and computation. |
| `--tokenizer-type=PretrainedFromHF` | Use a Tokenizer from Huggingface (would be loaded via `transformers.AutoTokenizer`) |
| `--distributed-checkpointing` | Distributed saving of checkpoint files. |

Megatron-LLaMA supports the canonical [data prepocessing](https://github.com/NVIDIA/Megatron-LM/blob/main/README.md#data-preprocessing) and [evaluation](https://github.com/NVIDIA/Megatron-LM/blob/main/README.md#evaluation-and-tasks) as mentioned in the Megatron-LM library.

### Future work

At present, we are actively working on the following items:

* Release the configurations and optimization scheme for the 30B and 65B/70B LLaMA model training
* Supplement the modifications in models such as Alibi and FlashAttention2
* Support for LLMs with other model structures
* We encourage the community to engage in discussions aimed at making LLaMA training even more accessible, efficient, and cost-effective

### License

Megatron-LLaMA is developed by Aicheng Technology, Alibaba Group and is based on the Megatron-LM project(https://github.com/NVIDIA/Megatron-LM) from Nvidia. Code is distributed under the Apache License (Version 2.0). This product contains various third-party components under other open source licenses. See the NOTICE file for more information.

### Credits

The following repositories are used in Megatron-LLaMA, either in close to original form or as an inspiration:

[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)

[LLaMA](https://github.com/facebookresearch/llama)

[DeepSpeed](https://github.com/microsoft/DeepSpeed)

