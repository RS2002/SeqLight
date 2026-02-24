# Light Decomposition RL

## How to Run

### 1. Policy Pre-training by BC

```shell
python bc.py --simple_layout
```

### 2. AIRL Training Phase

```shell
python airl.py --simple_layout --use_bc_loss --use_her --her_in_expert --her_in_bc --pretrained_path <trained model path from Phase 1>
```

### 3. RL Fine-tuning Phase

```shell
python ppo.py/grpo.py --simple_layout --use_bc_loss --use_her --her_in_bc --pretrained_path <trained model path from Phase 2>
```

### 4. Evaluate

```
python eval.py --model_path <trained model path from Phase 3>
```

