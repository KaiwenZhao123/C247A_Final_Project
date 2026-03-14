# C247A_Final_Project
## Group Members

- **Jingze Fu** — UID: 906202950 — fujingze1019@g.ucla.edu  
- **Mu Li** — UID: 306780665 — lim29@ucla.edu  
- **Maoqi Xu** — UID: 006770427 — mqxu@ucla.edu  
- **Kaiwen Zhao** — UID: 606763706 — kaiwenzhao2@g.ucla.edu
## Included files for Best Model
1. modules.py
   - Modified model/module definitions.
   - This file contains the main code changes relative to the baseline.

2. lightning.py
   - Modified training / PyTorch Lightning related code.
   - This file was updated to support the experiments and training setup used in our project.

3. gru_ctc.yaml
   - Configuration file for the GRU-CTC experiment.
   - Contains the hyperparameter and training settings used for this model.

4. best.ckpt
   - Best-performing checkpoint (model weights) obtained during training.
   - This is the checkpoint corresponding to the best validation performance among our runs.

Notes:
- Except for the files listed above, the rest of the codebase is unchanged from the baseline.
- To run the project, please combine these files with the original baseline code.
- The checkpoint is provided for reproducibility / reference of the final trained model.

This code submission only includes the files that were modified from the baseline implementation.
All other files are identical to the provided baseline and are therefore not included again.

- These models were trained and tested locally using the following commands:
  python -m emg2qwerty.train user="single_user" model="gru_ctc" trainer.accelerator=gpu trainer.devices=1 trainer.max_epochs=400
  python -m emg2qwerty.train user="single_user" model="gru_ctc" checkpoint="best.ckpt" train=False trainer.accelerator=gpu decoder=ctc_beam hydra.launcher.mem_gb=64

## Best Results Summary

Our group has included all implemented models in this repository. The best-performing models from each experiment branch are summarized below.

| Branch | Best model / setting | Key idea | Val CER ↓ | Test CER ↓ |
|---|---|---|---:|---:|
| Shared | TDSConv baseline | Baseline model | 18.48 | 21.85 |
| A | 🏆 **CNN+GRU hybrid** | Convolutional front-end with recurrent temporal modeling | **10.04** | **9.44** |
| A | CNN+LSTM hybrid | Convolutional front-end with LSTM temporal modeling | 14.07 | 14.76 |
| B | Dilated Residual TCN | Dilated temporal convolutions with shared beam decoding | 17.77 | 19.47 |
| C | Dual-stream late fusion | Separate left/right streams with fusion after temporal encoding | 12.36 | 11.59 |

In summary, the **CNN+GRU hybrid** from **Branch A** achieved the best overall performance.
