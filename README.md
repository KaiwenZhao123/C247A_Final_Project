# C247A_Final_Project

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
