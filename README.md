# MNIST-V0

## Learnable Parameters Navigation 

<img src="https://github.com/kishkath/S6-MNIST-V1/assets/60026221/dbbf36be-2034-465f-941a-77d0043de4f9" width = 720 height = 360>

<img src="https://github.com/kishkath/S6-MNIST-V1/assets/60026221/9aae722e-7a34-40f2-aa31-7235dc7f66ad" width = 720 height = 360>



## Part1: Backpropagation

Reference: https://towardsdatascience.com/understanding-backpropagation-abcc509ca9d0

* Backpropagation is a way of finding the gradients of loss & helps in updating the learnable parameters as in a way of making the model converge to global minima. 

* The Entire process of backpropagation is made in the Excel sheet: 

* Two Inputs are made which combinely make a input layer of 2 neurons, which include a single hidden layer, and a output layer which makes a three-layered Network(in case of input as a layer, else 2)

      First layer: 2 Input neurons are present, which has 4 weights to be learnt (W1,W2,W3,W4) 

      Second layer: A hidden layer with 2 neurons are present which in-take input neurons as a weight sum w.r.t to weights (W1,W2,W3,W4).

      Third layer: The outputs of second layer using newly formed weights (W5,W6,W7,W8) computes with an activation function and gets the loss. 


* The main procedure of backpropagation is that the obtained loss which inturn computes from last to first layer.

* The loss computes gradients w.r.t parameters, like the weights (W5,W6,W7,W8) finds their rate of change of loss w.r.t to change in weights using chain rule including activation functions in between.                                        
                                     
                * We Compute this derivatives dL/dw5, dL/dw6, dL/dw7, dL/dw8, dL/dw1, dL/dw2, dL/dw3, dL/dw4 using chain rule 
    
* Figure: Calculating derivatives of first layer from last

<img src="https://github.com/kishkath/S6-MNIST-V1/assets/60026221/2b6851d0-839c-48d8-9b3f-e7ae2a49ee07" width = 720 height = 360>

* Figure: Calculating derivatives of second layer from last
<img src="https://github.com/kishkath/S6-MNIST-V1/assets/60026221/c8e2b46b-bc7b-4f33-be96-108d954b26f8" width = 720 height = 360>
          
* Figure: varying error plots with varying learning rate values
<img src= "https://github.com/kishkath/S6-MNIST-V1/assets/60026221/33510b7d-f24a-49ce-ab0a-34a18d208e61" width = 720 height = 360>


## Part2: Achieving Desired Accuracy

* Designing the neural network for MNIST dataset such that it achieves the accuracy of 99.4%.
* The notebook contains all the related work for the achieving the accuracy: https://github.com/kishkath/S6-MNIST-V1/blob/main/ERA_S6_Finalised.ipynb
* The neural network is designed in a way to be as a lighter model.
                
                Architecture: Conv1 -> Conv2 -> MaxPool -> Conv3 -> Dropout -> Conv4 -> Conv5 -> Dropout -> Conv6 -> Dropout -> Conv7 -> GAP(3)
               'Its a lighter model right!'
               ---------------------------
               Total params: 16,050
               Input size (MB): 0.00
               Forward/backward pass size (MB): 0.66
               Params size (MB): 0.06
               Estimated Total Size (MB): 0.73
 
 * The last 6 epoch runs: 
              
              Epoch 16
              Train: Loss=0.0232 Batch_id=468 Accuracy=98.94: 100%|██████████| 469/469 [00:24<00:00, 18.98it/s]
              Test set: Average loss: 0.0002, Accuracy: 9949/10000 (99.49%)

              Adjusting learning rate of group 0 to 1.0000e-05.
              Epoch 17
              Train: Loss=0.0150 Batch_id=468 Accuracy=98.96: 100%|██████████| 469/469 [00:25<00:00, 18.68it/s]
              Test set: Average loss: 0.0002, Accuracy: 9950/10000 (99.50%)

              Adjusting learning rate of group 0 to 1.0000e-05.
              Epoch 18
              Train: Loss=0.0436 Batch_id=468 Accuracy=98.96: 100%|██████████| 469/469 [00:25<00:00, 18.70it/s]
              Test set: Average loss: 0.0002, Accuracy: 9952/10000 (99.52%)

              Adjusting learning rate of group 0 to 1.0000e-05.
              Epoch 19
              Train: Loss=0.0134 Batch_id=468 Accuracy=98.97: 100%|██████████| 469/469 [00:25<00:00, 18.55it/s]
              Test set: Average loss: 0.0002, Accuracy: 9948/10000 (99.48%)

              Adjusting learning rate of group 0 to 1.0000e-05.
              Epoch 20
              Train: Loss=0.0292 Batch_id=468 Accuracy=98.94: 100%|██████████| 469/469 [00:25<00:00, 18.41it/s]
              Test set: Average loss: 0.0002, Accuracy: 9954/10000 (99.54%)
                 

* Final result: "Achieved desired Test-Accuracy"
