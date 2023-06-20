## Transformer-based sentiment analysis for flare prediction

**Running the code**

Run the autoregressive denoising task to warm up the model weights. 

```python
python running_inputter.py
```

Run the classification starting with the weights from the denoising task.
```python
python running_classifier.py
```

Train separate CNN model. 
```python
python running_cnn.py
```

Derive saliency maps from the train CNN classifier by applying guided Grad-CAM to all instances.
```python
python running_grad_cam.py
```