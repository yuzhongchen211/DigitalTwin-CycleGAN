 # Train in Simulation

## 1. Collecting Training Data

```python
python train_init.py
# collected training data and saved as replaybuffer
```

## 2. Training 

```python
python train_obs_pos.py
# training and saved the detection model 
```

## 3. Test

```python
python testdemo.py
# test the successate rate
```

## Other

For the methods of CycleGAN and RetinaGAN, post training is necessary which regrad the GAN as a data augment methods.

**Required: ** Images and GAN augmented Images with labels.

**Return:** detection model

```python
python train_post.py
```

## Prepare for GAN training

The training of GAN need pure (not complex background for better performance) simulation images and labels. Therefore, we generate a euqal numbers of images between real images and simulation images.

```python
python utils.py
```

