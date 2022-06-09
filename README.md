# Radio Estimation U-Net

Current Update: 3.0

Radio Estimation U-Net or REUNET for short is a Radio Mapping tool using Convolutional Neural Network. REUNET takes sparse sensor measurements as input (Rx) in order to generate the Radio Map.
You may request the models (.pt files), documentation and report from me at kokchinyi01@gmail.com.

I also included Kriging (using OpenTurns) in this respository as I used it as a yardstick for REUNET.

## Dataset

The dataset I used is [RadioMapSeer](https://drive.google.com/file/d/1PTaPpLOKraVCRZU_Tzev4D5ZO32tpqMO/view) and it can be found in [RonLevie Repository - RadioUNet](https://github.com/RonLevie/RadioUNet).

For Multi-Transmitters/64x64/128x128 images, please request them from me as well. However, for 64x64/128x128 images, you can generate them on the fly. As for multi-transmitters images, the code provided in this respository is capable of generating them from RadioMapSeer.

# File Structure

Ensure that the files are in this structure:

- lib
  - \_\_pycache\_\_ /
  - loaders.py
  - modules.py
  - test.py
  - trainandtest.py
- Models
  - whatevermodel.pt
- REUNETData
  - gain/IRT4/\*.png
  - png/buildings_complete/\*.png
- base.py
- kriging.py
- reunet_demo.ipynb
- REUNET.py

Keep in mind that `Models` and `REUNETData` are not in this Github directory.

# Demo

Please ensure that you have installed the required dependencies using:

```bash
$ pip install -r requirements.txt
```

Open `reunet_demo.ipynb` and you may start the Demo by changing basic parameters.

Firstly, load the model by specifying the _model_path_:

```python
model_path = 'Models/whatevermodel.pt'
```

Secondly, edit the _folder_paths_ according to to the Dataset folder name. In this case, I have `REUNETData` as the single transmitter folder and `REUNETDataMulti` as the multi transmitters folder.

```python
folder_paths = {
    'multi' : 'REUNETDataMulti',
    'single' : 'REUNETData'
}
```

Lastly, you may change the parameters of the demo. Take note that there are only **680** maps available for REUNETDataMulti.

```python
folder_path = folder_paths['single']    # Edit single/multi
idx = 618                              # Map idx choose from 0 to 700.
measurements = 50                      # Total number of sensor measurements

# Blocking radius | Put 0 if you do not want to block. Ranges from 0 to 128. | Indicate min 1 even if you use your own image
# Might have error if the value is too high as there are no locations to place the sensors. (ie. dont block the whole map)
blocking_radius = 0
```

Click :arrow_forward: **Run All** and scroll to the bottom to see the results.

![Demo Image](https://github.com/thekopiman/REUNET/blob/master/readmeimages/demo_image.png)

```bash
MSE Results:
REUNET - 39.10242374641288
Kriging - 71.27807225243905

SSIM Results:
REUNET - 0.8654114954167863
Kriging - 0.37974226033858455

Time Results:
REUNET - 0.8382966999999972
Kriging - 0.20946469999999806
```

Please check the documentations for an indepth explanation of other parameters.

# Training

Due to the amount number of ways to conduct Transfer Learning, having a blanket template for training in this case is impractical. As such, I will briefly outline 3 main steps for Training - DataLoader, Model Load and Train Model. I added a sample training jupyter notebook in this respository - `train-n-test.ipynb`. You may use this jupyter notebook for reference when creating your custom training parameters.

## DataLoader

Firstly, the data has to be loaded in:

```python
Radio_train = loaders.RadioUNet_s_sprseIRT4(phase="train", simulation="IRT4", cityMap="complete",fix_samples = 300,n_iterations=10)
Radio_val = loaders.RadioUNet_s_sprseIRT4(phase="val", simulation="IRT4", cityMap="complete",fix_samples = 300,n_iterations=10)
Radio_test = loaders.RadioUNet_s_sprseIRT4(phase="test", simulation="IRT4", cityMap="complete",fix_samples = 300,n_iterations=10)

image_datasets = {
    'train': Radio_train, 'val': Radio_val
}

batch_size = 15

dataloaders = {
    'train': DataLoader(Radio_train, batch_size=batch_size, shuffle=False, num_workers=1),
    'val': DataLoader(Radio_val, batch_size=batch_size, shuffle=False, num_workers=1)
}
```

## Model Load

Secondly, the model has to be loaded in. (You can do this step first if you like to) If you like to train from scratch, it is not necessary to load the model (`model.load`) but it is still necessary to initialise it (`model - modules.RadioWNet`).

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model =modules.RadioWNet(inputs=2, phase="firstU")
model.load_state_dict(torch.load('NewRMModel/Trained_Model_FirstU_1.pt', map_location=device))

```

## Train Model

Lastly, you can train the model as such

```python
optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

Training_loaddata = TrainModel(dataloaders)

model = Training_loaddata.train_model(model, optimizer_ft, exp_lr_scheduler, targetType = 'dense', num_epochs=10,WNetPhase="firstU",train_log=True)
```

## Remember to save the model (.pt) file

```python
torch.save(model.state_dict(), 'NewRMModel/Trained_Model_FirstU_1.pt')
```
