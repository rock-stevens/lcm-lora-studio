Safetensors Base SD and SDXL Models downloaded from the internet go in here.

Note: '*.safetensors' files only.

All of those Safetensors Models downloaded normally have sample generated images, 
plus a model card etc... that you can also download as well as the model itself.

The procedure for saving sample images for downloaded LoRA models, except everything is in one folder for the Safetensors Base (non-LoRA) models.

This is for the Safetensors Base models, Model card, and Sample Images.

All must be created by the user manually. (For now...)

That information goes in this directory, structured as so...

safetensors_models
├── mymodelnameabc.safetensors     <---the actual model *.safetensors file
├── mymodelnameabc                 <---folder for model card and images
│   ├── mymodelnameabc.md          <---markdown language model card
│   ├── image3124141.jpg           <---sample image
│   └── image3124141.txt           <---sample image generation text file
├── mymodelname123.safetensors     <---the actual model *.safetensors file
├── mymodelname123                 <---folder for model card and images
│   ├── mymodelname123.md
│   ├── image1.png
│   ├── image1.txt
│   ├── image2.png
│   ├── image2.txt
│   ├── image3.png
│   └── image3.txt
└── mymodelnameXYZ.safetensors     <---the actual model *.safetensors file
└── mymodelnameXYZ                 <---folder for model card and images
    ├── mymodelnameXYZ.md
    ├── image1.jpg
    ├── image1.txt
    ├── image2.webp
    └── image2.txt
    

Example: 

You downloaded a LoRA model named 'mymodelnameXYZ.safetensors', and saved it in the directory : 'safetensors_models'.

Now you want to grab the information about the LoRA model.

Create a directory in the 'safetensors_models' directory named named: 'mymodelnameXYZ'

Inside of the new 'mymodelnameXYZ' directory create a a file named: 'mymodelnameXYZ.md'

Open that with a right click 'Open with Notepad' or something like that and copy/paste 
all of the LoRA model's information into this new file. 

Then save.

Now you have a model card.

You can then go back and fix up the card in Markdown or just copy/paste a model's 
model card if already in Markdown format, directly into the new file.

Now go grab your images by saving them in the 'mymodelnameXYZ' directory along with the model card, and for each image save any generation parameters, CFG, Seed, Prompt, etc...
with the same name as your image, but with the file extension 'txt'.

Example:

myimage29354236.png

Create a text file named 'myimage29354236.txt'

Open it and Copy/Paste the generation parameters into this file.

That's all.

Doing it this way allows the images and model card to be displayed in the Safetensors Model Gallery. Also allows more images to be added.

You can delete this file.