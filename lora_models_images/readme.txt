Any LoRA Models downloaded from the internet normally have sample generated images, 
plus a model card etc... that you can also download as well as the model itself.

That information goes in this directory, structured as so...

lcmlora_models_images
├── mymodelnameabc                 <---folder for model card and images
│   ├── mymodelnameabc.md          <---markdown language model card
│   ├── image3124141.jpg           <---sample image
│   └── image3124141.txt           <---sample image generation text file
├── mymodelname123                 <---folder for model card and images
│   ├── mymodelname123.md          <---markdown language model card
│   ├── image1.png
│   ├── image1.txt
│   ├── image2.png                 <---sample image
│   ├── image2.txt                 <---sample image generation text file
│   ├── image3.png
│   └── image3.txt
└── mymodelnameXYZ                 <---folder for model card and images
    ├── mymodelnameXYZ.md          <---markdown language model card
    ├── image1.jpg
    ├── image1.txt
    ├── image2.webp
    └── image2.txt
    

Example: 

You downloaded a LoRA model named 'mymodelnameXYZ.safetensors' already, and saved it in the directory : 'lora_models'.

Now you want to grab the information about the LoRA model.

Create a directory in the directory named 'lcmlora_models_images' named: 'mymodelnameXYZ'

Inside of that directory create a a file named: 'mymodelnameXYZ.md'

Open that with a right click 'Open with Notepad' or something like that and copy/paste 
all of the LoRA model's information into this new file. 

Then save.

Now you have a model card.

You can go back and fix up the card into Markdown or just copy/paste a model's 
model card if already in Markdown format, directly into the new file.


You can delete this file.