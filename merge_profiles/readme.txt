Profiles for Merge Models go in here.

Note: '*.json' files only.

Hint: Manually editing JSON files:

If they end up looking like the one below, it'll work if ya got good weights (numbers)

We use only the first 25 Weights for SD1.5
We use all 40 weights for SDXL.

The JSON file MUST have ALL 40 weights.
Regardless as to if it is an SD1.5 or SDXL profile.

Just set the ones beyond the SD1.5 values to '0.5' when using SD1.5.

JSON File Definitions:
----------------------------

model_type: SD1.5 | SDXL
description: "Quoted Text"
text_alpha: float (0.00-1.00)
vae_alpha: float (0.00-1.00)
weights: list of weight values * 40 (float) (0.00-1.00)

Here is a sample Merge Profile (JSON)

----------cut here---------------------

{
    "model_type": "SD1.5",
    "description": "Keeps foreground details from A intact while pulling environmental variables from B.",
	"text_alpha": 0.5,
	"vae_alpha": 0.5,
    "weights": [
        0,
        0,
        0,
        0,
        0,
        0,
        0.1,
        0.3,
        0.6,
        0.8,
        0.9,
        0.9,
        1,
        0.7,
        0.3,
        0.1,
        0,
        0,
        0,
        0.1,
        0.4,
        0.7,
        0.9,
        1,
        1,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5
    ]
}

----------cut here---------------------




You can delete this file.