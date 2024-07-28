from paligemma import PaliGemma

# Initialize the PaliGemma model
model = PaliGemma()

# Define the text prompt and image URL
text_prompt = "A beautiful sunset over the ocean."
image_url = "https://i.sstatic.net/YAwqo.png"

# Run the PaliGemma model
output = model.run(text_prompt, image_url)
# Print the generated output
print("DEBUG:"+output)