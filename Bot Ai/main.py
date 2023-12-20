from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np


import discord
from discord.ext import commands

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='$', intents=intents)

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

@bot.command()
async def img(ctx):
    if ctx.message.attachments:
        for attachment in ctx.message.attachments:
            file_name = attachment.filename
            file_url = attachment.url
            await attachment.save(f"./img/{attachment.filename}")
            await ctx.send("Si hay un archivo adjunto "+  str(attachment.filename))


            model = load_model("keras_model.h5", compile=False)
            class_names = open("labels.txt", "r").readlines()
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            image = Image.open(f"./img/{file_name}").convert("RGB")
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
            data[0] = normalized_image_array
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]
            await ctx.send(f"Class: {class_name[2:]}")
            await ctx.send(f"Confidence Score: {confidence_score}")


    else:
        print("No hay un archivo adjunto")
        await ctx.send("No hay un archivo adjunto")

@bot.command()
async def hello(ctx):
    await ctx.send(f'Hi! I am a bot {bot.user}!')

@bot.command()
async def heh(ctx, count_heh = 5):
    await ctx.send("he" * count_heh)



bot.run("Token")