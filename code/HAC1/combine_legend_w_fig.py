from PIL import Image

fig_path = "fig/citycar_dfa2.png"
# 打开两张图像
img1 = Image.open(fig_path)  # 替换为你的第一张图片路径
img2 = Image.open("legend.png")  # 替换为你的第二张图片路径

# 确保两张图像的宽度相同（如果不同，可以调整大小）
if img1.width != img2.width:
    img2 = img2.resize((img1.width, int(img2.height * img1.width / img2.width)))

# 计算合并后图像的尺寸
new_width = img1.width
new_height = img1.height + img2.height

# 创建一个新的空白图像（白色背景）
combined_img = Image.new("RGB", (new_width, new_height), color=(255, 255, 255))

# 粘贴两张图像到新图像上
combined_img.paste(img1, (0, 0))  # 将第一张图像粘贴到顶部
combined_img.paste(img2, (0, img1.height))  # 将第二张图像粘贴到下面

# 保存或显示合并后的图像
combined_img.save(fig_path)  # 保存结果
combined_img.show()  # 显示结果