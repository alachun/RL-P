from PIL import Image
import os

def merge_images(image_paths, output_path, rows=2, cols=5):
    # 确保提供的图片数量与预期一致
    assert rows * cols == len(image_paths), "图片数量与行列数不匹配"

    # 创建一个新的空白图片，大小为所有图片合并后的尺寸
    images = [Image.open(path) for path in image_paths]
    image_width, image_height = images[0].size
    merged_image = Image.new('RGB', (image_width * cols, image_height * rows))

    # 将每张图片粘贴到合适的位置
    for index, image in enumerate(images):
        row = index // cols
        col = index % cols
        location = (col * image_width, row * image_height)
        merged_image.paste(image, location)

    # 保存合并后的图片
    merged_image.save(output_path)

dfa = 2
# PNG图片地址列表，这里假设图片地址已经以某种方式获取并存储在列表中
png_image_paths = [
'fig/pfile05_dfa' + str(dfa) + '.png',
'fig/pfile05-2_dfa' + str(dfa) + '.png',
'fig/pfile06-2_dfa' + str(dfa) + '.png',
'fig/pfile07-2_dfa' + str(dfa) + '.png',
'fig/pfile08_dfa' + str(dfa) + '.png',
'fig/pfile08-2_dfa' + str(dfa) + '.png',
'fig/pfile09_dfa' + str(dfa) + '.png',
'fig/pfile09-2_dfa' + str(dfa) + '.png',
'fig/pfile10_dfa' + str(dfa) + '.png',
'fig/pfile10-2_dfa' + str(dfa) + '.png',

]

# 输出路径
output_path = 'fig/child_snack_dfa2.png'

# 调用函数进行图片合并
merge_images(png_image_paths, output_path)