#!/usr/bin/env python3
"""
海面偏振渲染脚本
渲染海面在太阳光照射下的偏振效应
输出 S0(强度), S1, S2, S3 (Stokes分量)
"""

import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from background import *

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 全局变量：存储偏振度为0的像素位置
zero_polarization_pixels = None


def render_ocean_scene(scene_file='scenes/ocean_pol.xml', spp=None):
    """
    渲染海面偏振场景

    参数:
        scene_file: 场景XML文件路径
        spp: 每像素采样数（None则使用XML中的默认值）
    """

    print("=" * 70)
    print("🌊 海面偏振渲染")
    print("=" * 70)

    # 设置偏振变体
    print("\n 设置Mitsuba变体...")
    mi.set_variant('cuda_ad_spectral_polarized')

    print(f" 当前变体: {mi.variant()}")

    # 加载场景
    print(f"\n加载场景: {scene_file}")
    if not os.path.exists(scene_file):
        print(f" 场景文件不存在: {scene_file}")
        return None

    try:
        scene = mi.load_file(scene_file)
        print(" 场景加载成功")
    except Exception as e:
        print(f" 场景加载失败: {e}")
        return None

    # 显示场景信息
    print("\n场景信息:")
    print(f"  积分器: {scene.integrator()}")
    print(f"  相机数量: {len(scene.sensors())}")
    print(f"  光源数量: {len(scene.emitters())}")

    # 渲染
    print("\n🎨 开始渲染...")
    if spp is None:
        print("  使用场景默认采样数")
        image = mi.render(scene)
    else:
        print(f"  每像素采样数: {spp}")
        image = mi.render(scene, spp=spp)

    print("✅ 渲染完成")
    print(f"  图像形状: {image.shape}")

    return scene, image


def analyze_ocean_polarization(scene, image, output_prefix,
                               output_dir):
    """分析并保存海面偏振结果"""

    print("\n💾 原数据分析成像...")

    try:
        bitmap = mi.Bitmap(
            image,
            channel_names=['R', 'G', 'B'] + scene.integrator().aov_names()
        )
    except Exception as e:
        print(f"❌ EXR保存失败: {e}")
        return

    # 分析Stokes分量
    print("\n📊 分析Stokes分量...")

    # 提取通道
    channels = dict(bitmap.split())
    print(f"可用通道: {list(channels.keys())}")

    s0 = np.array(channels['S0'])[:, :, 0]  # 强度   单通道
    s1 = np.array(channels['S1'])[:, :, 0]  # 水平vs垂直偏振
    s2 = np.array(channels['S2'])[:, :, 0]  # 对角偏振
    s3 = np.array(channels['S3'])[:, :, 0]  # 圆偏振

    print(f"RGB三通道偏振分析")
    analyze_rgb_polarization(
        channels,
        output_prefix,
        output_dir,
    )

    return s0, s1, s2, s3


def analyze_rgb_polarization(channels, output_prefix, output_dir):
    """RGB三通道偏振分析"""

    import os
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n🎨 RGB三通道偏振分析...")

    # 提取RGB三通道的Stokes参数
    s0_rgb = np.array(channels['S0'])  # (H, W, 3) - S0的RGB三通道
    s1_rgb = np.array(channels['S1'])  # (H, W, 3) - S1的RGB三通道
    s2_rgb = np.array(channels['S2'])  # (H, W, 3) - S2的RGB三通道
    s3_rgb = np.array(channels['S3'])  # (H, W, 3) - S3的RGB三通道

    calculate_rgb_polarization_angles(
        s0_rgb,
        s1_rgb,
        s2_rgb,
        s3_rgb,
        output_prefix,
        output_dir,
    )


def calculate_rgb_polarization_angles(
    s0_rgb,
    s1_rgb,
    s2_rgb,
    s3_rgb,
    output_prefix='ocean',
    output_dir='E:/project_lw/infrad/pol-mitsuba/sea/final',
):
    """计算RGB三通道的偏振角度强度图"""

    import os
    from PIL import Image

    print(f"\n🔄 计算RGB三通道偏振角度强度图...")

    print(f"  计算各角度强度...")
    i0_rgb = (s0_rgb + s1_rgb) / 2  # 0°强度 RGB
    i45_rgb = (s0_rgb + s2_rgb) / 2  # 45°强度 RGB
    i90_rgb = (s0_rgb - s1_rgb) / 2  # 90°强度 RGB
    i135_rgb = (s0_rgb - s2_rgb) / 2  # 135°强度 RGB

    print(f"  ✅ RGB各角度强度计算完成")
    print(f"    0°强度范围: [{i0_rgb.min():.6f}, {i0_rgb.max():.6f}]")
    print(f"    45°强度范围: [{i45_rgb.min():.6f}, {i45_rgb.max():.6f}]")
    print(f"    90°强度范围: [{i90_rgb.min():.6f}, {i90_rgb.max():.6f}]")
    print(f"    135°强度范围: [{i135_rgb.min():.6f}, {i135_rgb.max():.6f}]")

    # 对s0_rgb原始三通道数据进行(0,1)截断，然后乘255，得到原始S0的三通道RGB数组
    print(f"  🗺️ 处理S0原始RGB数据...")
    s0_rgb_clipped = np.clip(s0_rgb, 0, 1)  # (0,1)截断
    s0_rgb_uint8 = (s0_rgb_clipped * 255.0).astype(np.uint8)  # 乘255并转换为uint8
    print(f"    ✅ S0原始RGB数据已处理，范围: [{s0_rgb_uint8.min()}, {s0_rgb_uint8.max()}] (0-255)")

    # # 保存原始S0 RGB图像
    # s0_img = Image.fromarray(s0_rgb_uint8, mode='RGB')
    # s0_img.save(os.path.join(output_dir, f'{output_prefix}_S0_RGB.png'))
    # print(f"  ✅ S0原始RGB图像已保存: {output_prefix}_S0_RGB.png")

    # 计算各角度强度占总强度的比例
    print(f"  计算各角度偏振比例...")
    ratio_0_rgb = i0_rgb / s0_rgb
    ratio_45_rgb = i45_rgb / s0_rgb
    ratio_90_rgb = i90_rgb / s0_rgb
    ratio_135_rgb = i135_rgb / s0_rgb

    # 计算最终RGB值：比例 × S0原始RGB值
    s0_rgb_float = s0_rgb_uint8.astype(np.float32)
    final_rgb_0 = ratio_0_rgb * s0_rgb_float
    final_rgb_45 = ratio_45_rgb * s0_rgb_float
    final_rgb_90 = ratio_90_rgb * s0_rgb_float
    final_rgb_135 = ratio_135_rgb * s0_rgb_float

    # 限制在0-255范围内（8位）
    final_rgb_0 = np.clip(final_rgb_0, 0, 255).astype(np.uint8)
    final_rgb_45 = np.clip(final_rgb_45, 0, 255).astype(np.uint8)
    final_rgb_90 = np.clip(final_rgb_90, 0, 255).astype(np.uint8)
    final_rgb_135 = np.clip(final_rgb_135, 0, 255).astype(np.uint8)

    # 保存各角度强度图（PNG格式）
    # print(f"  💾 保存各角度偏振强度图（PNG格式）...")
    # img_0_8bit = Image.fromarray(final_rgb_0, mode='RGB')
    # img_0_8bit.save(os.path.join(output_dir, f'{output_prefix}_RGB_0deg_8bit.png'))
    # print(f"    ✅ 0°偏振强度图已保存: {output_prefix}_RGB_0deg_8bit.png")
    #
    # img_45_8bit = Image.fromarray(final_rgb_45, mode='RGB')
    # img_45_8bit.save(os.path.join(output_dir, f'{output_prefix}_RGB_45deg_8bit.png'))
    # print(f"    ✅ 45°偏振强度图已保存: {output_prefix}_RGB_45deg_8bit.png")
    #
    # img_90_8bit = Image.fromarray(final_rgb_90, mode='RGB')
    # img_90_8bit.save(os.path.join(output_dir, f'{output_prefix}_RGB_90deg_8bit.png'))
    # print(f"    ✅ 90°偏振强度图已保存: {output_prefix}_RGB_90deg_8bit.png")
    #
    # img_135_8bit = Image.fromarray(final_rgb_135, mode='RGB')
    # img_135_8bit.save(os.path.join(output_dir, f'{output_prefix}_RGB_135deg_8bit.png'))
    # print(f"    ✅ 135°偏振强度图已保存: {output_prefix}_RGB_135deg_8bit.png")

    # 转换为灰度图并保存
    convert_rgb_to_grayscale(s0_rgb_uint8, final_rgb_0, final_rgb_45,
                             final_rgb_90, final_rgb_135,
                             output_prefix, output_dir)


def convert_16bit_to_32bit(data_16bit):
    """
    将16位数据转换为32位灰度图
    使用标准线性映射: gray_32bit = (gray_16bit / 65535.0) * 4294967295.0
    然后映射到32位范围 (0-4294967295)
    """
    # 使用float64进行计算，确保精度
    data_float = data_16bit.astype(np.float64)
    
    # 映射到32位范围 (0-4294967295)
    # 使用float64和精确的常量，避免精度丢失
    # 4294967295 = 2^32 - 1
    max_32bit = np.float64(4294967295.0)
    max_16bit = np.float64(65535.0)
    
    gray_32bit = (data_float / max_16bit * max_32bit).astype(np.uint32)
    return gray_32bit


def convert_rgb_to_grayscale(s0_rgb, rgb_0, rgb_45, rgb_90, rgb_135,
                             output_prefix='ocean', output_dir='E:/project_lw/infrad/pol-mitsuba/sea/final'):
    """
    将RGB图像转换为16位灰度图并保存（TIFF格式）

    参数:
        s0_rgb: (H, W, 3) S0 RGB图像，uint8格式
        rgb_0: (H, W, 3) 0°偏振RGB图像，uint8格式
        rgb_45: (H, W, 3) 45°偏振RGB图像，uint8格式
        rgb_90: (H, W, 3) 90°偏振RGB图像，uint8格式
        rgb_135: (H, W, 3) 135°偏振RGB图像，uint8格式
        output_prefix: 输出文件名前缀
        output_dir: 输出目录

    输出:
        保存5个16位灰度图（TIFF格式），灰度值范围0-65535
    """
    import os
    from PIL import Image

    print(f"\n🔄 将RGB图像转换为16位灰度图...")

    def rgb_to_grayscale_16bit(rgb_image):
        """
        将RGB图像转换为16位灰度图
        使用标准公式: Gray = 0.299*R + 0.587*G + 0.114*B
        然后映射到16位范围（0-65535）

        参数:
            rgb_image: (H, W, 3) RGB图像，uint8格式

        返回:
            grayscale: (H, W) 灰度图像，uint16格式（0-65535）
        """
        # 确保输入是uint8格式
        if rgb_image.dtype != np.uint8:
            rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

        # 使用标准RGB到灰度转换公式
        # Gray = 0.299*R + 0.587*G + 0.114*B
        weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        grayscale_8bit = np.sum(rgb_image.astype(np.float32) * weights, axis=2)
        grayscale_8bit = np.clip(grayscale_8bit, 0, 255)

        # 映射到16位范围（0-255 → 0-65535）
        # 线性映射：gray_16bit = gray_8bit * (65535 / 255)
        grayscale_16bit = (grayscale_8bit * (65535.0 / 255.0)).astype(np.uint16)

        return grayscale_16bit

    # 转换为16位灰度图
    print(f"  🔄 转换S0 RGB为16位灰度图...")
    s0_gray = rgb_to_grayscale_16bit(s0_rgb)

    print(f"  🔄 转换0°偏振RGB为16位灰度图...")
    rgb_0_gray = rgb_to_grayscale_16bit(rgb_0)

    print(f"  🔄 转换45°偏振RGB为16位灰度图...")
    rgb_45_gray = rgb_to_grayscale_16bit(rgb_45)

    print(f"  🔄 转换90°偏振RGB为16位灰度图...")
    rgb_90_gray = rgb_to_grayscale_16bit(rgb_90)

    print(f"  🔄 转换135°偏振RGB为16位灰度图...")
    rgb_135_gray = rgb_to_grayscale_16bit(rgb_135)

    # 保存16位灰度图（使用TIFF格式，支持16位）
    print(f"  💾 保存16位灰度图（TIFF格式）...")

    # 保存S0灰度图
    s0_gray_img = Image.fromarray(s0_gray, mode='I;16')
    s0_gray_path = os.path.join(output_dir, f'{output_prefix}_S0_grayscale_16bit.tif')
    s0_gray_img.save(s0_gray_path, compression='tiff_deflate')
    print(f"    ✅ S0灰度图已保存: {output_prefix}_S0_grayscale_16bit.tif")
    print(f"       范围: [{s0_gray.min()}, {s0_gray.max()}] (16位)")

    # 保存0°偏振灰度图
    rgb_0_gray_img = Image.fromarray(rgb_0_gray, mode='I;16')
    rgb_0_gray_path = os.path.join(output_dir, f'{output_prefix}_RGB_0deg_grayscale_16bit.tif')
    rgb_0_gray_img.save(rgb_0_gray_path, compression='tiff_deflate')
    print(f"    ✅ 0°偏振灰度图已保存: {output_prefix}_RGB_0deg_grayscale_16bit.tif")
    print(f"       范围: [{rgb_0_gray.min()}, {rgb_0_gray.max()}] (16位)")

    # 保存45°偏振灰度图
    rgb_45_gray_img = Image.fromarray(rgb_45_gray, mode='I;16')
    rgb_45_gray_path = os.path.join(output_dir, f'{output_prefix}_RGB_45deg_grayscale_16bit.tif')
    rgb_45_gray_img.save(rgb_45_gray_path, compression='tiff_deflate')
    print(f"    ✅ 45°偏振灰度图已保存: {output_prefix}_RGB_45deg_grayscale_16bit.tif")
    print(f"       范围: [{rgb_45_gray.min()}, {rgb_45_gray.max()}] (16位)")

    # 保存90°偏振灰度图
    rgb_90_gray_img = Image.fromarray(rgb_90_gray, mode='I;16')
    rgb_90_gray_path = os.path.join(output_dir, f'{output_prefix}_RGB_90deg_grayscale_16bit.tif')
    rgb_90_gray_img.save(rgb_90_gray_path, compression='tiff_deflate')
    print(f"    ✅ 90°偏振灰度图已保存: {output_prefix}_RGB_90deg_grayscale_16bit.tif")
    print(f"       范围: [{rgb_90_gray.min()}, {rgb_90_gray.max()}] (16位)")

    # 保存135°偏振灰度图
    rgb_135_gray_img = Image.fromarray(rgb_135_gray, mode='I;16')
    rgb_135_gray_path = os.path.join(output_dir, f'{output_prefix}_RGB_135deg_grayscale_16bit.tif')
    rgb_135_gray_img.save(rgb_135_gray_path, compression='tiff_deflate')
    print(f"    ✅ 135°偏振灰度图已保存: {output_prefix}_RGB_135deg_grayscale_16bit.tif")
    print(f"       范围: [{rgb_135_gray.min()}, {rgb_135_gray.max()}] (16位)")

    print(f"\n✅ 所有16位灰度图转换完成！")
    print(f"  共保存5个16位灰度图（TIFF格式）:")
    print(f"    - {output_prefix}_S0_grayscale_16bit.tif")
    print(f"    - {output_prefix}_RGB_0deg_grayscale_16bit.tif")
    print(f"    - {output_prefix}_RGB_45deg_grayscale_16bit.tif")
    print(f"    - {output_prefix}_RGB_90deg_grayscale_16bit.tif")
    print(f"    - {output_prefix}_RGB_135deg_grayscale_16bit.tif")
    print(f"  灰度值范围: 0-65535 (16位)")

    # 转换为32位并保存32位TIFF灰度图
    print(f"\n💾 转换为32位并保存32位TIFF灰度图到: {output_dir}")

    # 转换各角度强度图为32位
    s0_gray_32bit = convert_16bit_to_32bit(s0_gray)
    rgb_0_gray_32bit = convert_16bit_to_32bit(rgb_0_gray)
    rgb_45_gray_32bit = convert_16bit_to_32bit(rgb_45_gray)
    rgb_90_gray_32bit = convert_16bit_to_32bit(rgb_90_gray)
    rgb_135_gray_32bit = convert_16bit_to_32bit(rgb_135_gray)

    # 尝试使用tifffile保存32位TIFF（更可靠）
    try:
        import tifffile
        # 保存32位TIFF灰度图
        tifffile.imwrite(os.path.join(output_dir, f'{output_prefix}_S0_grayscale_32bit.tif'), s0_gray_32bit, dtype='uint32')
        tifffile.imwrite(os.path.join(output_dir, f'{output_prefix}_RGB_0deg_grayscale_32bit.tif'), rgb_0_gray_32bit, dtype='uint32')
        tifffile.imwrite(os.path.join(output_dir, f'{output_prefix}_RGB_45deg_grayscale_32bit.tif'), rgb_45_gray_32bit, dtype='uint32')
        tifffile.imwrite(os.path.join(output_dir, f'{output_prefix}_RGB_90deg_grayscale_32bit.tif'), rgb_90_gray_32bit, dtype='uint32')
        tifffile.imwrite(os.path.join(output_dir, f'{output_prefix}_RGB_135deg_grayscale_32bit.tif'), rgb_135_gray_32bit, dtype='uint32')
        print(f"✅ 使用tifffile保存32位TIFF灰度图")
    except ImportError:
        # 如果没有tifffile，尝试使用PIL（可能不支持32位，会降级处理）
        try:
            # PIL的mode='I'是32位有符号整数，对于无符号整数需要特殊处理
            # 将uint32转换为int32（会丢失最高位，但通常不会用到）
            # 或者直接让PIL自动处理
            Image.fromarray(s0_gray_32bit.astype(np.int32), mode='I').save(os.path.join(output_dir, f'{output_prefix}_S0_grayscale_32bit.tif'), compression='tiff_deflate')
            Image.fromarray(rgb_0_gray_32bit.astype(np.int32), mode='I').save(os.path.join(output_dir, f'{output_prefix}_RGB_0deg_grayscale_32bit.tif'), compression='tiff_deflate')
            Image.fromarray(rgb_45_gray_32bit.astype(np.int32), mode='I').save(os.path.join(output_dir, f'{output_prefix}_RGB_45deg_grayscale_32bit.tif'), compression='tiff_deflate')
            Image.fromarray(rgb_90_gray_32bit.astype(np.int32), mode='I').save(os.path.join(output_dir, f'{output_prefix}_RGB_90deg_grayscale_32bit.tif'), compression='tiff_deflate')
            Image.fromarray(rgb_135_gray_32bit.astype(np.int32), mode='I').save(os.path.join(output_dir, f'{output_prefix}_RGB_135deg_grayscale_32bit.tif'), compression='tiff_deflate')
            print(f"⚠️ 使用PIL保存32位TIFF（注意：PIL使用有符号整数，最大值可能受限）")
        except Exception as e:
            print(f"⚠️ 无法保存32位TIFF: {e}")
            print(f"  建议安装tifffile库: pip install tifffile")

    print(f"\n✅ 已保存5个32位TIFF灰度图:")
    print(f"    - {output_prefix}_S0_grayscale_32bit.tif")
    print(f"    - {output_prefix}_RGB_0deg_grayscale_32bit.tif")
    print(f"    - {output_prefix}_RGB_45deg_grayscale_32bit.tif")
    print(f"    - {output_prefix}_RGB_90deg_grayscale_32bit.tif")
    print(f"    - {output_prefix}_RGB_135deg_grayscale_32bit.tif")
    print(f"  转换公式: gray_32bit = (gray_16bit / 65535.0) * 4294967295.0")
    print(f"  灰度值范围: 0-4294967295 (32位)")


def save_grayscale_images(s0, s1, s2, s3, dop, output_prefix='ocean',
                          output_dir='E:/project_lw/infrad/pol-mitsuba/sea/final'):
    """保存单独的PNG图像 - 使用原值，图例显示真实数值"""

    import os
    os.makedirs(output_dir, exist_ok=True)

    print("\n💾 保存原值图像（不归一化）...")

    # S0 强度（原值，灰度或彩色）
    fig, ax = plt.subplots(figsize=(10, 8))
    im0 = ax.imshow(s0, cmap='viridis')  # 使用彩色图例更清晰
    ax.set_title(f'S0: 强度（原值）\n范围: [{s0.min():.6f}, {s0.max():.6f}]', size=14, weight='bold')
    ax.axis('off')
    cbar = plt.colorbar(im0, ax=ax, fraction=0.046)
    cbar.set_label('S0 原值', rotation=270, labelpad=20, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_prefix}_S0_intensity.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # S1: 原值红蓝映射
    fig, ax = plt.subplots(figsize=(10, 8))
    s1_max = max(abs(s1.min()), abs(s1.max()))
    im1 = ax.imshow(s1, cmap='RdBu_r', vmin=-s1_max, vmax=s1_max)
    ax.set_title(f'S1: 水平 vs 垂直偏振（原值）\n范围: [{s1.min():.6f}, {s1.max():.6f}]', size=14, weight='bold')
    ax.axis('off')
    cbar = plt.colorbar(im1, ax=ax, fraction=0.046)
    cbar.set_label('S1 原值', rotation=270, labelpad=20, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_prefix}_S1_polarization.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # S2: 原值红蓝映射
    fig, ax = plt.subplots(figsize=(10, 8))
    s2_max = max(abs(s2.min()), abs(s2.max()))
    im2 = ax.imshow(s2, cmap='RdBu_r', vmin=-s2_max, vmax=s2_max)
    ax.set_title(f'S2: 对角偏振 (±45°)（原值）\n范围: [{s2.min():.6f}, {s2.max():.6f}]', size=14, weight='bold')
    ax.axis('off')
    cbar = plt.colorbar(im2, ax=ax, fraction=0.046)
    cbar.set_label('S2 原值', rotation=270, labelpad=20, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_prefix}_S2_diagonal.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # S3: 原值红蓝映射
    fig, ax = plt.subplots(figsize=(10, 8))
    s3_max = max(abs(s3.min()), abs(s3.max()))
    im3 = ax.imshow(s3, cmap='RdBu_r', vmin=-s3_max, vmax=s3_max)
    ax.set_title(f'S3: 圆偏振 (左/右)（原值）\n范围: [{s3.min():.6f}, {s3.max():.6f}]', size=14, weight='bold')
    ax.axis('off')
    cbar = plt.colorbar(im3, ax=ax, fraction=0.046)
    cbar.set_label('S3 原值', rotation=270, labelpad=20, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_prefix}_S3_circular.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # DOP: 本身就在[0,1]范围
    fig, ax = plt.subplots(figsize=(10, 8))
    im_dop = ax.imshow(dop, cmap='hot', vmin=0, vmax=1)
    ax.set_title(f'偏振度 (DOP)\n范围: [0, 1]', size=14, weight='bold')
    ax.axis('off')
    cbar = plt.colorbar(im_dop, ax=ax, fraction=0.046)
    cbar.set_label('DOP', rotation=270, labelpad=20, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_prefix}_DOP.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ 已保存5个原值PNG图像到: {output_dir}")
    print(f"   S0范围: [{s0.min():.6f}, {s0.max():.6f}]")
    print(f"   S1范围: [{s1.min():.6f}, {s1.max():.6f}]")
    print(f"   S2范围: [{s2.min():.6f}, {s2.max():.6f}]")
    print(f"   S3范围: [{s3.min():.6f}, {s3.max():.6f}]")
    print(f"   图例显示真实数值范围 ✅")

    # 额外保存带原值范围标注的彩色图像
    save_original_value_images(s0, s1, s2, s3, dop, output_prefix, output_dir)


def save_original_value_images(s0, s1, s2, s3, dop, output_prefix='ocean',
                               output_dir='E:/project_lw/infrad/pol-mitsuba/sea/final'):
    """保存带原值范围标注的彩色图像"""

    import os
    os.makedirs(output_dir, exist_ok=True)

    print("\n🎨 保存原值彩色图像...")

    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('海面偏振渲染结果 - 原值显示', size=16, weight='bold')

    # S0: 强度（原值，彩色图）
    ax = axes[0, 0]
    im0 = ax.imshow(s0, cmap='viridis', vmin=s0.min(), vmax=s0.max())
    ax.set_title(f'S0: 强度（原值）\n范围: [{s0.min():.6f}, {s0.max():.6f}]', size=12, weight='bold')
    ax.axis('off')
    cbar0 = plt.colorbar(im0, ax=ax, fraction=0.046)
    cbar0.set_label('S0 原值', rotation=270, labelpad=15)

    # S1: 水平vs垂直偏振（红蓝映射）
    ax = axes[0, 1]
    s1_max = max(abs(s1.min()), abs(s1.max()))
    im1 = ax.imshow(s1, cmap='RdBu_r', vmin=-s1_max, vmax=s1_max)
    ax.set_title(f'S1: 水平 vs 垂直偏振\n范围: [{s1.min():.3f}, {s1.max():.3f}]', size=12, weight='bold')
    ax.axis('off')
    cbar1 = plt.colorbar(im1, ax=ax, fraction=0.046)
    cbar1.set_label('S1 原值', rotation=270, labelpad=15)

    # S2: 对角偏振
    ax = axes[0, 2]
    s2_max = max(abs(s2.min()), abs(s2.max()))
    im2 = ax.imshow(s2, cmap='RdBu_r', vmin=-s2_max, vmax=s2_max)
    ax.set_title(f'S2: 对角偏振 (±45°)\n范围: [{s2.min():.3f}, {s2.max():.3f}]', size=12, weight='bold')
    ax.axis('off')
    cbar2 = plt.colorbar(im2, ax=ax, fraction=0.046)
    cbar2.set_label('S2 原值', rotation=270, labelpad=15)

    # S3: 圆偏振
    ax = axes[1, 0]
    s3_max = max(abs(s3.min()), abs(s3.max()))
    im3 = ax.imshow(s3, cmap='RdBu_r', vmin=-s3_max, vmax=s3_max)
    ax.set_title(f'S3: 圆偏振 (左/右)\n范围: [{s3.min():.3f}, {s3.max():.3f}]', size=12, weight='bold')
    ax.axis('off')
    cbar3 = plt.colorbar(im3, ax=ax, fraction=0.046)
    cbar3.set_label('S3 原值', rotation=270, labelpad=15)

    # 偏振度
    ax = axes[1, 1]
    im4 = ax.imshow(dop, cmap='hot', vmin=0, vmax=1)
    ax.set_title(f'偏振度 (DOP)\n范围: [{dop.min():.3f}, {dop.max():.3f}]', size=12, weight='bold')
    ax.axis('off')
    cbar4 = plt.colorbar(im4, ax=ax, fraction=0.046)
    cbar4.set_label('DOP', rotation=270, labelpad=15)

    # 原值统计信息
    ax = axes[1, 2]
    ax.axis('off')

    # 创建统计信息文本
    stats_text = f"""原值统计信息:

S0 (强度):
  范围: [{s0.min():.6f}, {s0.max():.6f}]
  均值: {s0.mean():.6f}
  标准差: {s0.std():.6f}

S1 (水平vs垂直):
  范围: [{s1.min():.6f}, {s1.max():.6f}]
  均值: {s1.mean():.6f}
  标准差: {s1.std():.6f}

S2 (对角):
  范围: [{s2.min():.6f}, {s2.max():.6f}]
  均值: {s2.mean():.6f}
  标准差: {s2.std():.6f}

S3 (圆偏振):
  范围: [{s3.min():.6f}, {s3.max():.6f}]
  均值: {s3.mean():.6f}
  标准差: {s3.std():.6f}

DOP (偏振度):
  范围: [{dop.min():.6f}, {dop.max():.6f}]
  均值: {dop.mean():.6f}
  标准差: {dop.std():.6f}"""

    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()

    # 保存图像
    output_file = os.path.join(output_dir, f'{output_prefix}_original_values.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ 原值彩色图像已保存: {output_file}")

    # plt.show()


def main():
    """主函数"""

    input_xml = r'E:\project_lw\infrad\pol-mitsuba\sea\scenes-visible'
    background = ["sea_background1", "sea_background2", "sea_background3", "sea_background4"]
    level = ["Level_55.xml", "Level_35.xml", "Level_20.xml", "Level_5.xml"]

    base_output_dir = r'E:/project_lw/infrad/pol-mitsuba/sea/final/Visible'

    import re
    import os

    # 双重循环处理所有background和level的组合
    total_combinations = len(background) * len(level)
    current_combination = 0

    print("=" * 70)
    print(f"🔄 开始批量处理: {total_combinations} 个组合")
    print("=" * 70)

    for bg in background:
        for lvl in level:
            current_combination += 1
            print("\n" + "=" * 70)
            print(f"📦 处理组合 {current_combination}/{total_combinations}: {bg} × {lvl}")
            print("=" * 70)

            # 组合scene_file路径
            scene_file = os.path.join(input_xml, bg, lvl)

            # 从level文件名中提取角度（去掉.xml后缀）
            level_name = lvl.replace('.xml', '')  # 例如: "Level_55"
            scene_match = re.search(r'Level_(\d+)', level_name)

            # 组合output_dir路径（不带.xml）
            if scene_match:
                angle = int(scene_match.group(1))
                output_dir = os.path.join(base_output_dir, bg, f'Level-{angle}')
            else:
                output_dir = os.path.join(base_output_dir, bg, level_name)

            print(f"\n📁 场景文件: {scene_file}")
            print(f"📁 输出目录: {output_dir}")
            print("  - OBJ海浪网格模型")
            print("  - 太阳光源（directional）")
            print("  - 水面材质（roughdielectric）")

            # 渲染
            result = render_ocean_scene(scene_file)

            if result is None:
                print(f"❌ 跳过组合 {current_combination}/{total_combinations}: 场景加载失败")
                continue

            scene, image = result

            # 分析
            output_prefix = 'sea'
            s0, s1, s2, s3 = analyze_ocean_polarization(
                scene,
                image,
                output_prefix,
                output_dir,
            )

            print(f"\n✅ 完成组合 {current_combination}/{total_combinations}: {bg} × {lvl}")

    print("\n" + "=" * 70)
    print(f"🎉 批量处理完成！共处理 {total_combinations} 个组合")
    print("=" * 70)
    print("\n生成的文件:")
    print(f"  - ocean_RGB_S0_RGB.png (S0原始RGB图像)")
    print(f"  - ocean_RGB_RGB_*deg_8bit.png (各角度偏振强度图，PNG格式)")
    print(f"  - ocean_RGB_*_grayscale_16bit.tif (16位灰度图，TIFF格式)")
    print(f"  - ocean_RGB_*_grayscale_32bit.tif (32位灰度图，TIFF格式)")


if __name__ == "__main__":
    main()

