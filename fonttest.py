#!/usr/bin/env python3
"""
日本語フォント自動設定スクリプト
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def find_japanese_font():
    """
    システムから日本語フォントを自動検出
    """
    # 優先順位順に日本語フォント名リスト
    preferred_fonts = [
        'MS Gothic', 'MS PGothic', 'MS UI Gothic',
        'Yu Gothic', 'Yu Gothic UI', 'YuGothic',
        'Meiryo', 'Meiryo UI',
        'IPAexGothic', 'IPAGothic',
        'TakaoPGothic', 'VL PGothic',
        'Hiragino Sans', 'Hiragino Kaku Gothic Pro',
        'AppleGothic'
    ]
    
    # システムの全フォントを取得
    available_fonts = set([f.name for f in fm.fontManager.ttflist])
    
    # 使用可能な日本語フォントを探す
    for font in preferred_fonts:
        if font in available_fonts:
            print(f"✓ 日本語フォント検出: {font}")
            return font
    
    # 見つからない場合は部分一致で探す
    for font in available_fonts:
        if any(keyword in font for keyword in ['Gothic', 'ゴシック', 'Mincho', '明朝']):
            print(f"✓ 日本語フォント検出（部分一致）: {font}")
            return font
    
    print("⚠ 警告: 日本語フォントが見つかりませんでした")
    print("  英語で出力されます")
    return None

# フォント設定
japanese_font = find_japanese_font()

if japanese_font:
    plt.rcParams['font.family'] = japanese_font
else:
    plt.rcParams['font.family'] = 'DejaVu Sans'

plt.rcParams['axes.unicode_minus'] = False

# テスト描画
import numpy as np

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([1, 2, 3], [1, 4, 2], marker='o', label='テストデータ')
ax.set_xlabel('横軸（日本語テスト）')
ax.set_ylabel('縦軸（日本語テスト）')
ax.set_title('日本語フォントテスト')
ax.legend()
ax.grid(True, alpha=0.3)

plt.savefig('font_test.png', dpi=150, bbox_inches='tight')
print(f"\n✓ テスト画像生成: font_test.png")
print("  この画像で日本語が正しく表示されているか確認してください")

plt.close()