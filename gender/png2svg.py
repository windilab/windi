import base64 # 標準ライブラリ
import svgwrite


# 変換前画像ファイルパス
input_file = ''

# 変換後SVGファイルパス
output_file = input_file.split(".")[0] + '.svg'


# 変換前画像ファイルを開く
with open(input_file, "rb") as f:
    img = base64.b64encode(f.read())

# 変換後ファイルを書き込む準備
dwg = svgwrite.Drawing(output_file)

# 保存画像のサイズ等指定と書き込み
dwg.add(dwg.image('data:image/png;base64,' + img.decode("ascii"),
                  size=(500, 500)
                 )
        )

# ファイル保存
dwg.save()