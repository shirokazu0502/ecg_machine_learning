import numpy as np
import pyvista as pv
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt

# ==========================================
# 1. 形状データの準備（トルソー・メッシュ）
# ==========================================
# 実際の研究では、MRIやCTから作成した被験者固有のSTL/VTKファイルを読み込みます。
# 例: torso_mesh = pv.read('subject_torso.stl')

# ここではデモ用に、PyVistaで楕円体を生成してトルソーに見立てます。
# x, y方向に少し潰し、z方向（身長方向）に伸ばします。
torso_mesh = pv.ParametricEllipsoid(
    xradius=0.3, yradius=0.2, zradius=0.5, u_res=50, v_res=50
)
# 扱いやすいようにPolyData形式に変換し、位置を調整
torso_mesh = pv.PolyData(torso_mesh)
torso_mesh.translate([0, 0, 0.5], inplace=True)  # 足元をz=0付近に

# メッシュの全頂点座標を取得（補間のターゲット点となります）
mesh_points = torso_mesh.points
mx, my, mz = mesh_points[:, 0], mesh_points[:, 1], mesh_points[:, 2]


# ==========================================
# 2. 電極位置と電位データの準備（ダミーデータ）
# ==========================================
# 実際の研究では、体表測定した電極の3D座標と、ある時刻の電位値を使用します。

# デモ用に、トルソー表面からランダムに64点を「電極位置」として選びます。
np.random.seed(42)  # 再現性のため
num_electrodes = 64
electrode_indices = np.random.choice(torso_mesh.n_points, num_electrodes, replace=False)
electrode_points = mesh_points[electrode_indices]
ex, ey, ez = electrode_points[:, 0], electrode_points[:, 1], electrode_points[:, 2]


# デモ用に「電位データ」を生成します。
# ここでは単純なダイポール（双極子）のようなパターンを作ります。
# 前胸部上部をプラス、下部をマイナスにしてみます。
def generate_dummy_potential(x, y, z):
    # zが高いほどプラス、yが前側(ポジティブ)ほど強い、のような単純な関数
    pot = (z - 0.5) * 2.0 + (y * 0.5)
    # 少しノイズを加える
    pot += np.random.normal(0, 0.1, size=pot.shape)
    return pot


electrode_potentials = generate_dummy_potential(ex, ey, ez)


# ==========================================
# 3. 空間補間 (Interpolation) - 重要！
# ==========================================
# 限られた電極点のデータを、滑らかな曲面データに変換します。
# 生体信号の補間には「動径基底関数 (RBF: Radial Basis Function)」がよく用いられます。

# SciPyのRbfを使って補間関数を作成
# 'multiquadric'や'gaussian'が滑らかな曲面に向いています。
rbf_interpolator = Rbf(
    ex, ey, ez, electrode_potentials, function="multiquadric", epsilon=0.1
)

# メッシュの全頂点座標に対して電位を推定（補間）
interpolated_potentials = rbf_interpolator(mx, my, mz)

# 補間されたデータをメッシュに追加
torso_mesh["Potentials"] = interpolated_potentials


# ==========================================
# 4. 可視化と論文用画像保存 (PyVista)
# ==========================================

# --- 設定 ---
# カラーマップの選択: 電位図では「赤(正)-白(ゼロ)-青(負)」のような発散型が一般的です。
# matplotlibの'bwr' (blue-white-red) や 'seismic' が適しています。
colormap = "bwr"

# 電位の最大最小を対称にして、0が中心（白）に来るように設定します。
v_max = np.max(np.abs(interpolated_potentials))
clim = [-v_max, v_max]

# --- プロッターの準備 ---
# オフスクリーン（画面にウィンドウを出さない）で高解像度レンダリング設定
plotter = pv.Plotter(off_screen=True, window_size=[1024, 768])
plotter.set_background("white")  # 論文用は背景白が基本

# メッシュの追加
plotter.add_mesh(
    torso_mesh,
    scalars="Potentials",  # 表示するデータ名
    cmap=colormap,
    clim=clim,  # カラーバーの範囲
    smooth_shading=True,  # 表面を滑らかに表示
    show_edges=False,  # メッシュの線を消す
    scalar_bar_args={  # カラーバーの設定
        "title": "Potential (mV)",
        "color": "black",  # 文字色
        "vertical": True,  # 縦置き
        "position_x": 0.85,
        "position_y": 0.1,
    },
)

# (オプション) 電極位置を黒い点で表示する場合
# plotter.add_points(electrode_points, color='black', point_size=5, render_points_as_spheres=True)


# --- アングル設定と保存 (前面図) ---
plotter.view_xz()  # 前面からのビューに設定 (座標系によって調整が必要)
plotter.camera.azimuth += 180  # 必要に応じて回転
plotter.camera.elevation += 10  # 少し上から見下ろすなど調整

# 画像を保存 (PNG形式, 高解像度)
output_filename = "BSPM_anterior_view.png"
plotter.screenshot(output_filename, scale=2)  # scale=2で2倍の解像度で保存
print(f"前面図を保存しました: {output_filename}")

# --- アングル設定と保存 (背面図) ---
# 同じプロッターで視点だけ変えて保存
plotter.view_xz()
# azimuthを調整して後ろに回す
plotter.camera.azimuth += 0
output_filename_back = "BSPM_posterior_view.png"
plotter.screenshot(output_filename_back, scale=2)
print(f"背面図を保存しました: {output_filename_back}")

# メモリ解放
plotter.close()

print("完了。")
