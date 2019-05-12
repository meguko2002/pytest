import cv2
import numpy as np

dt = 500/1000/1000  #2kHz　【注意】パターン周期よりもdtが十分小さいこと


def scanbmp(file , dt):
    img = cv2.imread(file, 0)
    #画像をスライス
    img = img[6000:16000, :]

    # 各rowsについて、columnの値を平均してwaveに代入
    wave = np.mean(img, axis=1)
    # 画像パターンで白からに変化するエッジの間隔を格納
    edgewave = makeedgewace(wave)
    M, = edgewave.shape        #パターン個数を計算
    p_ave = np.mean(edgewave)  #パターン間隔の平均index数
    dpitch = p_ave * dt        #平均パターン間隔(=サンプリング周期)[sec]
    print('dpitch = {}'.format(dpitch))
    t = np.arange(0, M*dpitch, dpitch)  #時間軸設定
    edgewave = edgewave - p_ave   #ピッチずれ量を平均値ゼロに正規化
    F = np.fft.fft(edgewave)
    F_abs = np.abs(F)
    F_abs_amp = F_abs / M *2
    F_abs_amp[0] = F_abs_amp[0] /2
    fq = np.linspace(0, 1.0/dpitch, M)  #周波数軸設定
    return t, wave, edgewave, fq, F_abs_amp


def makeedgewave(wave, thd=150):  # thd:白黒の閾値(0~255)
    dotspace =[]
    pre_data = 255    # ITBに何も書かれていないと白(200付近)だから
    for i, data in enumerate(wave):
        if data <= thd and pre_data>thd:  # パターン(2dot3space)のエッジのindexを格納
            dotspace.append(i)
    array_dotspace = np.array(dotspace)  #listをndarrayに変換
    array_dotspace = np.diff(array_dotspace, n=1)  #要素の差分（パターン間隔）を格納
    return array_dotspace